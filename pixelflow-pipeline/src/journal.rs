//! The one research-journal writer.
//!
//! A journal line is a claim about a run: what config produced it, what
//! numbers came out. Every journal-writing binary has its own record schema
//! (the fields genuinely differ; that stays), but all need the *same* append
//! mechanics: serialize
//! to one JSON line, create the parent directory if it's missing, refuse to
//! corrupt an unsmudged Git LFS pointer, append-or-panic. That mechanism was
//! two copies (one of them missing the LFS guard) before this module
//! (docs/plans/2026-08-17-cost-model-domain.md, J15) — a fix to one could
//! silently fail to reach the other, the same NO-SILENT-FAILURES class as
//! J14's `run_scalar`.
//!
//! The mechanics above are the writer's easy half. Its other job — owning the
//! *shape* every journal line takes, not just how the bytes land — used to be
//! absent: the (since-deleted) extraction-policy bench hand-rolled a
//! `config_hash` from source revision, corpus bytes, weights bytes, protocol
//! params, and the build environment, entirely inside that one binary; the
//! extraction-head trainer wrote a journal line with none of that provenance
//! at all — two runs of it against different corpora or different commits
//! were indistinguishable.
//! [`ConfigHash`], [`ArtifactId`], and [`JournalEntry`] below are that shared
//! shape (docs/plans/2026-08-17-cost-model-domain.md, J15): every
//! journal-writing binary composes a `ConfigHash` the same way, names its
//! weights the same way, and wraps its own metrics in the same envelope, so
//! two binaries can no longer diverge on what "this run's provenance" means.
use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

/// Re-exported so journal-writing binaries need one `use` for both the
/// writer mechanics and the content-hash primitive their `ConfigHash` inputs
/// (corpus identity, weights identity, diff hash) are built from.
pub use crate::schema::fnv1a64_hex;

/// Refuse to append to a journal that is still a Git LFS pointer.
///
/// `.gitattributes` sends every `*.jsonl` through LFS, so in a clone where the
/// objects were never pulled, the journal path holds the three-line pointer
/// stub instead of the journal. Appending there produces a file that is
/// neither — malformed JSONL behind a pointer header — and staging it can
/// commit that text as the tracked payload, taking the real history with it.
/// A run that got this far has already spent its measurements, so say
/// exactly what to run rather than failing in a way that reads as a corrupt
/// journal.
fn refuse_unsmudged_lfs_pointer(path: &Path) {
    const POINTER_MAGIC: &str = "version https://git-lfs.github.com/spec/";
    let Ok(head) = fs::read(path) else {
        return; // Absent is fine — the append creates it.
    };
    // A pointer stub is a few hundred bytes; a real journal's first line is a
    // JSON object. Only the prefix needs looking at.
    let prefix = String::from_utf8_lossy(&head[..head.len().min(POINTER_MAGIC.len())]);
    assert!(
        prefix != POINTER_MAGIC,
        "{} is an unsmudged Git LFS pointer, not the journal.\n\
         Appending would corrupt it. Materialize it first:\n  \
         git lfs install && git lfs pull --include {}",
        path.display(),
        path.display()
    );
}

/// Serialize `record` to one JSON line and append it to the journal at
/// `path`, creating the parent directory if needed. Every failure — encode,
/// mkdir, open, write — is a hard panic naming the path: a journal line that
/// silently didn't land is a run whose claim can never be checked again.
pub fn append_record<T: Serialize>(path: &Path, record: &T) {
    let line = serde_json::to_string(record).unwrap_or_else(|e| {
        panic!(
            "failed to serialize journal record for {}: {e}",
            path.display()
        )
    });
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", parent.display()));
    }
    refuse_unsmudged_lfs_pointer(path);
    let mut journal = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", path.display()));
    writeln!(journal, "{line}")
        .unwrap_or_else(|e| panic!("failed to append to {}: {e}", path.display()));
    eprintln!("Journal appended: {}", path.display());
}

// ── Provenance: ConfigHash, ArtifactId, JournalEntry (J15) ─────────────────

/// Paths a config hash must not vary with: a run WRITES into these
/// (`pixelflow-pipeline/data/` holds this run's own JSONL output;
/// `docs/results/` holds the journal this line is about to join), so
/// including them in a dirty-tree diff would make the hash depend on the
/// run's own prior output — a second invocation of an otherwise-unchanged
/// tree would then hash differently from the first, the opposite of an
/// identity.
///
/// The `top,` magic is load-bearing, not decoration: [`git`] runs with
/// `current_dir` set to this crate's own manifest directory
/// (`.../pixelflow-pipeline`), so a bare `:(exclude)pixelflow-pipeline/data`
/// is resolved relative to THAT directory — i.e. as
/// `pixelflow-pipeline/pixelflow-pipeline/data`, which never exists — and
/// silently excludes nothing. `:(top,exclude)` anchors the pathspec to the
/// repository root regardless of the process's cwd, which is what a
/// repo-root-relative path here was always meant to mean.
pub const DIFF_HASH_EXCLUDES: [&str; 2] = [
    ":(top,exclude)pixelflow-pipeline/data",
    ":(top,exclude)docs/results",
];

/// Raw, untrimmed stdout on success. Deliberately not trimmed here: `diff
/// HEAD` output is hashed byte-for-byte by [`SourceVersion::current`] (see
/// [`SourceVersion::diff_hash`]), and a `.trim()` in this shared helper would
/// have silently dropped trailing-whitespace-only differences from every
/// caller's diff hash, aliasing two distinct working trees onto one
/// `diff_hash` — defeating the provenance guarantee the journal exists for.
/// Callers that want a trimmed single-line answer (`rev-parse HEAD`,
/// `status --porcelain`) trim at their own call site instead.
fn git(args: &[&str]) -> Result<String, String> {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    match Command::new("git")
        .args(args)
        .current_dir(manifest_dir)
        .output()
    {
        Ok(out) if out.status.success() => Ok(String::from_utf8_lossy(&out.stdout).into_owned()),
        Ok(out) => Err(format!(
            "`git {}` exited {}: {}",
            args.join(" "),
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        )),
        Err(e) => Err(format!("failed to spawn git: {e}")),
    }
}

/// What source revision a run's numbers came off.
///
/// `rev` alone is not enough: every uncommitted variant of the same HEAD
/// collapses onto one `"<sha>-dirty"` string, so two working trees measuring
/// different code would hash identically and produce indistinguishable
/// journal lines. `diff_hash` restores the distinction by hashing the actual
/// uncommitted diff over tracked source (excluding [`DIFF_HASH_EXCLUDES`]).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SourceVersion {
    /// `git rev-parse HEAD`, suffixed `-dirty` on an unclean tree, or
    /// `"unversioned"` when git itself cannot answer.
    pub rev: String,
    /// FNV-1a hex of `git diff HEAD` over tracked source (excluding
    /// [`DIFF_HASH_EXCLUDES`]). `None` on a clean tree. FNV rather than a
    /// cryptographic hash because the job is *distinguishing* local working
    /// trees, not resisting an adversary.
    pub diff_hash: Option<String>,
}

impl SourceVersion {
    /// Resolve the working tree's current version. Runs git in this crate's
    /// manifest directory so the answer names this repo regardless of cwd.
    ///
    /// On any git failure the revision is `"unversioned"`, printed loudly to
    /// stderr rather than silently swallowed — a run whose code cannot be
    /// attributed to a commit is still worth having, but a caller must be
    /// able to see why cross-run comparisons against it cannot be trusted.
    #[must_use]
    pub fn current(excludes: &[&str]) -> Self {
        let unversioned = |why: &str| -> SourceVersion {
            eprintln!(
                "WARNING: could not resolve the source revision ({why}) — this run's \
                 config hash will record source_rev=\"unversioned\", making it \
                 UNATTRIBUTABLE to a commit."
            );
            SourceVersion {
                rev: "unversioned".to_string(),
                diff_hash: None,
            }
        };
        // `git()` hands back raw stdout now (see its doc comment) — `rev-parse`
        // and `status` are trimmed here, at their own call sites, since only
        // the `diff` branch below needs the untrimmed bytes.
        let head = match git(&["rev-parse", "HEAD"]) {
            Ok(s) if !s.trim().is_empty() => s.trim().to_string(),
            Ok(_) => return unversioned("`git rev-parse HEAD` produced no output"),
            Err(why) => return unversioned(&why),
        };
        // Filtered by the SAME `excludes` the diff below uses: a tree whose
        // only changes are under an excluded path (typically this run's own
        // prior output, `docs/results/journal.jsonl`) must not read as dirty
        // when the diff that follows would be empty anyway — otherwise a
        // clean run and an excluded-paths-only rerun disagree on `rev`
        // ("<sha>" vs "<sha>-dirty") despite hashing the identical (empty)
        // diff (P2 finding on the fix commit for PR #1019).
        let mut status_args = vec!["status", "--porcelain", "--"];
        status_args.extend(excludes.iter().copied());
        let status = match git(&status_args) {
            Ok(s) => s,
            Err(why) => return unversioned(&why),
        };
        if status.trim().is_empty() {
            return SourceVersion {
                rev: head,
                diff_hash: None,
            };
        }
        let mut diff_args = vec!["diff", "HEAD", "--"];
        diff_args.extend(excludes.iter().copied());
        match git(&diff_args) {
            Ok(diff) => SourceVersion {
                rev: format!("{head}-dirty"),
                // Raw bytes, untrimmed: a dirty diff whose only difference
                // from another is trailing whitespace on the last changed
                // line must not hash the same as that other diff (P2 finding
                // on PR #1019) — trimming here would silently alias two
                // distinct working trees onto one `diff_hash`.
                diff_hash: Some(fnv1a64_hex(diff.as_bytes())),
            },
            Err(why) => {
                eprintln!(
                    "WARNING: the working tree is dirty but the diff could not be hashed \
                     ({why}) — this run's config hash cannot distinguish it from other \
                     uncommitted variants of {head}."
                );
                SourceVersion {
                    rev: format!("{head}-dirty"),
                    diff_hash: None,
                }
            }
        }
    }

    fn hash_input(&self) -> String {
        match &self.diff_hash {
            Some(h) => format!("{};diff={h}", self.rev),
            None => self.rev.clone(),
        }
    }
}

/// The build and machine facts that select the emitted instructions — same
/// source, different machine or build, different timings, so these belong in
/// every config hash beside the source revision.
///
/// The ISA tier is the one the JIT selected at startup from the CPU (or
/// `PIXELFLOW_ISA`) — the fact that actually selects the emitter — and the
/// profile decides whether the timings are meaningful at all. The tier
/// implies the `MulAdd` rounding: every selectable tier has a hardware FMA
/// (AVX2 requires FMA3, AVX-512 has it, NEON has `FMLA`), so the separate
/// `fma=` field two runs of materially different kernels once hid behind
/// (P2 finding on the fix commit for PR #1019: an `avx2,+fma` and an
/// `avx2,-fma` build under one fingerprint) has nothing left to distinguish.
/// The specific CPU model is deliberately not read here — it needs a syscall
/// or `/proc`, varies across otherwise-identical cloud instances, and a run's
/// sentinel calibration already records per-run clock behavior.
#[must_use]
pub fn environment_fingerprint() -> String {
    format!(
        "arch={};os={};ptr={};isa={};profile={}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        usize::BITS,
        pixelflow_codegen::isa::detect().name(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )
}

/// Where an artifact lives and the content hash binding this record to those
/// exact bytes — the [`crate::training::mint::weights_identity`] pattern,
/// generalized to any file a journal line needs to name.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ArtifactId {
    pub path: String,
    pub identity: String,
}

impl ArtifactId {
    /// Name `path`, content-hashing `bytes` as its identity.
    #[must_use]
    pub fn of(path: impl Into<String>, bytes: &[u8]) -> Self {
        Self {
            path: path.into(),
            identity: fnv1a64_hex(bytes),
        }
    }

    /// Name `path` with an identity already computed elsewhere, so a caller
    /// that already hashed the artifact's bytes need not re-hash them just
    /// to fill this in.
    #[must_use]
    pub fn with_identity(path: impl Into<String>, identity: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            identity: identity.into(),
        }
    }
}

/// Everything that shapes a measurement, content-hashed into one value: the
/// source revision (+ diff), which corpus was measured, which weights
/// produced the run, and the machine it ran on. Two runs with the same
/// `value` are the same experiment; the individual fields stay alongside it
/// so a journal reader does not have to decompose the hash to find out WHY
/// two runs differ.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConfigHash {
    pub source_rev: String,
    pub source_diff_hash: Option<String>,
    pub corpus_identity: String,
    pub weights_identity: String,
    pub environment: String,
    /// Caller-supplied protocol parameters (seeds, sample sizes, tier,
    /// benchmark mode, epoch count, ...) that are not artifacts but still
    /// define the experiment. Free text, since the two binaries' protocols
    /// differ and forcing one shape onto both would be artificial
    /// commonality (docs/plans/2026-08-17-cost-model-domain.md, journal.rs
    /// module doc: "each with its own record schema... that stays").
    pub protocol: String,
    /// FNV-1a 64 hex over every field above, concatenated in declaration
    /// order — the single value two runs compare to ask "same setup?".
    pub value: String,
}

impl ConfigHash {
    #[must_use]
    pub fn compose(
        source: &SourceVersion,
        corpus_identity: &str,
        weights_identity: &str,
        protocol: &str,
    ) -> Self {
        let environment = environment_fingerprint();
        let composed = format!(
            "rev={};corpus={corpus_identity};weights={weights_identity};env={environment};\
             protocol={protocol}",
            source.hash_input(),
        );
        Self {
            source_rev: source.rev.clone(),
            source_diff_hash: source.diff_hash.clone(),
            corpus_identity: corpus_identity.to_string(),
            weights_identity: weights_identity.to_string(),
            environment,
            protocol: protocol.to_string(),
            value: fnv1a64_hex(composed.as_bytes()),
        }
    }
}

/// One line of `docs/results/journal.jsonl`: the envelope every
/// journal-writing binary shares — which record type, when, under what
/// configuration, and which weights produced it — wrapping the caller's own
/// metrics. Two binaries hand-rolling this envelope independently, so a fix
/// or a new provenance field in one could silently fail to reach the other,
/// is exactly the structurally-divergent-journal-line risk this type exists
/// to close (docs/plans/2026-08-17-cost-model-domain.md, J15): the shape a
/// caller can even attempt to serialize is fixed here, once.
#[derive(Serialize)]
pub struct JournalEntry<T: Serialize> {
    pub record: &'static str,
    pub ts_unix: u64,
    pub config: ConfigHash,
    pub weights: ArtifactId,
    #[serde(flatten)]
    pub metrics: T,
}

impl<T: Serialize> JournalEntry<T> {
    #[must_use]
    pub fn new(record: &'static str, config: ConfigHash, weights: ArtifactId, metrics: T) -> Self {
        Self {
            record,
            ts_unix: crate::schema::unix_now_s(),
            config,
            weights,
            metrics,
        }
    }

    /// Append this entry to `path` via [`append_record`].
    pub fn append(&self, path: &Path) {
        append_record(path, self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[derive(Serialize)]
    struct Rec {
        n: u32,
    }

    #[test]
    fn append_record_creates_parent_dir_and_appends_one_line_per_call() {
        let mut dir = std::env::temp_dir();
        dir.push(format!("pf_journal_test_{}", std::process::id()));
        let path = dir.join("nested/journal.jsonl");
        let _ = fs::remove_dir_all(&dir);

        append_record(&path, &Rec { n: 1 });
        append_record(&path, &Rec { n: 2 });

        let contents = fs::read_to_string(&path).expect("journal file should exist");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines, vec!["{\"n\":1}", "{\"n\":2}"]);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn git_returns_raw_untrimmed_stdout() {
        // `git rev-parse HEAD` always terminates its answer with a trailing
        // newline. If `git()` still trimmed internally (the defect this
        // guards against — a trimmed `diff HEAD` silently loses
        // trailing-whitespace-only differences before `SourceVersion` hashes
        // it, aliasing two distinct working trees onto one `diff_hash`), that
        // newline would never survive to a caller. Black-box probe of the
        // shared helper's contract, not of this repo's source-control state.
        match git(&["rev-parse", "HEAD"]) {
            Ok(raw) => assert!(
                raw.ends_with('\n'),
                "git() must hand back raw untrimmed stdout, including the trailing \
                 newline every git subcommand emits, or a hashed diff could silently \
                 lose trailing whitespace: got {raw:?}"
            ),
            Err(_) => {
                // No git binary, or not a repo, in this environment — nothing
                // to assert here; SourceVersion::current's "unversioned"
                // fallback covers this case at the call site.
            }
        }
    }

    /// Run `git` in `cwd`, panicking on spawn failure or nonzero exit, and
    /// return raw stdout. Shared by the scratch-repo tests below, which
    /// exercise `git()`'s cwd/pathspec contract against an isolated repo
    /// rather than this session's actual working tree.
    fn run_git(cwd: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .args(args)
            .current_dir(cwd)
            .env("GIT_AUTHOR_NAME", "test")
            .env("GIT_AUTHOR_EMAIL", "test@example.com")
            .env("GIT_COMMITTER_NAME", "test")
            .env("GIT_COMMITTER_EMAIL", "test@example.com")
            .output()
            .unwrap_or_else(|e| panic!("failed to spawn git {args:?}: {e}"));
        assert!(
            out.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    }

    /// A scratch repo shaped like this one (a `pixelflow-pipeline/`
    /// subdirectory below the repo root, with a `data/` output dir and a
    /// `src/` source dir inside it), committed with one file in each. Returns
    /// `(repo_root, crate_dir)`. Callers dirty the tree and run `git` from
    /// `crate_dir` — exactly where the real `git()` helper runs from — to
    /// exercise the cwd/pathspec mismatch these tests guard against.
    fn scratch_repo(tag: &str) -> (PathBuf, PathBuf) {
        let root = std::env::temp_dir().join(format!(
            "pf_journal_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock is before the unix epoch")
                .as_nanos()
        ));
        let crate_dir = root.join("pixelflow-pipeline");
        let data_dir = crate_dir.join("data");
        let src_dir = crate_dir.join("src");
        fs::create_dir_all(&data_dir).expect("create data dir");
        fs::create_dir_all(&src_dir).expect("create src dir");

        run_git(&root, &["init", "-q"]);
        fs::write(data_dir.join("out.jsonl"), "before\n").expect("write data file");
        fs::write(src_dir.join("lib.rs"), "// before\n").expect("write source file");
        run_git(&root, &["add", "-A"]);
        run_git(&root, &["commit", "-q", "-m", "initial"]);

        (root, crate_dir)
    }

    #[test]
    fn diff_hash_exclusions_survive_running_from_a_subdirectory() {
        // Reproduces the real defect (P2 finding on the fix commit for PR
        // #1019): `git()` runs with `current_dir` set to THIS crate's own
        // manifest directory, not the repository root, so a bare
        // `:(exclude)pixelflow-pipeline/data` pathspec is resolved relative
        // to that directory — i.e. as
        // `pixelflow-pipeline/pixelflow-pipeline/data`, which never exists —
        // and silently excludes nothing. `:(top,exclude)` anchors to the
        // repo root regardless of cwd.
        let (root, crate_dir) = scratch_repo("pathspec_test");

        // Dirty the tree: change BOTH the run-output file (should be
        // excluded) and a real source file (should NOT be excluded).
        fs::write(crate_dir.join("data/out.jsonl"), "after\n").expect("rewrite data file");
        fs::write(crate_dir.join("src/lib.rs"), "// after\n").expect("rewrite source file");

        // `git()` runs from the crate directory, exactly like the real
        // helper — this is what makes the bare pathspec resolve wrong.
        let bare_diff = run_git(
            &crate_dir,
            &[
                "diff",
                "HEAD",
                "--",
                ":(exclude)pixelflow-pipeline/data",
                ":(exclude)docs/results",
            ],
        );
        assert!(
            bare_diff.contains("out.jsonl"),
            "test assumption: the bare pathspec must reproduce the bug (fail to \
             exclude the data file) when run from the crate subdirectory, or this \
             test is no longer exercising it:\n{bare_diff}"
        );

        let mut anchored_args: Vec<&str> = vec!["diff", "HEAD", "--"];
        anchored_args.extend(DIFF_HASH_EXCLUDES);
        let anchored_diff = run_git(&crate_dir, &anchored_args);
        assert!(
            !anchored_diff.contains("out.jsonl"),
            "the top-anchored exclude must drop the run's own output file even when \
             git runs from the crate subdirectory:\n{anchored_diff}"
        );
        assert!(
            anchored_diff.contains("lib.rs"),
            "the top-anchored exclude must NOT drop a real source change:\n{anchored_diff}"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn status_filtering_ignores_excluded_paths_only_changes() {
        // Reproduces the real defect (P2 finding on the fix commit for PR
        // #1019): `git status --porcelain` without the same `excludes` the
        // diff uses reports the tree dirty even when the ONLY changes are
        // under an excluded path (typically this run's own prior output),
        // so `SourceVersion::current` would record rev="<sha>-dirty" while
        // hashing an empty diff — disagreeing with a genuinely clean run on
        // `rev` despite both having nothing to say about tracked source.
        let (root, crate_dir) = scratch_repo("status_filter_test");

        // Dirty ONLY the excluded output file — no source change at all.
        fs::write(crate_dir.join("data/out.jsonl"), "after\n").expect("rewrite data file");

        let unfiltered = run_git(&crate_dir, &["status", "--porcelain"]);
        assert!(
            !unfiltered.trim().is_empty(),
            "test assumption: unfiltered status must see the excluded-only change, or \
             this test is no longer exercising the bug"
        );

        let mut filtered_args: Vec<&str> = vec!["status", "--porcelain", "--"];
        filtered_args.extend(DIFF_HASH_EXCLUDES);
        let filtered = run_git(&crate_dir, &filtered_args);
        assert!(
            filtered.trim().is_empty(),
            "status filtered by DIFF_HASH_EXCLUDES must report clean when the only \
             changes are under an excluded path: {filtered:?}"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn environment_fingerprint_records_the_tier_the_jit_selected() {
        // The defect this guards, in its current form: the tier is what
        // selects the emitter, and it is decided at runtime — a fingerprint
        // read off `cfg!(target_feature)` would say `baseline` on every
        // machine while the JIT ran AVX-512 on some and AVX2 on others, two
        // materially different kernels under one fingerprint (the shape of
        // the P2 finding on the fix commit for PR #1019).
        let fp = environment_fingerprint();
        let isa = pixelflow_codegen::isa::detect().name();
        assert!(
            fp.contains(&format!(";isa={isa};")),
            "the fingerprint must name the tier the JIT selected ({isa}): {fp}"
        );
    }

    #[test]
    fn diff_hash_distinguishes_trailing_whitespace_only_changes() {
        // The defect this guards: hashing a TRIMMED diff would collapse two
        // working trees whose only difference is trailing whitespace on the
        // last changed line onto the same `diff_hash` (P2 finding on PR
        // #1019) — defeating the journal's provenance guarantee that two
        // distinct trees never share an identity.
        let a = "diff --git a/x b/x\n+some line\n";
        let b = "diff --git a/x b/x\n+some line \n"; // trailing space added
        assert_ne!(fnv1a64_hex(a.as_bytes()), fnv1a64_hex(b.as_bytes()));
    }

    #[test]
    #[should_panic(expected = "unsmudged Git LFS pointer")]
    fn append_record_refuses_an_lfs_pointer_stub() {
        let mut path = std::env::temp_dir();
        path.push(format!("pf_journal_lfs_test_{}.jsonl", std::process::id()));
        fs::write(
            &path,
            "version https://git-lfs.github.com/spec/v1\noid sha256:0\nsize 1\n",
        )
        .expect("write pointer stub");

        append_record(&path, &Rec { n: 1 });
    }
}
