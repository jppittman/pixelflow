//! Split manifest for the tiered benchmark corpus (plan 0.2, audit L4).
//!
//! The corpus is split TRAIN / DEV / FINAL **by family** — a family is one
//! `(band, seed)` generator stream — never by random split. Random splits
//! over a single generator leak: the generator reuses structural motifs
//! across draws, so near-duplicates land on both sides and the "held-out"
//! score measures memorization. The checked-in manifest
//! (`pixelflow-pipeline/corpus_split.toml`) is the single source of truth
//! for which families belong to which tier; this module loads it, validates
//! it loudly (overlaps and empty tiers are errors at load, not surprises at
//! training time), and answers tier-assignment queries.
//!
//! The manifest is a deliberately tiny TOML subset — `[section]` headers,
//! integer arrays, `[start, end]` range pairs, and string arrays — parsed
//! here strictly (unknown keys, duplicate keys, and multi-line values are
//! errors) rather than through a TOML dependency the workspace doesn't
//! otherwise need.

use std::collections::HashSet;
use std::fmt;
use std::marker::PhantomData;
use std::path::Path;

use pixelflow_ir::{ExprData, Node, node_count_subtree};

use super::structural::FenceKey;

/// Corpus tier. Ordered by holdout sacredness: `Train < Dev < Final`, so the
/// corpus build can assert it admits more-sacred tiers first and cross-tier
/// duplicates are always dropped from the less sacred side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Tier {
    /// Trained on; grows with mined adversarial expressions.
    Train,
    /// Held out by family; drives per-round SELECT decisions.
    Dev,
    /// Held out by family; touched only for publication claims.
    Final,
}

impl Tier {
    /// All tiers, in ascending sacredness order.
    pub const ALL: [Tier; 3] = [Tier::Train, Tier::Dev, Tier::Final];

    /// Lowercase tier name, used in corpus file names (`corpus_train.bin`).
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Tier::Train => "train",
            Tier::Dev => "dev",
            Tier::Final => "final",
        }
    }
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A manifest load/parse/validation failure. Always fatal for the caller —
/// a corpus built from a half-understood manifest is worse than no corpus.
#[derive(Debug)]
pub struct SplitError {
    /// 1-based manifest line the error was detected on, when known.
    line: Option<usize>,
    msg: String,
}

impl SplitError {
    fn new(msg: impl Into<String>) -> Self {
        Self {
            line: None,
            msg: msg.into(),
        }
    }

    fn at_line(line: usize, msg: impl Into<String>) -> Self {
        Self {
            line: Some(line),
            msg: msg.into(),
        }
    }
}

impl fmt::Display for SplitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.line {
            Some(line) => write!(f, "split manifest line {line}: {}", self.msg),
            None => write!(f, "split manifest: {}", self.msg),
        }
    }
}

impl std::error::Error for SplitError {}

/// Inclusive family-seed range `[start, end]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeedRange {
    pub start: u64,
    pub end: u64,
}

impl SeedRange {
    #[must_use]
    pub fn contains(self, seed: u64) -> bool {
        (self.start..=self.end).contains(&seed)
    }

    #[must_use]
    pub fn overlaps(self, other: SeedRange) -> bool {
        self.start <= other.end && other.start <= self.end
    }

    /// Number of seeds in the range. Inclusive, so never zero for a valid
    /// (non-inverted) range.
    #[must_use]
    pub fn count(self) -> u64 {
        self.end - self.start + 1
    }
}

/// One generator family: a `(band, seed)` draw stream — the split unit
/// (docs/plans/2026-08-17-cost-model-domain.md, J7). A newtype instead of a
/// bare tuple so a swapped argument order (`(seed, band)`) is a type error
/// at the call site instead of a silently wrong RNG seed or tier lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Family {
    /// Index into the generator's band table.
    pub band: usize,
    /// The family's draw-stream seed.
    pub seed: u64,
}

impl Family {
    #[must_use]
    pub fn new(band: usize, seed: u64) -> Self {
        Self { band, seed }
    }
}

impl fmt::Display for Family {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "band {} family {}", self.band, self.seed)
    }
}

/// One tier's family set: the cartesian product of `bands` and the seeds in
/// `seed_ranges`.
#[derive(Clone, Debug, Default)]
pub struct TierSpec {
    /// Indices into the generator's band table.
    pub bands: Vec<usize>,
    /// Inclusive family-seed ranges, applying to every band of this tier.
    pub seed_ranges: Vec<SeedRange>,
}

impl TierSpec {
    /// Whether `family` belongs to this tier.
    #[must_use]
    pub fn contains(&self, family: Family) -> bool {
        self.bands.contains(&family.band)
            && self.seed_ranges.iter().any(|r| r.contains(family.seed))
    }

    /// Total number of families in this tier.
    ///
    /// # Panics
    ///
    /// Panics if the count overflows `usize` — a manifest describing more
    /// families than addressable memory is a manifest bug.
    #[must_use]
    pub fn family_count(&self) -> usize {
        let seeds: u128 = self.seed_ranges.iter().map(|r| u128::from(r.count())).sum();
        let total = seeds * self.bands.len() as u128;
        usize::try_from(total)
            .unwrap_or_else(|_| panic!("TierSpec::family_count overflows usize: {total} families"))
    }

    /// All families of this tier, band-major, in manifest order — the
    /// deterministic generation order for the corpus build.
    pub fn families(&self) -> impl Iterator<Item = Family> + '_ {
        self.bands.iter().flat_map(move |&band| {
            self.seed_ranges
                .iter()
                .flat_map(move |r| (r.start..=r.end).map(move |seed| Family::new(band, seed)))
        })
    }
}

/// The parsed, validated three-tier split.
#[derive(Clone, Debug)]
pub struct SplitManifest {
    pub train: TierSpec,
    pub dev: TierSpec,
    pub final_tier: TierSpec,
    /// Names of the real production kernels in FINAL (resolved to arena
    /// builders by the corpus build; an unknown name is a build-time panic).
    pub final_kernels: Vec<String>,
    /// Named DEV-only out-of-distribution families
    /// (`[dev] families = [...]`), entered the same way `final_kernels`
    /// enters FINAL: a mechanically real generator built once per name, not
    /// a `(band, seed)` draw stream from the shared `BwdGenerator`
    /// (docs/plans/2026-09-01-phase3-round1b-domain-shift-registration.md
    /// §3, "Out-of-distribution families"). Optional — a manifest that
    /// declares none parses with an empty list. `families` is accepted only
    /// under `[dev]`; the key is unknown (and therefore rejected) under
    /// `[train]` or `[final]`, and a name may not repeat within the list or
    /// collide with a `[final]` kernel name.
    pub dev_families: Vec<String>,
}

impl SplitManifest {
    /// Load and validate a manifest file. Any I/O, parse, or validation
    /// failure is an error — there is no partial or default manifest.
    pub fn load(path: &Path) -> Result<Self, SplitError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| SplitError::new(format!("failed to read {}: {e}", path.display())))?;
        Self::parse(&text)
    }

    /// Parse and validate manifest text.
    pub fn parse(text: &str) -> Result<Self, SplitError> {
        let raw = parse_sections(text)?;
        let manifest = Self {
            train: raw.tier("train")?,
            dev: raw.tier("dev")?,
            final_tier: raw.tier("final")?,
            final_kernels: raw.final_kernels()?,
            dev_families: raw.dev_families()?,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// The spec for one tier.
    #[must_use]
    pub fn spec(&self, tier: Tier) -> &TierSpec {
        match tier {
            Tier::Train => &self.train,
            Tier::Dev => &self.dev,
            Tier::Final => &self.final_tier,
        }
    }

    /// Which tier `family` belongs to, if any. Validation guarantees at
    /// most one tier matches.
    #[must_use]
    pub fn tier_of(&self, family: Family) -> Option<Tier> {
        Tier::ALL
            .into_iter()
            .find(|&tier| self.spec(tier).contains(family))
    }

    /// Largest band index referenced anywhere in the manifest, so the corpus
    /// build can assert it against the generator's band table.
    #[must_use]
    pub fn max_band_index(&self) -> usize {
        Tier::ALL
            .iter()
            .flat_map(|&t| self.spec(t).bands.iter().copied())
            .max()
            .expect("validated manifest has at least one band per tier")
    }

    fn validate(&self) -> Result<(), SplitError> {
        for &tier in &Tier::ALL {
            let spec = self.spec(tier);
            if spec.bands.is_empty() {
                return Err(SplitError::new(format!(
                    "tier [{tier}] has no bands — an empty tier cannot serve its role"
                )));
            }
            if spec.seed_ranges.is_empty() {
                return Err(SplitError::new(format!(
                    "tier [{tier}] has no seed ranges — an empty tier cannot serve its role"
                )));
            }
            let mut seen_bands = HashSet::new();
            for &band in &spec.bands {
                if !seen_bands.insert(band) {
                    return Err(SplitError::new(format!(
                        "tier [{tier}] lists band {band} twice"
                    )));
                }
            }
            for r in &spec.seed_ranges {
                if r.start > r.end {
                    return Err(SplitError::new(format!(
                        "tier [{tier}] has inverted seed range [{}, {}]",
                        r.start, r.end
                    )));
                }
            }
            for (i, a) in spec.seed_ranges.iter().enumerate() {
                for b in &spec.seed_ranges[i + 1..] {
                    if a.overlaps(*b) {
                        return Err(SplitError::new(format!(
                            "tier [{tier}] has overlapping seed ranges [{}, {}] and [{}, {}]",
                            a.start, a.end, b.start, b.end
                        )));
                    }
                }
            }
        }

        if self.final_kernels.is_empty() {
            return Err(SplitError::new(
                "tier [final] lists no kernels — the FINAL tier must contain \
                 the named production kernels",
            ));
        }
        let mut seen_kernels = HashSet::new();
        for name in &self.final_kernels {
            if !seen_kernels.insert(name.as_str()) {
                return Err(SplitError::new(format!(
                    "tier [final] lists kernel \"{name}\" twice"
                )));
            }
        }

        // dev_families: no name repeats within the list, and no name
        // collides with a [final] kernel — both are "named entry" pathways
        // outside the band/seed system, so a shared name would be ambiguous
        // about which generator/tier it identifies.
        let mut seen_families = HashSet::new();
        for name in &self.dev_families {
            if !seen_families.insert(name.as_str()) {
                return Err(SplitError::new(format!(
                    "section [dev] lists family \"{name}\" twice"
                )));
            }
        }
        for name in &self.dev_families {
            if seen_kernels.contains(name.as_str()) {
                return Err(SplitError::new(format!(
                    "\"{name}\" is both a [dev] family and a [final] kernel — cross-tier \
                     duplicate: a named entry must belong to exactly one tier"
                )));
            }
        }

        // Cross-tier disjointness: a family claimable by two tiers is
        // leakage by construction.
        for (i, &a) in Tier::ALL.iter().enumerate() {
            for &b in &Tier::ALL[i + 1..] {
                let (sa, sb) = (self.spec(a), self.spec(b));
                for &band in &sa.bands {
                    if !sb.bands.contains(&band) {
                        continue;
                    }
                    for ra in &sa.seed_ranges {
                        for rb in &sb.seed_ranges {
                            if ra.overlaps(*rb) {
                                return Err(SplitError::new(format!(
                                    "tiers [{a}] and [{b}] overlap on band {band}: seed ranges \
                                     [{}, {}] and [{}, {}] intersect — a family may belong to \
                                     exactly one tier",
                                    ra.start, ra.end, rb.start, rb.end
                                )));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

// ── Strict TOML-subset parsing ──────────────────────────────────────────────

/// Raw per-section key/value storage before assembly into [`SplitManifest`].
#[derive(Default)]
struct RawSection {
    bands: Option<Vec<usize>>,
    seeds: Option<Vec<SeedRange>>,
    kernels: Option<Vec<String>>,
    families: Option<Vec<String>>,
}

struct RawManifest {
    train: RawSection,
    dev: RawSection,
    final_section: RawSection,
}

impl RawManifest {
    fn section(&mut self, name: &str) -> Option<&mut RawSection> {
        match name {
            "train" => Some(&mut self.train),
            "dev" => Some(&mut self.dev),
            "final" => Some(&mut self.final_section),
            _ => None,
        }
    }

    fn section_ref(&self, name: &str) -> &RawSection {
        match name {
            "train" => &self.train,
            "dev" => &self.dev,
            "final" => &self.final_section,
            other => panic!("RawManifest::section_ref: unknown section {other}"),
        }
    }

    fn tier(&self, name: &str) -> Result<TierSpec, SplitError> {
        let section = self.section_ref(name);
        let bands = section
            .bands
            .clone()
            .ok_or_else(|| SplitError::new(format!("section [{name}] is missing `bands`")))?;
        let seed_ranges = section
            .seeds
            .clone()
            .ok_or_else(|| SplitError::new(format!("section [{name}] is missing `seeds`")))?;
        Ok(TierSpec { bands, seed_ranges })
    }

    fn final_kernels(&self) -> Result<Vec<String>, SplitError> {
        self.final_section
            .kernels
            .clone()
            .ok_or_else(|| SplitError::new("section [final] is missing `kernels`"))
    }

    /// `[dev] families`, or an empty list when the manifest declares none —
    /// unlike `kernels`, this key is optional: most manifests (every test
    /// fixture, and every manifest predating round 1b) name no OOD family
    /// at all.
    fn dev_families(&self) -> Result<Vec<String>, SplitError> {
        Ok(self.dev.families.clone().unwrap_or_default())
    }
}

fn parse_sections(text: &str) -> Result<RawManifest, SplitError> {
    let mut raw = RawManifest {
        train: RawSection::default(),
        dev: RawSection::default(),
        final_section: RawSection::default(),
    };
    let mut sections_seen: HashSet<String> = HashSet::new();
    let mut current: Option<String> = None;

    for (idx, raw_line) in text.lines().enumerate() {
        let lineno = idx + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if let Some(rest) = line.strip_prefix('[') {
            // Distinguish a section header `[name]` from a value line — value
            // lines always contain `=` before their brackets.
            let name = rest.strip_suffix(']').ok_or_else(|| {
                SplitError::at_line(lineno, format!("malformed section header `{line}`"))
            })?;
            let name = name.trim().to_string();
            if raw.section(&name).is_none() {
                return Err(SplitError::at_line(
                    lineno,
                    format!("unknown section [{name}] (expected [train], [dev], or [final])"),
                ));
            }
            if !sections_seen.insert(name.clone()) {
                return Err(SplitError::at_line(
                    lineno,
                    format!("duplicate section [{name}]"),
                ));
            }
            current = Some(name);
            continue;
        }

        let (key, value) = line.split_once('=').ok_or_else(|| {
            SplitError::at_line(
                lineno,
                format!("expected `key = value` or `[section]`, got `{line}`"),
            )
        })?;
        let key = key.trim();
        let value = value.trim();
        let section_name = current.clone().ok_or_else(|| {
            SplitError::at_line(lineno, format!("key `{key}` appears before any [section]"))
        })?;
        let section = raw
            .section(&section_name)
            .expect("current section validated at header");

        match key {
            "bands" => {
                if section.bands.is_some() {
                    return Err(SplitError::at_line(
                        lineno,
                        format!("duplicate key `bands` in [{section_name}]"),
                    ));
                }
                section.bands = Some(
                    parse_usize_list(value)
                        .map_err(|e| SplitError::at_line(lineno, format!("bands: {e}")))?,
                );
            }
            "seeds" => {
                if section.seeds.is_some() {
                    return Err(SplitError::at_line(
                        lineno,
                        format!("duplicate key `seeds` in [{section_name}]"),
                    ));
                }
                section.seeds = Some(
                    parse_seed_ranges(value)
                        .map_err(|e| SplitError::at_line(lineno, format!("seeds: {e}")))?,
                );
            }
            "kernels" if section_name == "final" => {
                if section.kernels.is_some() {
                    return Err(SplitError::at_line(
                        lineno,
                        "duplicate key `kernels` in [final]".to_string(),
                    ));
                }
                section.kernels = Some(
                    parse_string_list(value)
                        .map_err(|e| SplitError::at_line(lineno, format!("kernels: {e}")))?,
                );
            }
            "families" if section_name == "dev" => {
                if section.families.is_some() {
                    return Err(SplitError::at_line(
                        lineno,
                        "duplicate key `families` in [dev]".to_string(),
                    ));
                }
                section.families = Some(
                    parse_string_list(value)
                        .map_err(|e| SplitError::at_line(lineno, format!("families: {e}")))?,
                );
            }
            other => {
                return Err(SplitError::at_line(
                    lineno,
                    format!("unknown key `{other}` in [{section_name}]"),
                ));
            }
        }
    }

    Ok(raw)
}

/// Cut a line at the first `#` that is not inside a double-quoted string.
fn strip_comment(line: &str) -> &str {
    let mut in_string = false;
    for (i, c) in line.char_indices() {
        match c {
            '"' => in_string = !in_string,
            '#' if !in_string => return &line[..i],
            _ => {}
        }
    }
    line
}

/// `value` must be a single-line bracketed array; returns its inner text.
fn strip_brackets(value: &str) -> Result<&str, String> {
    let value = value.trim();
    value
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .ok_or_else(|| format!("expected a single-line `[...]` array, got `{value}`"))
}

/// Split array-inner text on commas at bracket depth 0. Allows one trailing
/// comma; rejects empty elements otherwise.
fn split_top_level(inner: &str) -> Result<Vec<&str>, String> {
    let mut parts = Vec::new();
    let mut depth: i32 = 0;
    let mut start = 0usize;
    for (i, c) in inner.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth < 0 {
                    return Err(format!("unbalanced `]` in `{inner}`"));
                }
            }
            ',' if depth == 0 => {
                parts.push(inner[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    if depth != 0 {
        return Err(format!("unbalanced `[` in `{inner}`"));
    }
    let tail = inner[start..].trim();
    if !tail.is_empty() {
        parts.push(tail);
    } else if !parts.is_empty() && inner.trim().ends_with(',') {
        // Trailing comma: fine, nothing to push.
    }
    if parts.iter().any(|p| p.is_empty()) {
        return Err(format!("empty element in `{inner}`"));
    }
    Ok(parts)
}

fn parse_usize_list(value: &str) -> Result<Vec<usize>, String> {
    let inner = strip_brackets(value)?;
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(inner)?
        .into_iter()
        .map(|p| {
            p.parse::<usize>()
                .map_err(|e| format!("`{p}` is not an unsigned integer: {e}"))
        })
        .collect()
}

fn parse_seed_ranges(value: &str) -> Result<Vec<SeedRange>, String> {
    let inner = strip_brackets(value)?;
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(inner)?
        .into_iter()
        .map(|pair| {
            let pair_inner = strip_brackets(pair)
                .map_err(|_| format!("expected `[start, end]` pair, got `{pair}`"))?;
            let ends = split_top_level(pair_inner)?;
            let [start, end] = ends.as_slice() else {
                return Err(format!(
                    "expected exactly two values in `[start, end]`, got {} in `{pair}`",
                    ends.len()
                ));
            };
            let start = start
                .parse::<u64>()
                .map_err(|e| format!("`{start}` is not an unsigned integer: {e}"))?;
            let end = end
                .parse::<u64>()
                .map_err(|e| format!("`{end}` is not an unsigned integer: {e}"))?;
            Ok(SeedRange { start, end })
        })
        .collect()
}

fn parse_string_list(value: &str) -> Result<Vec<String>, String> {
    let inner = strip_brackets(value)?;
    if inner.trim().is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(inner)?
        .into_iter()
        .map(|p| {
            let unquoted = p
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .ok_or_else(|| format!("expected a double-quoted string, got `{p}`"))?;
            if unquoted.is_empty() {
                return Err("empty string element".to_string());
            }
            if unquoted.contains(['"', '\\']) {
                return Err(format!(
                    "escapes and embedded quotes are not supported: `{p}`"
                ));
            }
            Ok(unquoted.to_string())
        })
        .collect()
}

// ── Holdout fence (plan 0.2, J7/J8) ─────────────────────────────────────────

mod private {
    /// Seals [`super::HoldoutSide`] to this module's two markers.
    pub trait Sealed {}
}

/// Which tier a [`Fence`] is allowed to be built from: always a held-out
/// tier, never the tier a fence exists to protect.
///
/// Sealed to [`DevSide`] and [`FinalSide`] — there is no impl (and cannot
/// be, from outside this module) for `Tier::Train`. "Build the fence from
/// the tier it exists to hold out of" is not a misuse a caller could make
/// and get away with; it is not an API a caller can *reach*, because
/// [`Fence::build`] takes no data from its caller at all — it derives its
/// own source file from `Self::TIER`. See the direction note on [`Fence`].
pub trait HoldoutSide: private::Sealed {
    /// The corpus tier this side is fenced from.
    const TIER: Tier;
}

/// Marker: the DEV tier.
pub struct DevSide;
/// Marker: the FINAL tier.
pub struct FinalSide;

impl private::Sealed for DevSide {}
impl private::Sealed for FinalSide {}
impl HoldoutSide for DevSide {
    const TIER: Tier = Tier::Dev;
}
impl HoldoutSide for FinalSide {
    const TIER: Tier = Tier::Final;
}

/// A holdout fence: every [`FenceKey`] belonging to tier `T`'s corpus file,
/// so a candidate expression from anywhere else can be checked for
/// "structure (in feature-quotient space) the model has already seen"
/// (docs/plans/2026-08-17-cost-model-domain.md, J7/J8/P1(d)).
///
/// # Direction
///
/// `T` is phantom, but not decorative: [`Fence::build`] is the only
/// constructor, and it does not accept entries from its caller — it reads
/// `corpus_dir/corpus_{T::TIER}.bin` itself. So a `Fence<DevSide>` is
/// *always* built from `corpus_dev.bin`, never from whatever the caller
/// happened to have on hand, and a `Fence<Train>` cannot be spelled at all
/// (`HoldoutSide` has no impl for `Tier::Train` — see the sealed trait
/// above). Both together are what "the backwards misuse must not compile"
/// means here: there is no code path that produces a fence built from the
/// tier it is meant to hold out of. This is not a convention a caller could
/// violate by passing the wrong argument — `HoldoutSide` being sealed means
/// no crate, including this one, can name a third implementor:
///
/// ```compile_fail
/// use pixelflow_pipeline::training::split::{Fence, HoldoutSide, Tier};
///
/// struct TrainSide;
/// // `private::Sealed` is not reachable outside this module, so this impl
/// // does not compile — and without it, `Fence<TrainSide>` below is not a
/// // well-formed type at all.
/// impl HoldoutSide for TrainSide {
///     const TIER: Tier = Tier::Train;
/// }
///
/// let _bad: Fence<TrainSide> = todo!();
/// ```
pub struct Fence<T: HoldoutSide> {
    keys: HashSet<FenceKey>,
    entries: usize,
    node_counts: Vec<usize>,
    _side: PhantomData<T>,
}

impl<T: HoldoutSide> Fence<T> {
    /// Build from every entry of `corpus_dir/corpus_{T::TIER}.bin`.
    ///
    /// # Panics
    ///
    /// Panics when the tier file is missing or fails to parse. A fence that
    /// silently held zero structures because its source file did not exist
    /// would validate nothing while looking like it validated everything.
    #[must_use]
    pub fn build(corpus_dir: &Path) -> Self {
        let tier = T::TIER;
        let path = corpus_dir.join(format!("corpus_{}.bin", tier.name()));
        assert!(
            path.exists(),
            "{tier} tier corpus not found at {} — a holdout fence cannot be built without it, \
             and training an unfenced stream would put {tier} structures into the training set \
             (the held-out metrics would stop measuring a holdout). Run gen_bench_corpus \
             (--output {}), which writes all three tiers together",
            path.display(),
            corpus_dir.display()
        );
        let entries = super::corpus::read_corpus(&path)
            .unwrap_or_else(|e| panic!("failed to read {tier} corpus {}: {e}", path.display()));
        // A zero-entry file parses fine (the writer permits an empty corpus)
        // but is not a fence: `blocked_by_either` would silently admit every
        // candidate on this side, and training could proceed as if the
        // promised {tier} holdout existed when it does not — existence and a
        // successful parse alone do not establish the fence invariant.
        assert!(
            !entries.is_empty(),
            "{tier} tier corpus at {} is empty (zero entries) — a holdout fence built from it \
             would block nothing, silently admitting every candidate on this side. Regenerate \
             the tiered corpus with gen_bench_corpus (--output {})",
            path.display(),
            corpus_dir.display()
        );

        let mut keys = HashSet::with_capacity(entries.len());
        let mut node_counts = Vec::with_capacity(entries.len());
        for (_name, expr) in &entries {
            keys.insert(FenceKey::of(expr.entry()));
            node_counts.push(node_count_subtree(expr.entry()));
        }
        Self {
            keys,
            entries: entries.len(),
            node_counts,
            _side: PhantomData,
        }
    }

    /// Whether `key` is already owned by this side of the fence.
    #[must_use]
    pub fn contains(&self, key: &FenceKey) -> bool {
        self.keys.contains(key)
    }

    /// Number of distinct feature-quotient structures fenced.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether this side holds no structures at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Raw entry count of the source corpus file (`>= len()`: two entries
    /// may collapse to one feature-quotient key).
    #[must_use]
    pub fn entries(&self) -> usize {
        self.entries
    }

    /// Node count of every fenced structure's *entry* (one per corpus
    /// entry, not deduplicated), for size-distribution reporting.
    #[must_use]
    pub fn node_counts(&self) -> &[usize] {
        &self.node_counts
    }
}

/// Check `root` against every held-out side (DEV and FINAL): a candidate
/// expression whose feature-quotient structure already belongs to either is
/// holdout, whichever side owns it.
#[must_use]
pub fn blocked_by_either(
    dev: &Fence<DevSide>,
    final_fence: &Fence<FinalSide>,
    root: Node<'_, ExprData>,
) -> bool {
    let key = FenceKey::of(root);
    dev.contains(&key) || final_fence.contains(&key)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_ir::{ExprBuilder, OpKind, Rooted};

    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock is before the unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("split_fence_{tag}_{nanos}"));
        std::fs::create_dir_all(&dir)
            .unwrap_or_else(|e| panic!("failed to create {}: {e}", dir.display()));
        dir
    }

    fn scaled_var(k: f32) -> Rooted<ExprData> {
        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let c = b.push_const(k);
        let root = b.push_binary(OpKind::Mul, x, c);
        b.finish(&[root]).0
    }

    fn write_tier(dir: &Path, tier: Tier, exprs: &[Rooted<ExprData>]) {
        let entries: Vec<super::super::corpus::Entry> = exprs
            .iter()
            .enumerate()
            .map(|(i, e)| (format!("{}_{i}", tier.name()), e.clone()))
            .collect();
        super::super::corpus::write_corpus(
            &dir.join(format!("corpus_{}.bin", tier.name())),
            &entries,
        )
        .unwrap_or_else(|e| panic!("failed to write {} corpus: {e}", tier.name()));
    }

    #[test]
    fn fence_of_x_times_2_blocks_x_times_3_the_review_case() {
        // The exact review case: a DEV entry `X * 2.0` must fence out a
        // TRAIN candidate `X * 3.0` — same feature-quotient structure,
        // different literal.
        let dir = scratch_dir("review_case");
        write_tier(&dir, Tier::Dev, &[scaled_var(2.0)]);
        write_tier(&dir, Tier::Final, &[scaled_var(99.0)]);

        let dev = Fence::<DevSide>::build(&dir);
        let final_fence = Fence::<FinalSide>::build(&dir);

        let candidate = scaled_var(3.0);
        assert!(
            blocked_by_either(&dev, &final_fence, candidate.entry()),
            "X * 3.0 must be fenced out by a DEV entry of X * 2.0 — both are the identical \
             input to the extraction head"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn genuinely_different_topology_is_not_fenced() {
        let dir = scratch_dir("different_topology");
        write_tier(&dir, Tier::Dev, &[scaled_var(2.0)]);
        write_tier(&dir, Tier::Final, &[scaled_var(99.0)]);

        let dev = Fence::<DevSide>::build(&dir);
        let final_fence = Fence::<FinalSide>::build(&dir);

        let mut b = ExprBuilder::new();
        let x = b.push_var(0);
        let c = b.push_const(2.0);
        let add = b.push_binary(OpKind::Add, x, c);
        let (rooted, _env) = b.finish(&[add]);
        assert!(
            !blocked_by_either(&dev, &final_fence, rooted.entry()),
            "X + 2.0 is a different op from the fenced X * 2.0 and must survive"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "holdout fence cannot be built")]
    fn fence_refuses_to_build_from_a_missing_tier_file() {
        let dir = scratch_dir("missing");
        let _ = Fence::<DevSide>::build(&dir);
    }

    #[test]
    #[should_panic(expected = "is empty (zero entries)")]
    fn fence_refuses_to_build_from_an_empty_tier_file() {
        // A zero-entry corpus is a valid PXCR file (the writer permits an
        // empty corpus), but a fence built from it would block nothing —
        // existence and a successful parse are not enough (P2 finding on the
        // fix commit for PR #1019).
        let dir = scratch_dir("empty_tier");
        write_tier(&dir, Tier::Dev, &[]);
        let _ = Fence::<DevSide>::build(&dir);
    }

    #[test]
    fn fence_reports_entries_and_distinct_keys_separately() {
        let dir = scratch_dir("entries_vs_keys");
        // Two entries, same feature-quotient structure: entries() counts
        // both, len() counts the one distinct key.
        write_tier(&dir, Tier::Dev, &[scaled_var(2.0), scaled_var(3.0)]);
        write_tier(&dir, Tier::Final, &[scaled_var(99.0)]);

        let dev = Fence::<DevSide>::build(&dir);
        assert_eq!(dev.entries(), 2);
        assert_eq!(dev.len(), 1);
        assert!(!dev.is_empty());

        std::fs::remove_dir_all(&dir).ok();
    }

    // Direction (J7 "backwards misuse must not compile") is exercised by the
    // `compile_fail` doc-test on [`Fence`]'s "# Direction" section, not a
    // `#[test]` here — `HoldoutSide` being sealed means the misuse has no
    // runtime form to assert against, only a type that fails to resolve.

    const VALID: &str = r#"
# leading comment
[train]
bands = [0, 2]  # trailing comment
seeds = [[0, 7]]

[dev]
bands = [1]
seeds = [[0, 7]]

[final]
bands = [3]
seeds = [[0, 3], [5, 6]]
kernels = ["swirl", "poly"]
"#;

    #[test]
    fn parses_valid_manifest_and_assigns_tiers() {
        let m = SplitManifest::parse(VALID).expect("valid manifest must parse");
        assert_eq!(m.tier_of(Family::new(0, 0)), Some(Tier::Train));
        assert_eq!(m.tier_of(Family::new(2, 7)), Some(Tier::Train));
        assert_eq!(m.tier_of(Family::new(1, 3)), Some(Tier::Dev));
        assert_eq!(m.tier_of(Family::new(3, 2)), Some(Tier::Final));
        assert_eq!(m.tier_of(Family::new(3, 5)), Some(Tier::Final));
        // Seed 4 falls in the gap between final's ranges.
        assert_eq!(m.tier_of(Family::new(3, 4)), None);
        // Band 9 is in no tier; seed 8 is outside every range.
        assert_eq!(m.tier_of(Family::new(9, 0)), None);
        assert_eq!(m.tier_of(Family::new(0, 8)), None);
        assert_eq!(m.final_kernels, vec!["swirl", "poly"]);
        assert_eq!(m.max_band_index(), 3);
        // train: 2 bands x 8 seeds; final: 1 band x (4 + 2) seeds.
        assert_eq!(m.train.family_count(), 16);
        assert_eq!(m.final_tier.family_count(), 6);
        let fams: Vec<Family> = m.final_tier.families().collect();
        assert_eq!(
            fams,
            vec![
                Family::new(3, 0),
                Family::new(3, 1),
                Family::new(3, 2),
                Family::new(3, 3),
                Family::new(3, 5),
                Family::new(3, 6),
            ]
        );
    }

    #[test]
    fn checked_in_manifest_is_valid() {
        let text = include_str!("../../corpus_split.toml");
        let m = SplitManifest::parse(text).expect("checked-in corpus_split.toml must be valid");
        assert_eq!(
            m.final_kernels.len(),
            17,
            "the five original named production kernels plus the withheld \
             12-kernel ShaderToy benchmark set (pixelflow_pipeline::shader_bench)"
        );
        assert_eq!(m.max_band_index(), 37, "band table has 38 bands (0..=37)");
        // Spot-check the by-family holdout: dev band 22 is dev at every seed,
        // and no train band collides with it.
        assert_eq!(m.tier_of(Family::new(22, 0)), Some(Tier::Dev));
        assert_eq!(m.tier_of(Family::new(22, 7)), Some(Tier::Dev));
        assert_eq!(m.tier_of(Family::new(12, 3)), Some(Tier::Final));
        assert_eq!(m.tier_of(Family::new(0, 3)), Some(Tier::Train));
    }

    #[test]
    fn shared_band_with_disjoint_seed_ranges_is_valid() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 7]]
[dev]
bands = [0]
seeds = [[8, 9]]
[final]
bands = [1]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let m = SplitManifest::parse(text).expect("disjoint seed ranges on a shared band are fine");
        assert_eq!(m.tier_of(Family::new(0, 7)), Some(Tier::Train));
        assert_eq!(m.tier_of(Family::new(0, 8)), Some(Tier::Dev));
    }

    #[test]
    fn cross_tier_overlap_is_rejected() {
        let text = r#"
[train]
bands = [0, 1]
seeds = [[0, 7]]
[dev]
bands = [1]
seeds = [[5, 9]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("overlap must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("overlap"), "unexpected error: {msg}");
        assert!(msg.contains("band 1"), "unexpected error: {msg}");
    }

    #[test]
    fn within_tier_seed_range_overlap_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 4], [4, 7]]
[dev]
bands = [1]
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("within-tier overlap must be rejected");
        assert!(
            err.to_string().contains("overlapping seed ranges"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn empty_tier_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 7]]
[dev]
bands = []
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("empty tier must be rejected");
        assert!(
            err.to_string().contains("[dev] has no bands"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn inverted_seed_range_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[7, 0]]
[dev]
bands = [1]
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("inverted range must be rejected");
        assert!(
            err.to_string().contains("inverted seed range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn missing_kernels_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 7]]
[dev]
bands = [1]
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
"#;
        let err = SplitManifest::parse(text).expect_err("missing kernels must be rejected");
        assert!(
            err.to_string().contains("missing `kernels`"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn duplicate_band_is_rejected() {
        let text = r#"
[train]
bands = [0, 0]
seeds = [[0, 7]]
[dev]
bands = [1]
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("duplicate band must be rejected");
        assert!(
            err.to_string().contains("band 0 twice"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unknown_key_and_unknown_section_are_rejected() {
        let err = SplitManifest::parse("[train]\nbandz = [0]\n").expect_err("unknown key");
        assert!(err.to_string().contains("unknown key"), "got: {err}");

        let err = SplitManifest::parse("[trainn]\n").expect_err("unknown section");
        assert!(err.to_string().contains("unknown section"), "got: {err}");
    }

    #[test]
    fn kernels_outside_final_is_rejected() {
        let err = SplitManifest::parse("[train]\nkernels = [\"swirl\"]\n")
            .expect_err("kernels key belongs to [final] only");
        assert!(err.to_string().contains("unknown key"), "got: {err}");
    }

    #[test]
    fn malformed_values_are_rejected_with_line_numbers() {
        let err = SplitManifest::parse("[train]\nbands = [0, x]\n").expect_err("bad integer");
        let msg = err.to_string();
        assert!(msg.contains("line 2"), "got: {msg}");

        let err =
            SplitManifest::parse("[train]\nseeds = [[0]]\n").expect_err("range needs two values");
        assert!(err.to_string().contains("exactly two"), "got: {err}");

        let err = SplitManifest::parse("[train]\nbands = [0\n").expect_err("unbalanced bracket");
        assert!(err.to_string().contains("array"), "got: {err}");
    }

    #[test]
    fn seed_range_contains_count_and_overlap_should_match_their_definitions() {
        let r = SeedRange { start: 2, end: 5 };
        assert!(r.contains(2) && r.contains(5) && !r.contains(6));
        assert_eq!(r.count(), 4);
        assert!(r.overlaps(SeedRange { start: 5, end: 9 }));
        assert!(!r.overlaps(SeedRange { start: 6, end: 9 }));
    }

    #[test]
    fn tier_ordering_reflects_sacredness() {
        assert!(Tier::Train < Tier::Dev);
        assert!(Tier::Dev < Tier::Final);
    }

    // ── [dev] families (round 1b, plan §6 step 1) ───────────────────────────

    #[test]
    fn manifest_without_families_key_parses_with_an_empty_list() {
        let m = SplitManifest::parse(VALID).expect("families is optional");
        assert!(m.dev_families.is_empty());
    }

    #[test]
    fn dev_families_key_parses() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 7]]
[dev]
bands = [1]
seeds = [[0, 7]]
families = ["sh", "bezier"]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let m = SplitManifest::parse(text).expect("valid dev families must parse");
        assert_eq!(m.dev_families, vec!["sh", "bezier"]);
    }

    #[test]
    fn families_key_under_train_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 7]]
families = ["sh"]
[dev]
bands = [1]
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("families under [train] must be rejected");
        assert!(err.to_string().contains("unknown key"), "got: {err}");
    }

    #[test]
    fn families_key_under_final_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 7]]
[dev]
bands = [1]
seeds = [[0, 7]]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
families = ["sh"]
"#;
        let err = SplitManifest::parse(text).expect_err("families under [final] must be rejected");
        assert!(err.to_string().contains("unknown key"), "got: {err}");
    }

    #[test]
    fn duplicate_dev_family_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 7]]
[dev]
bands = [1]
seeds = [[0, 7]]
families = ["sh", "sh"]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text).expect_err("duplicate family must be rejected");
        assert!(
            err.to_string().contains("family \"sh\" twice"),
            "got: {err}"
        );
    }

    #[test]
    fn dev_family_colliding_with_a_final_kernel_name_is_rejected() {
        let text = r#"
[train]
bands = [0]
seeds = [[0, 7]]
[dev]
bands = [1]
seeds = [[0, 7]]
families = ["swirl"]
[final]
bands = [2]
seeds = [[0, 0]]
kernels = ["swirl"]
"#;
        let err = SplitManifest::parse(text)
            .expect_err("a dev family sharing a name with a final kernel must be rejected");
        assert!(
            err.to_string().contains("cross-tier duplicate"),
            "got: {err}"
        );
    }

    #[test]
    fn checked_in_manifest_registers_the_sh_ood_family() {
        let text = include_str!("../../corpus_split.toml");
        let m = SplitManifest::parse(text).expect("checked-in manifest must be valid");
        assert!(
            m.dev_families.contains(&"sh".to_string()),
            "round 1b registers `sh` as a DEV-only OOD family: {:?}",
            m.dev_families
        );
    }

    #[test]
    fn checked_in_manifest_registers_the_bezier_ood_family() {
        let text = include_str!("../../corpus_split.toml");
        let m = SplitManifest::parse(text).expect("checked-in manifest must be valid");
        assert!(
            m.dev_families.contains(&"bezier".to_string()),
            "round 1b registers `bezier` as the polynomial-only DEV-only OOD control family: {:?}",
            m.dev_families
        );
    }
}
