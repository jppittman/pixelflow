# desloppify/src/skills

**Is.** A function from a directory to a finite map from name to text: the
shared review knowledge a rule names (desloppify/README.md, "Skills"). It is
a lookup and nothing else; what a rule does with the text is the rule's.
desloppify is not a pixelflow crate, so this cites its README, not DENOTATION.

## Owns
- What a skill is: a subdirectory of the skills directory that holds a
  `SKILL.md` (`SKILL_FILE`). Its name is the directory's name, its text the
  file's, verbatim.
- What is not a skill: a subdirectory without `SKILL.md` is skipped, and a
  missing skills directory is no skills, not an error.
- The failures of reading: an unreadable directory or file, or a directory
  name that is not UTF-8, is an `Err` naming the path.

## Never
- No interpretation of the text. `Skills::get` returns it verbatim; nothing
  here parses, trims, orders or merges it.
- No knowledge of rules, files under review, models or prompts. This module
  imports `std` and `anyhow` only; `rule::load_dir` asks it for names and
  `review` never does.
- No mutation after load: the map's field is `pub(super)`, so the only
  public read is `get`.
- No skill by default. A rule gets exactly the skills it names, in the order
  it names them (`rule/file.rs`); this module does not decide which apply.

## Contract
`mod.rs`: `Skills` (`get`), `load(dir) -> Result<Skills>`, `SKILL_FILE`
(`pub(crate)`). `load.rs` is the one implementation and is private.

## Enforced by
- Type: `Skills`'s map is not publicly constructible or mutable; `get` is its
  only reader.
- `desloppify/tests/rule.rs` `an_unknown_field_or_skill_is_refused`: a rule
  naming a skill that is not loaded is refused at load, before any call.
- `desloppify/tests/shipped_rules.rs` `every_shipped_rule_loads`: every skill
  a shipped rule names exists under `desloppify/skills/`.
- Nothing tests the skipped-subdirectory or missing-directory cases directly;
  `tests/rule.rs` reaches the missing directory only to get an empty set.

## Behind
- `desloppify/skills/pixelflow-architecture` and `style-guide` are distilled
  copies of CLAUDE.md and STYLE.md, regular files that drift from their
  source; `ontology` is already a symlink to `.claude/skills/ontology`
  (intent-lives-with-the-module §5, 2026-10-15).
- desloppify/README.md lists three skills (`style-guide`,
  `pixelflow-architecture`, `behavioral-contracts`) and omits `ontology`,
  which is loadable and named by no shipped rule. No plan or Linear issue
  closes this.

## Decided in
desloppify/README.md "Rules" (the `skills` field) and "Skills";
`docs/process/2026-10-15-intent-lives-with-the-module.md` §5 (skills as
symlinks to their source).
