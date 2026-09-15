# Design Doc Template

## Metadata
- **Author**:
- **Status**: (required; choose exactly one — see the vocabulary below)
- **Created**: YYYY-MM-DD
- **Verified against**: `<commit sha>` (required for any document that makes claims
  about the tree — see below)
- **Reviewers**:

---

### Status vocabulary

Choose exactly one:

| Status | Means |
|---|---|
| `Draft` | Written, not agreed. Nothing is built. |
| `Proposed` | Offered for a decision. |
| `Registered` | A pre-registration: constants and decision rules committed **before** the data exists, and not revisable after the first run. Results are appended, never edited in. |
| `Plan of record` | The agreed direction for work not yet fully implemented. |
| `Landed` | Implemented and merged. Say which PRs in the body. |
| `Closed` | Ran to completion; the result is recorded and no further work follows. A negative result is a completed document, not an abandoned one. |
| `Superseded` | Replaced. **Must name the replacement**, and the replacement should say what it keeps. |
| `Historical` | Kept for rationale or archaeology. Not authoritative about the tree. |

This list replaced `Draft | Review | Approved | Implemented` on 2026-09-09. The old
four were followed by 3 of 59 plans and designs; the other 56 had invented roughly
twenty words of their own, and several of those — `Registered`, `Plan of record`,
`Landed` — carried more information than any of the sanctioned four. **The
vocabulary was wrong, not the authors.** Describe archival relationships with
`Supersedes:` / `Superseded by:` in addition to the status, not instead of it.

### Why `Verified against` is required

A status document pinned to a moving base is stale by default, and the fix is not
more diligence — it is recording *which tree* a claim was checked on, so a later
reader can tell without re-deriving. Every audit, ledger row and triage table this
repository has produced records *when* it was written and none recorded this; the
2026-09-08 open-PR sweep hit that failure four times in one run, twice because of
merges it made itself.

Resolve the sha, do not name a ref. `main` is a moving target and a local `main`
can be days behind — `git rev-parse HEAD` at the moment you read the code.

### If the document names source paths

`scripts/check-doc-paths.sh` (CI job `doc-paths`) fails when a plan, design or note
cites a `.rs` file the tree does not have. A document is fine if the path exists, if
the surrounding prose says the file is gone, or if it carries one of the archival
statuses above — so the status is load-bearing, not decoration. A path you intend to
*create* goes in `scripts/doc-paths-baseline.txt` with a one-line reason.

## 1. Overview

### 1.1 Problem Statement
_What problem are we solving? Why now?_

### 1.2 Goals
_What does success look like? Be specific._

### 1.3 Non-Goals
_What are we explicitly NOT doing?_

---

## 2. Background

### 2.1 Current State
_How does the system work today?_

### 2.2 Prior Art
_What existing solutions/papers/libraries informed this design?_

---

## 3. Design

### 3.1 Architecture
_High-level structure. Diagrams encouraged._

```
┌─────────┐     ┌─────────┐
│ Input   │────▶│ Output  │
└─────────┘     └─────────┘
```

### 3.2 Interfaces
_Define the contracts. This is what gets copied into issues._

```rust
pub trait Foo {
    fn bar(&self) -> Result<Baz>;
}
```

### 3.3 Data Flow
_How does data move through the system?_

### 3.4 Error Handling
_What can go wrong? How do we handle it?_

---

## 4. Implementation Plan

### 4.1 Task Breakdown

| Task | File(s) | Deps | Estimate | Assignee |
|------|---------|------|----------|----------|
| T1: Description | `path/file.rs` | None | S/M/L | Jules/Claude |
| T2: Description | `path/other.rs` | T1 | S/M/L | Jules/Claude |

### 4.2 Parallelization
_Which tasks can run concurrently?_

```
T1 ──────────────┐
                 ├──▶ T4 (integration)
T2 ──┬──▶ T3 ───┘
     │
     └──▶ (blocked)
```

### 4.3 Risk Assessment
_What could go wrong? Mitigation?_

---

## 5. Testing Strategy

### 5.1 Unit Tests
_Per-task test requirements._

### 5.2 Integration Tests
_End-to-end validation._

---

## 6. Alternatives Considered

| Alternative | Pros | Cons | Why Not |
|-------------|------|------|---------|
| Option A | ... | ... | ... |

---

## 7. Open Questions
_Decisions that need input before proceeding._

- [ ] Question 1?
- [ ] Question 2?
