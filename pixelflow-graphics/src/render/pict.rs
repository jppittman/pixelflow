// src/render/pict.rs

//! Proof-of-concept #2: PICT-style pairwise (combinatorial) testing for the
//! color/pixel layer of the renderer.
//!
//! This is the same dependency-free pairwise covering-array generator first
//! introduced for the ANSI/SGR layer (`core-term`'s `ansi::pict`). It is
//! duplicated here rather than shared through a crate because the workspace
//! deliberately keeps its dependency graph minimal — a ~90-line test helper
//! does not justify a new shared crate. If a third consumer appears, that is
//! the moment to extract it.
//!
//! See [`pairwise`] for the algorithm; the worked example lives in
//! [`super::pict_color_tests`].

/// A pairwise (2-way) covering-array generator.
///
/// Given the number of levels (distinct values) of each factor, returns a set
/// of rows — each row assigns one level index to every factor — such that for
/// any two factors `i, j` and any pair of levels `(a, b)`, at least one row has
/// `row[i] == a && row[j] == b`.
///
/// Deterministic one-row-at-a-time greedy heuristic (the core of AETG/PICT):
/// seed each new row with a still-uncovered pair, then fill the remaining
/// factors by greedily choosing the level that covers the most currently
/// uncovered pairs against the already-assigned factors. No RNG, so failures
/// reproduce exactly.
pub fn pairwise(level_counts: &[usize]) -> Vec<Vec<usize>> {
    use std::collections::BTreeSet;

    let n = level_counts.len();
    assert!(n >= 2, "pairwise testing needs at least two factors");
    assert!(
        level_counts.iter().all(|&c| c >= 1),
        "every factor needs at least one level"
    );

    // A pair is keyed (factor_i, level_a, factor_j, level_b) with i < j. Using a
    // BTreeSet keeps seed selection deterministic (always the smallest key).
    let mut uncovered: BTreeSet<(usize, usize, usize, usize)> = BTreeSet::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for a in 0..level_counts[i] {
                for b in 0..level_counts[j] {
                    uncovered.insert((i, a, j, b));
                }
            }
        }
    }

    let pair_key = |i: usize, a: usize, j: usize, b: usize| {
        if i < j {
            (i, a, j, b)
        } else {
            (j, b, i, a)
        }
    };

    let mut rows: Vec<Vec<usize>> = Vec::new();

    while let Some(&(si, sa, sj, sb)) = uncovered.iter().next() {
        // Seed the row with the smallest still-uncovered pair.
        let mut row: Vec<Option<usize>> = vec![None; n];
        row[si] = Some(sa);
        row[sj] = Some(sb);

        // Fill remaining factors in index order, greedily.
        for f in 0..n {
            if row[f].is_some() {
                continue;
            }
            let mut best_level = 0;
            let mut best_gain = usize::MAX; // sentinel forcing first real compare
            for level in 0..level_counts[f] {
                let mut gain = 0;
                for (g, assigned) in row.iter().enumerate() {
                    let Some(av) = *assigned else { continue };
                    let (i, a, j, b) = pair_key(f, level, g, av);
                    if uncovered.contains(&(i, a, j, b)) {
                        gain += 1;
                    }
                }
                if best_gain == usize::MAX || gain > best_gain {
                    best_gain = gain;
                    best_level = level;
                }
            }
            row[f] = Some(best_level);
        }

        let row: Vec<usize> = row
            .into_iter()
            .map(|v| v.expect("row fully assigned"))
            .collect();

        // Mark every pair this row realizes as covered.
        for i in 0..n {
            for j in (i + 1)..n {
                uncovered.remove(&(i, row[i], j, row[j]));
            }
        }
        rows.push(row);
    }

    rows
}

#[cfg(test)]
mod generator_tests {
    use super::pairwise;
    use std::collections::HashSet;

    fn assert_covers_all_pairs(level_counts: &[usize]) {
        let rows = pairwise(level_counts);
        let n = level_counts.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let mut seen: HashSet<(usize, usize)> = HashSet::new();
                for row in &rows {
                    seen.insert((row[i], row[j]));
                }
                for a in 0..level_counts[i] {
                    for b in 0..level_counts[j] {
                        assert!(
                            seen.contains(&(a, b)),
                            "pair (f{i}={a}, f{j}={b}) never covered for {level_counts:?}",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn covers_uniform_and_mixed_factors() {
        assert_covers_all_pairs(&[8, 8, 8, 8]);
        assert_covers_all_pairs(&[6, 4, 8, 2, 3]);
    }

    #[test]
    fn is_deterministic() {
        assert_eq!(pairwise(&[6, 6, 6]), pairwise(&[6, 6, 6]));
    }
}
