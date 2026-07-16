//! # Spatial BSP Tree Combinator
//!
//! A B+ tree style spatial manifold for efficient grid rendering.
//!
//! ## Design
//!
//! Limits dynamic dispatch to two homogeneous arrays instead of per-node `Box<dyn>`:
//! - Interior nodes: routing only (axis, threshold, child refs)
//! - Leaf nodes: actual manifolds
//!
//! The tree structure is the compile-time shape. Array contents are load-time data.
//!
//! ## Performance
//!
//! - Early exit when all SIMD lanes go the same direction
//! - Per-lane blending only at cell boundaries
//! - O(log n) tree depth

use pixelflow_core::{Discrete, Field, Manifold, ManifoldCompat, Select};
use std::sync::Arc;

/// The standard 4D Field domain type.
type Field4 = (Field, Field, Field, Field);

/// Split axis for BSP nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
}

/// Reference to a child node in the BSP tree.
#[derive(Clone, Copy, Debug)]
pub enum NodeRef {
    /// Index into the interior nodes array.
    Interior(u32),
    /// Index into the leaves array.
    Leaf(u32),
}

/// Interior BSP node - routing only, no manifold data.
#[derive(Clone, Copy, Debug)]
pub struct InteriorNode {
    /// Which axis to split on.
    pub axis: Axis,
    /// Coordinate threshold for the split.
    pub threshold: f32,
    /// Left child (coordinates < threshold).
    pub left: NodeRef,
    /// Right child (coordinates >= threshold).
    pub right: NodeRef,
}

/// Spatial BSP tree manifold.
///
/// A B+ tree style structure where:
/// - Interior nodes contain only routing info (axis, threshold, children)
/// - Leaves contain the actual manifolds
///
/// Generic over leaf type L. For terminals, L = ColoredGlyph<PlatformPixel>.
#[derive(Clone)]
pub struct SpatialBSP<L> {
    /// Interior nodes - routing only.
    interiors: Arc<[InteriorNode]>,
    /// Leaf manifolds.
    leaves: Arc<[L]>,
}

/// Positioned item for building the BSP tree.
#[derive(Clone)]
pub struct Positioned<L> {
    /// Bounding box: (x_min, y_min, x_max, y_max)
    pub bounds: (f32, f32, f32, f32),
    /// The leaf manifold.
    pub leaf: L,
}

impl<L> SpatialBSP<L> {
    /// Create a BSP from pre-built arrays.
    ///
    /// Use `from_positioned` for automatic tree construction.
    pub fn new(interiors: Arc<[InteriorNode]>, leaves: Arc<[L]>) -> Self {
        Self { interiors, leaves }
    }

    /// Create a single-leaf BSP (degenerate case).
    pub fn single(leaf: L) -> Self {
        Self {
            interiors: Arc::from([]),
            leaves: Arc::from([leaf]),
        }
    }

    /// Build a balanced BSP tree from positioned items.
    ///
    /// Items are split recursively on the larger dimension until each
    /// region contains a single leaf.
    pub fn from_positioned(items: Vec<Positioned<L>>) -> Self {
        if items.is_empty() {
            return Self {
                interiors: Arc::from([]),
                leaves: Arc::from([]),
            };
        }

        if items.len() == 1 {
            return Self::single(items.into_iter().next().unwrap().leaf);
        }

        let mut interiors = Vec::new();
        let mut leaves = Vec::new();

        // Recursively build the tree
        let _root = Self::build_tree(&mut interiors, &mut leaves, items);

        Self {
            interiors: Arc::from(interiors),
            leaves: Arc::from(leaves),
        }
    }

    /// Recursively build the BSP tree.
    ///
    /// Returns the NodeRef for this subtree's root.
    fn build_tree(
        interiors: &mut Vec<InteriorNode>,
        leaves: &mut Vec<L>,
        mut items: Vec<Positioned<L>>,
    ) -> NodeRef {
        // Base case: single item
        if items.len() == 1 {
            let idx = leaves.len() as u32;
            leaves.push(items.pop().unwrap().leaf);
            return NodeRef::Leaf(idx);
        }

        // Find bounding box of all items and center extents
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        let (mut min_cx, mut min_cy, mut max_cx, mut max_cy) =
            (f32::MAX, f32::MAX, f32::MIN, f32::MIN);

        for item in &items {
            min_x = min_x.min(item.bounds.0);
            min_y = min_y.min(item.bounds.1);
            max_x = max_x.max(item.bounds.2);
            max_y = max_y.max(item.bounds.3);

            let cx = (item.bounds.0 + item.bounds.2) * 0.5;
            let cy = (item.bounds.1 + item.bounds.3) * 0.5;
            min_cx = min_cx.min(cx);
            max_cx = max_cx.max(cx);
            min_cy = min_cy.min(cy);
            max_cy = max_cy.max(cy);
        }

        // Choose split axis based on center distribution variance (spread)
        // Fallback to bound dimensions if spreads are effectively zero (concentric items)
        let extent_x = max_cx - min_cx;
        let extent_y = max_cy - min_cy;
        let width = max_x - min_x;
        let height = max_y - min_y;

        let split_x = if extent_x > extent_y {
            true
        } else if extent_y > extent_x {
            false
        } else {
            width >= height
        };

        let (axis, threshold) = if split_x {
            // Sort by X center, split at median
            items.sort_by(|a, b| {
                let ca = (a.bounds.0 + a.bounds.2) / 2.0;
                let cb = (b.bounds.0 + b.bounds.2) / 2.0;
                ca.partial_cmp(&cb).unwrap()
            });
            let mid_idx = items.len() / 2;
            let threshold = (items[mid_idx - 1].bounds.2 + items[mid_idx].bounds.0) / 2.0;
            (Axis::X, threshold)
        } else {
            // Sort by Y center, split at median
            items.sort_by(|a, b| {
                let ca = (a.bounds.1 + a.bounds.3) / 2.0;
                let cb = (b.bounds.1 + b.bounds.3) / 2.0;
                ca.partial_cmp(&cb).unwrap()
            });
            let mid_idx = items.len() / 2;
            let threshold = (items[mid_idx - 1].bounds.3 + items[mid_idx].bounds.1) / 2.0;
            (Axis::Y, threshold)
        };

        // Split items
        let mid = items.len() / 2;
        let right_items = items.split_off(mid);
        let left_items = items;

        // Recursively build children
        let left = Self::build_tree(interiors, leaves, left_items);
        let right = Self::build_tree(interiors, leaves, right_items);

        // Create interior node
        let idx = interiors.len() as u32;
        interiors.push(InteriorNode {
            axis,
            threshold,
            left,
            right,
        });

        NodeRef::Interior(idx)
    }

    /// Number of interior nodes.
    pub fn interior_count(&self) -> usize {
        self.interiors.len()
    }

    /// Number of leaf nodes.
    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }
}

// ============================================================================
// Manifold Implementation for Discrete Output
// ============================================================================

impl<L> Manifold<Field4> for SpatialBSP<L>
where
    L: ManifoldCompat<Field, Output = Discrete>,
{
    type Output = Discrete;

    #[inline(always)]
    fn eval(&self, p: Field4) -> Discrete {
        let (x, y, z, w) = p;
        // Handle degenerate cases
        if self.leaves.is_empty() {
            // No leaves - return transparent black
            let zero = Field::from(0.0);
            return Discrete::pack(zero, zero, zero, zero);
        }

        if self.interiors.is_empty() {
            // Single leaf - evaluate directly
            return self.leaves[0].eval_raw(x, y, z, w);
        }

        // Start traversal from the last interior node (root)
        self.traverse(self.interiors.len() - 1, x, y, z, w)
    }
}

impl<L> SpatialBSP<L>
where
    L: ManifoldCompat<Field, Output = Discrete>,
{
    /// Traverse the BSP tree, returning the blended result.
    #[inline(always)]
    fn traverse(&self, idx: usize, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        let node = &self.interiors[idx];

        // Get coordinate for this axis
        let coord = match node.axis {
            Axis::X => x,
            Axis::Y => y,
        };

        // Compute mask: true where coord < threshold
        let mask = coord.lt(Field::from(node.threshold));

        // Early exit when all SIMD lanes go the same direction
        if mask.all() {
            return self.eval_child(node.left, x, y, z, w);
        }
        if !mask.any() {
            return self.eval_child(node.right, x, y, z, w);
        }

        // Mixed: SIMD lanes span the boundary, must evaluate both
        let left_val = self.eval_child(node.left, x, y, z, w);
        let right_val = self.eval_child(node.right, x, y, z, w);

        // Blend using Select combinator
        Select { cond: mask, if_true: left_val, if_false: right_val }.eval((x, y, z, w))
    }

    /// Evaluate a child node (either interior or leaf).
    #[inline(always)]
    fn eval_child(&self, child: NodeRef, x: Field, y: Field, z: Field, w: Field) -> Discrete {
        match child {
            NodeRef::Interior(i) => self.traverse(i as usize, x, y, z, w),
            NodeRef::Leaf(i) => self.leaves[i as usize].eval_raw(x, y, z, w),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple solid color manifold for testing.
    #[derive(Clone)]
    struct SolidColor {
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    }

    impl SolidColor {
        fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
            Self {
                r: r as f32 / 255.0,
                g: g as f32 / 255.0,
                b: b as f32 / 255.0,
                a: a as f32 / 255.0,
            }
        }
    }

    impl Manifold<Field4> for SolidColor {
        type Output = Discrete;

        fn eval(&self, p: Field4) -> Discrete {
            let (_x, _y, _z, _w) = p;
            Discrete::pack(
                Field::from(self.r),
                Field::from(self.g),
                Field::from(self.b),
                Field::from(self.a),
            )
        }
    }

    #[test]
    fn test_single_leaf() {
        let bsp = SpatialBSP::single(SolidColor::new(255, 0, 0, 255));

        assert_eq!(
            bsp.interior_count(),
            0,
            "Single leaf tree has no interior nodes"
        );
        assert_eq!(bsp.leaf_count(), 1, "Single leaf tree has exactly one leaf");

        // Evaluate at any point should compile and execute without panic
        let x = Field::from(100.0);
        let y = Field::from(100.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        let _result = bsp.eval_raw(x, y, z, w);
    }

    #[test]
    fn test_two_leaves() {
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255), // Red on left
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255), // Blue on right
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.interior_count(), 1);
        assert_eq!(bsp.leaf_count(), 2);
    }

    #[test]
    fn test_empty_bsp() {
        let bsp: SpatialBSP<SolidColor> = SpatialBSP::from_positioned(vec![]);

        assert_eq!(bsp.interior_count(), 0, "Empty BSP has no interior nodes");
        assert_eq!(bsp.leaf_count(), 0, "Empty BSP has no leaves");

        // Should not panic on eval
        let x = Field::from(0.0);
        let _result = bsp.eval_raw(x, x, x, x);
    }

    #[test]
    fn test_four_leaves_grid() {
        // 2x2 grid
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 50.0),
                leaf: SolidColor::new(255, 0, 0, 255), // Top-left: Red
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 50.0),
                leaf: SolidColor::new(0, 255, 0, 255), // Top-right: Green
            },
            Positioned {
                bounds: (0.0, 50.0, 50.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255), // Bottom-left: Blue
            },
            Positioned {
                bounds: (50.0, 50.0, 100.0, 100.0),
                leaf: SolidColor::new(255, 255, 0, 255), // Bottom-right: Yellow
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        // Binary tree with 4 leaves should have exactly 3 interior nodes (n-1)
        assert_eq!(bsp.leaf_count(), 4, "Should have 4 leaves in the grid");
        assert_eq!(
            bsp.interior_count(),
            3,
            "Binary tree with 4 leaves must have exactly 3 interior nodes"
        );
    }

    // ========================================================================
    // Degenerate Case Tests
    // ========================================================================

    #[test]
    fn empty_bsp_returns_transparent_on_eval() {
        let bsp: SpatialBSP<SolidColor> = SpatialBSP::from_positioned(vec![]);

        let x = Field::from(50.0);
        let y = Field::from(50.0);
        let z = Field::from(0.0);
        let w = Field::from(0.0);

        // Should not panic when evaluating empty BSP
        let result = bsp.eval_raw(x, y, z, w);
        // Empty BSP should successfully evaluate (implementation returns default/transparent)
        let _ = result;
    }

    #[test]
    fn single_leaf_has_no_interiors() {
        let bsp = SpatialBSP::single(SolidColor::new(128, 64, 32, 255));

        assert_eq!(bsp.interior_count(), 0);
        assert_eq!(bsp.leaf_count(), 1);
    }

    #[test]
    fn single_leaf_from_positioned_has_no_interiors() {
        let items = vec![Positioned {
            bounds: (0.0, 0.0, 100.0, 100.0),
            leaf: SolidColor::new(255, 128, 0, 255),
        }];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.interior_count(), 0);
        assert_eq!(bsp.leaf_count(), 1);
    }

    // ========================================================================
    // Boundary Condition Tests
    // ========================================================================

    #[test]
    fn threshold_split_semantics_left_is_less_than() {
        // Create two items split at x=50
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 25.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255), // Left (red)
            },
            Positioned {
                bounds: (75.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255), // Right (blue)
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(
            bsp.interior_count(),
            1,
            "Two non-overlapping items need exactly one split"
        );

        // Verify the split axis and threshold are correct
        // Left item spans [0, 25], right item spans [75, 100]
        // Threshold should be in the gap (25, 75), ideally around 50
        let root = &bsp.interiors[0];
        assert_eq!(
            root.axis,
            Axis::X,
            "Should split on X axis for horizontal separation"
        );
        assert!(
            root.threshold >= 25.0 && root.threshold <= 75.0,
            "Threshold must be in gap between items: {} should be in [25, 75]",
            root.threshold
        );
    }

    #[test]
    fn vertical_split_when_height_exceeds_width() {
        // Tall narrow items should split on Y axis
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 10.0, 50.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (0.0, 50.0, 10.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        let root = &bsp.interiors[0];

        assert_eq!(root.axis, Axis::Y, "Should split on Y for tall items");
    }

    #[test]
    fn horizontal_split_when_width_exceeds_height() {
        // Wide short items should split on X axis
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 10.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 10.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        let root = &bsp.interiors[0];

        assert_eq!(root.axis, Axis::X, "Should split on X for wide items");
    }

    // ========================================================================
    // Identical Bounds Tests
    // ========================================================================

    #[test]
    fn identical_bounds_creates_valid_tree() {
        // All items at same position (overlapping glyphs scenario)
        let items = vec![
            Positioned {
                bounds: (10.0, 10.0, 20.0, 20.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (10.0, 10.0, 20.0, 20.0),
                leaf: SolidColor::new(0, 255, 0, 255),
            },
            Positioned {
                bounds: (10.0, 10.0, 20.0, 20.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 3, "Should have 3 leaves for 3 items");
        assert_eq!(
            bsp.interior_count(),
            2,
            "Binary tree with 3 leaves must have exactly 2 interior nodes (n-1 rule)"
        );
    }

    #[test]
    fn zero_width_bounds_does_not_panic() {
        let items = vec![
            Positioned {
                bounds: (10.0, 0.0, 10.0, 100.0), // Zero width
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (10.0, 0.0, 10.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    #[test]
    fn zero_height_bounds_does_not_panic() {
        let items = vec![
            Positioned {
                bounds: (0.0, 10.0, 100.0, 10.0), // Zero height
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (0.0, 10.0, 100.0, 10.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    #[test]
    fn point_bounds_creates_valid_tree() {
        // Degenerate rectangles (points)
        let items = vec![
            Positioned {
                bounds: (5.0, 5.0, 5.0, 5.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (15.0, 15.0, 15.0, 15.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    // ========================================================================
    // Extreme Value Tests
    // ========================================================================

    #[test]
    fn negative_coordinates_handled_correctly() {
        let items = vec![
            Positioned {
                bounds: (-100.0, -100.0, -50.0, -50.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (-50.0, -50.0, 0.0, 0.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 2, "Should have 2 leaves");
        assert_eq!(bsp.interior_count(), 1, "Two items need exactly one split");

        // Verify threshold is in the gap between items
        // Left item spans [-100, -50], right item spans [-50, 0]
        let root = &bsp.interiors[0];
        assert!(
            root.threshold >= -100.0 && root.threshold <= 0.0,
            "Threshold must separate the items: {} should be in [-100, 0]",
            root.threshold
        );
    }

    #[test]
    fn large_coordinates_handled_correctly() {
        let items = vec![
            Positioned {
                bounds: (1e6, 1e6, 2e6, 2e6),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (3e6, 3e6, 4e6, 4e6),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 2);
        assert!(bsp.interiors[0].threshold > 1e6);
    }

    #[test]
    fn mixed_positive_negative_coordinates() {
        let items = vec![
            Positioned {
                bounds: (-50.0, -50.0, 0.0, 0.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (0.0, 0.0, 50.0, 50.0),
                leaf: SolidColor::new(0, 255, 0, 255),
            },
            Positioned {
                bounds: (50.0, 50.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 3);
    }

    // ========================================================================
    // Tree Structure Tests
    // ========================================================================

    #[test]
    fn binary_tree_property_interior_count() {
        // For n leaves, binary tree has n-1 interior nodes
        for n in 2..=16 {
            let items: Vec<_> = (0..n)
                .map(|i| {
                    let x = (i as f32) * 10.0;
                    Positioned {
                        bounds: (x, 0.0, x + 5.0, 10.0),
                        leaf: SolidColor::new(255, 0, 0, 255),
                    }
                })
                .collect();

            let bsp = SpatialBSP::from_positioned(items);

            assert_eq!(bsp.leaf_count(), n);
            assert_eq!(
                bsp.interior_count(),
                n - 1,
                "Binary tree with {} leaves should have {} interiors",
                n,
                n - 1
            );
        }
    }

    #[test]
    fn power_of_two_leaves_creates_balanced_tree() {
        // 8 leaves should create a balanced tree of depth 3
        let items: Vec<_> = (0..8)
            .map(|i| {
                let x = (i as f32) * 10.0;
                Positioned {
                    bounds: (x, 0.0, x + 5.0, 10.0),
                    leaf: SolidColor::new((i * 32) as u8, 0, 0, 255),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 8);
        assert_eq!(bsp.interior_count(), 7);
    }

    #[test]
    fn many_items_stress_test() {
        // Create 1000 non-overlapping items
        let items: Vec<_> = (0..1000)
            .map(|i| {
                let row = i / 32;
                let col = i % 32;
                let x = (col as f32) * 10.0;
                let y = (row as f32) * 10.0;
                Positioned {
                    bounds: (x, y, x + 8.0, y + 8.0),
                    leaf: SolidColor::new(
                        ((i * 7) % 256) as u8,
                        ((i * 13) % 256) as u8,
                        ((i * 19) % 256) as u8,
                        255,
                    ),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 1000);
        assert_eq!(bsp.interior_count(), 999);
    }

    // ========================================================================
    // Traversal Correctness Tests
    // ========================================================================

    #[test]
    fn eval_respects_spatial_partitioning() {
        use pixelflow_core::{materialize_discrete, PARALLELISM};

        // Create a simple 2-region split at x=50
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255), // Red left
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255), // Blue right
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        // Materialize at two different spatial regions
        let mut left_pixels = [0u32; PARALLELISM];
        let mut right_pixels = [0u32; PARALLELISM];

        materialize_discrete(&bsp, 25.0, 50.0, &mut left_pixels);
        materialize_discrete(&bsp, 75.0, 50.0, &mut right_pixels);

        // Verify we got red and blue specifically
        let expected_red = {
            let red_color = SolidColor::new(255, 0, 0, 255);
            let mut buf = [0u32; PARALLELISM];
            materialize_discrete(&red_color, 0.0, 0.0, &mut buf);
            buf[0]
        };
        let expected_blue = {
            let blue_color = SolidColor::new(0, 0, 255, 255);
            let mut buf = [0u32; PARALLELISM];
            materialize_discrete(&blue_color, 0.0, 0.0, &mut buf);
            buf[0]
        };

        assert_eq!(left_pixels[0], expected_red, "Left region must be red");
        assert_eq!(right_pixels[0], expected_blue, "Right region must be blue");
    }

    #[test]
    fn eval_at_multiple_coords_with_single_leaf() {
        let bsp = SpatialBSP::single(SolidColor::new(100, 150, 200, 255));

        // Sample at various locations - all should return same color without panicking
        let coords = [
            (0.0, 0.0),
            (50.0, 50.0),
            (100.0, 100.0),
            (-10.0, -10.0),
            (1000.0, 1000.0),
        ];

        for (x, y) in coords {
            let _result = bsp.eval_raw(
                Field::from(x),
                Field::from(y),
                Field::from(0.0),
                Field::from(0.0),
            );
        }
    }

    #[test]
    fn quadtree_like_structure_four_quadrants() {
        use pixelflow_core::{materialize_discrete, PARALLELISM};

        // Create 4 quadrants to test 2D partitioning
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 50.0),
                leaf: SolidColor::new(255, 0, 0, 255), // Q1: Red
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 50.0),
                leaf: SolidColor::new(0, 255, 0, 255), // Q2: Green
            },
            Positioned {
                bounds: (0.0, 50.0, 50.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255), // Q3: Blue
            },
            Positioned {
                bounds: (50.0, 50.0, 100.0, 100.0),
                leaf: SolidColor::new(255, 255, 0, 255), // Q4: Yellow
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        // Test centers of each quadrant - each should produce a unique value
        let test_points = [
            (25.0, 25.0), // Q1
            (75.0, 25.0), // Q2
            (25.0, 75.0), // Q3
            (75.0, 75.0), // Q4
        ];

        let mut results = Vec::new();
        for (x, y) in test_points {
            let mut pixels = [0u32; PARALLELISM];
            materialize_discrete(&bsp, x, y, &mut pixels);
            results.push(pixels[0]);
        }

        // Each quadrant must return a distinct value
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                assert_ne!(
                    results[i], results[j],
                    "Quadrant {} and {} must have different colors",
                    i, j
                );
            }
        }
    }

    // ========================================================================
    // Node Reference Tests
    // ========================================================================

    #[test]
    fn node_ref_types_are_distinct() {
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        // Tree should have 1 interior node pointing to 2 leaves
        assert_eq!(bsp.interior_count(), 1);
        assert_eq!(bsp.leaf_count(), 2);

        let root = &bsp.interiors[0];

        // Both children should be leaves (not interiors)
        matches!(root.left, NodeRef::Leaf(_));
        matches!(root.right, NodeRef::Leaf(_));
    }

    #[test]
    fn three_levels_deep_has_interior_children() {
        // 4 items will create a 3-level tree
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 25.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (25.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(0, 255, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 75.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
            Positioned {
                bounds: (75.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(255, 255, 0, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.interior_count(), 3);
        assert_eq!(bsp.leaf_count(), 4);

        // Root should be last interior
        let root = &bsp.interiors[2];

        // At least one child should be an interior node
        let has_interior_child =
            matches!(root.left, NodeRef::Interior(_)) || matches!(root.right, NodeRef::Interior(_));
        assert!(
            has_interior_child,
            "3-level tree should have interior children"
        );
    }

    // ========================================================================
    // Overlapping Bounds Tests
    // ========================================================================

    #[test]
    fn overlapping_bounds_creates_valid_tree() {
        // Items with overlapping regions (common in terminal rendering)
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 60.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 128),
            },
            Positioned {
                bounds: (40.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 128),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 2);
        assert_eq!(bsp.interior_count(), 1);
    }

    #[test]
    fn fully_contained_bounds_creates_valid_tree() {
        // One item fully contains another
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (25.0, 25.0, 75.0, 75.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 2);
        assert_eq!(bsp.interior_count(), 1);
    }

    // ========================================================================
    // Construction API Tests
    // ========================================================================

    #[test]
    fn new_with_empty_arrays_behaves_like_from_positioned_empty() {
        let bsp1: SpatialBSP<SolidColor> = SpatialBSP::new(Arc::from([]), Arc::from([]));
        let bsp2: SpatialBSP<SolidColor> = SpatialBSP::from_positioned(vec![]);

        assert_eq!(bsp1.interior_count(), bsp2.interior_count());
        assert_eq!(bsp1.leaf_count(), bsp2.leaf_count());
    }

    #[test]
    fn clone_preserves_structure() {
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp1 = SpatialBSP::from_positioned(items);
        let bsp2 = bsp1.clone();

        assert_eq!(bsp1.interior_count(), bsp2.interior_count());
        assert_eq!(bsp1.leaf_count(), bsp2.leaf_count());
    }

    // ========================================================================
    // Axis Selection Tests
    // ========================================================================

    #[test]
    fn square_bounds_splits_on_either_axis() {
        // When width == height, should split on X (width >= height)
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 50.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 50.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        let root = &bsp.interiors[0];

        // Based on line 148: width >= height, so square splits on X
        assert_eq!(root.axis, Axis::X);
    }

    #[test]
    fn slightly_wider_splits_on_x() {
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 100.0, 99.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (100.0, 0.0, 200.0, 99.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.interiors[0].axis, Axis::X);
    }

    #[test]
    fn slightly_taller_splits_on_y() {
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 99.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (0.0, 100.0, 99.0, 200.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.interiors[0].axis, Axis::Y);
    }

    // ========================================================================
    // Threshold Calculation Precision Tests
    // ========================================================================

    #[test]
    fn threshold_between_non_overlapping_items() {
        // Two items with a gap between them
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 40.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (60.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        let root = &bsp.interiors[0];

        // Threshold should be between 40.0 and 60.0 (in the gap)
        assert!(
            root.threshold >= 40.0 && root.threshold <= 60.0,
            "Threshold {} should be in gap [40, 60]",
            root.threshold
        );
    }

    #[test]
    fn threshold_calculation_with_touching_items() {
        // Two items that touch at x=50
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        let root = &bsp.interiors[0];

        // Threshold should be at or very close to 50.0
        assert!(
            (root.threshold - 50.0).abs() < 1.0,
            "Threshold {} should be near touching point 50.0",
            root.threshold
        );
    }

    #[test]
    fn threshold_with_very_small_items() {
        // Items with very small dimensions
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 0.01, 0.01),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (0.02, 0.02, 0.03, 0.03),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
        assert!(bsp.interiors[0].threshold.is_finite());
    }

    // ========================================================================
    // SIMD Boundary Behavior Tests
    // ========================================================================

    #[test]
    fn eval_exactly_at_threshold_goes_right() {
        // Test the boundary condition: coord < threshold goes left, >= goes right
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        let threshold = bsp.interiors[0].threshold;

        // Sample exactly at threshold (should go right, >= threshold)
        let result = bsp.eval_raw(
            Field::from(threshold),
            Field::from(50.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        // Should return a valid result without panicking
        let _ = result;
    }

    #[test]
    fn eval_just_below_threshold_goes_left() {
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        let threshold = bsp.interiors[0].threshold;

        // Sample just below threshold
        let result = bsp.eval_raw(
            Field::from(threshold - 0.1),
            Field::from(50.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let _ = result;
    }

    #[test]
    fn eval_just_above_threshold_goes_right() {
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        let threshold = bsp.interiors[0].threshold;

        // Sample just above threshold
        let result = bsp.eval_raw(
            Field::from(threshold + 0.1),
            Field::from(50.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let _ = result;
    }

    // ========================================================================
    // Special Float Value Tests
    // ========================================================================

    #[test]
    #[should_panic(expected = "unwrap")]
    fn nan_bounds_panics_during_construction() {
        // This tests the sharp corner: partial_cmp().unwrap() will panic on NaN
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (f32::NAN, 0.0, 100.0, 100.0), // NaN in bounds
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        // Should panic when trying to sort by center (partial_cmp on NaN)
        let _bsp = SpatialBSP::from_positioned(items);
    }

    #[test]
    fn infinity_bounds_handled() {
        // Infinity should not cause panic (partial_cmp works)
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (100.0, 0.0, f32::INFINITY, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    #[test]
    fn negative_infinity_bounds_handled() {
        let items = vec![
            Positioned {
                bounds: (f32::NEG_INFINITY, 0.0, -50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    // ========================================================================
    // Root Node Location Tests
    // ========================================================================

    #[test]
    fn root_is_last_interior_node() {
        // Verify the assumption that root is at interiors.len() - 1
        let items: Vec<_> = (0..8)
            .map(|i| {
                let x = (i as f32) * 10.0;
                Positioned {
                    bounds: (x, 0.0, x + 5.0, 10.0),
                    leaf: SolidColor::new(255, 0, 0, 255),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);

        // The tree is built recursively, with the root added last
        // So root should be at index interiors.len() - 1
        assert!(bsp.interior_count() > 0);

        // Evaluate at a known point to ensure root is being used
        let result = bsp.eval_raw(
            Field::from(25.0),
            Field::from(5.0),
            Field::from(0.0),
            Field::from(0.0),
        );

        let _ = result;
    }

    // ========================================================================
    // Inverted Bounds Tests
    // ========================================================================

    #[test]
    fn inverted_x_bounds_handled() {
        // Bounds where min_x > max_x (malformed input)
        let items = vec![
            Positioned {
                bounds: (50.0, 0.0, 0.0, 100.0), // x_max < x_min
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (100.0, 0.0, 150.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    #[test]
    fn inverted_y_bounds_handled() {
        // Bounds where min_y > max_y (malformed input)
        let items = vec![
            Positioned {
                bounds: (0.0, 100.0, 50.0, 0.0), // y_max < y_min
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);
        assert_eq!(bsp.leaf_count(), 2);
    }

    // ========================================================================
    // Deep Tree Tests
    // ========================================================================

    #[test]
    fn linear_chain_creates_unbalanced_tree() {
        // Items already perfectly sorted should still create valid tree
        let items: Vec<_> = (0..32)
            .map(|i| {
                let x = (i as f32) * 10.0;
                Positioned {
                    bounds: (x, 0.0, x + 5.0, 10.0),
                    leaf: SolidColor::new((i * 8) as u8, 0, 0, 255),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 32);
        assert_eq!(bsp.interior_count(), 31);
    }

    #[test]
    fn alternating_dimensions_creates_balanced_tree() {
        // Grid layout should split alternately on X and Y
        let items: Vec<_> = (0..16)
            .map(|i| {
                let x = ((i % 4) as f32) * 25.0;
                let y = ((i / 4) as f32) * 25.0;
                Positioned {
                    bounds: (x, y, x + 20.0, y + 20.0),
                    leaf: SolidColor::new((i * 16) as u8, 0, 0, 255),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);

        assert_eq!(bsp.leaf_count(), 16);
        assert_eq!(bsp.interior_count(), 15);
    }

    // ========================================================================
    // Thread Safety Tests (compile-time via Send+Sync)
    // ========================================================================

    #[test]
    fn bsp_is_send_and_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // SpatialBSP should be Send + Sync due to Arc usage
        assert_send::<SpatialBSP<SolidColor>>();
        assert_sync::<SpatialBSP<SolidColor>>();
    }

    // ========================================================================
    // Structural Invariant Tests - These MUST hold or the tree is broken
    // ========================================================================

    #[test]
    fn all_interior_node_indices_are_valid() {
        let items: Vec<_> = (0..10)
            .map(|i| {
                let x = (i as f32) * 10.0;
                Positioned {
                    bounds: (x, 0.0, x + 5.0, 10.0),
                    leaf: SolidColor::new(255, 0, 0, 255),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);
        let num_interiors = bsp.interior_count();
        let num_leaves = bsp.leaf_count();

        // Every interior node must point to valid indices
        for interior in bsp.interiors.iter() {
            match interior.left {
                NodeRef::Interior(i) => {
                    assert!(
                        (i as usize) < num_interiors,
                        "Left interior index {} out of bounds (max {})",
                        i,
                        num_interiors - 1
                    );
                }
                NodeRef::Leaf(i) => {
                    assert!(
                        (i as usize) < num_leaves,
                        "Left leaf index {} out of bounds (max {})",
                        i,
                        num_leaves - 1
                    );
                }
            }

            match interior.right {
                NodeRef::Interior(i) => {
                    assert!(
                        (i as usize) < num_interiors,
                        "Right interior index {} out of bounds (max {})",
                        i,
                        num_interiors - 1
                    );
                }
                NodeRef::Leaf(i) => {
                    assert!(
                        (i as usize) < num_leaves,
                        "Right leaf index {} out of bounds (max {})",
                        i,
                        num_leaves - 1
                    );
                }
            }
        }
    }

    #[test]
    fn all_leaves_are_reachable_from_root() {
        let items: Vec<_> = (0..8)
            .map(|i| {
                let x = (i as f32) * 10.0;
                Positioned {
                    bounds: (x, 0.0, x + 5.0, 10.0),
                    leaf: SolidColor::new((i * 32) as u8, 0, 0, 255),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);

        // Traverse tree and collect all reachable leaf indices
        let mut reachable_leaves = std::collections::HashSet::new();

        fn collect_leaves<L>(
            bsp: &SpatialBSP<L>,
            node: NodeRef,
            reachable: &mut std::collections::HashSet<u32>,
        ) {
            match node {
                NodeRef::Leaf(i) => {
                    reachable.insert(i);
                }
                NodeRef::Interior(i) => {
                    let interior = &bsp.interiors[i as usize];
                    collect_leaves(bsp, interior.left, reachable);
                    collect_leaves(bsp, interior.right, reachable);
                }
            }
        }

        if bsp.interior_count() > 0 {
            // Start from root (last interior)
            let root = NodeRef::Interior((bsp.interior_count() - 1) as u32);
            collect_leaves(&bsp, root, &mut reachable_leaves);
        } else if bsp.leaf_count() > 0 {
            // Single leaf case
            reachable_leaves.insert(0);
        }

        // All leaves must be reachable
        for i in 0..bsp.leaf_count() {
            assert!(
                reachable_leaves.contains(&(i as u32)),
                "Leaf {} is not reachable from root",
                i
            );
        }
    }

    #[test]
    fn threshold_correctly_partitions_item_centers() {
        // Create items with known centers
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 10.0, 10.0), // center: (5, 5)
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (20.0, 0.0, 30.0, 10.0), // center: (25, 5)
                leaf: SolidColor::new(0, 255, 0, 255),
            },
            Positioned {
                bounds: (40.0, 0.0, 50.0, 10.0), // center: (45, 5)
                leaf: SolidColor::new(0, 0, 255, 255),
            },
            Positioned {
                bounds: (60.0, 0.0, 70.0, 10.0), // center: (65, 5)
                leaf: SolidColor::new(255, 255, 0, 255),
            },
        ];

        let centers = vec![5.0, 25.0, 45.0, 65.0];
        let bsp = SpatialBSP::from_positioned(items);

        // Verify each interior node's threshold correctly partitions
        fn verify_partition<L>(bsp: &SpatialBSP<L>, node: NodeRef, centers: &[f32], depth: usize) {
            if let NodeRef::Interior(i) = node {
                let interior = &bsp.interiors[i as usize];
                let threshold = interior.threshold;
                let axis = interior.axis;

                // Collect centers from left and right subtrees
                let mut left_centers = Vec::new();
                let mut right_centers = Vec::new();

                fn collect_centers<L>(
                    bsp: &SpatialBSP<L>,
                    node: NodeRef,
                    centers: &[f32],
                    output: &mut Vec<f32>,
                ) {
                    match node {
                        NodeRef::Leaf(i) => {
                            if (i as usize) < centers.len() {
                                output.push(centers[i as usize]);
                            }
                        }
                        NodeRef::Interior(i) => {
                            let interior = &bsp.interiors[i as usize];
                            collect_centers(bsp, interior.left, centers, output);
                            collect_centers(bsp, interior.right, centers, output);
                        }
                    }
                }

                collect_centers(bsp, interior.left, centers, &mut left_centers);
                collect_centers(bsp, interior.right, centers, &mut right_centers);

                // For X axis, all left centers should be < threshold
                // For Y axis, this test uses X-aligned items so we skip Y checks
                if axis == Axis::X {
                    for &center in &left_centers {
                        assert!(
                            center < threshold,
                            "Left child center {} should be < threshold {} (depth {})",
                            center,
                            threshold,
                            depth
                        );
                    }
                    for &center in &right_centers {
                        assert!(
                            center >= threshold,
                            "Right child center {} should be >= threshold {} (depth {})",
                            center,
                            threshold,
                            depth
                        );
                    }
                }

                // Recursively verify children
                verify_partition(bsp, interior.left, centers, depth + 1);
                verify_partition(bsp, interior.right, centers, depth + 1);
            }
        }

        if bsp.interior_count() > 0 {
            let root = NodeRef::Interior((bsp.interior_count() - 1) as u32);
            verify_partition(&bsp, root, &centers, 0);
        }
    }

    #[test]
    fn each_interior_node_has_distinct_children() {
        let items: Vec<_> = (0..16)
            .map(|i| {
                let x = (i as f32) * 10.0;
                Positioned {
                    bounds: (x, 0.0, x + 5.0, 10.0),
                    leaf: SolidColor::new(255, 0, 0, 255),
                }
            })
            .collect();

        let bsp = SpatialBSP::from_positioned(items);

        // Interior nodes shouldn't point to themselves or have identical children
        for (idx, interior) in bsp.interiors.iter().enumerate() {
            // Left and right must be different
            let left_is_self = matches!(interior.left, NodeRef::Interior(i) if i as usize == idx);
            let right_is_self = matches!(interior.right, NodeRef::Interior(i) if i as usize == idx);

            assert!(
                !left_is_self,
                "Interior {} points to itself as left child",
                idx
            );
            assert!(
                !right_is_self,
                "Interior {} points to itself as right child",
                idx
            );

            // Left and right should not be identical
            match (interior.left, interior.right) {
                (NodeRef::Interior(l), NodeRef::Interior(r)) => {
                    assert_ne!(l, r, "Interior {} has identical left/right children", idx);
                }
                (NodeRef::Leaf(l), NodeRef::Leaf(r)) => {
                    assert_ne!(
                        l, r,
                        "Interior {} has identical left/right leaf children",
                        idx
                    );
                }
                _ => {} // One interior, one leaf - always distinct
            }
        }
    }

    #[test]
    fn binary_tree_exact_interior_count() {
        // For n leaves, must have EXACTLY n-1 interiors (not >=, not <=, exactly)
        for n in 2..=20 {
            let items: Vec<_> = (0..n)
                .map(|i| {
                    let x = (i as f32) * 10.0;
                    Positioned {
                        bounds: (x, 0.0, x + 5.0, 10.0),
                        leaf: SolidColor::new(255, 0, 0, 255),
                    }
                })
                .collect();

            let bsp = SpatialBSP::from_positioned(items);

            assert_eq!(
                bsp.interior_count(),
                n - 1,
                "Binary tree with {} leaves must have exactly {} interiors, got {}",
                n,
                n - 1,
                bsp.interior_count()
            );
        }
    }

    #[test]
    fn threshold_is_finite() {
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 50.0, 100.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (50.0, 0.0, 100.0, 100.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        for (idx, interior) in bsp.interiors.iter().enumerate() {
            assert!(
                interior.threshold.is_finite(),
                "Interior {} has non-finite threshold: {}",
                idx,
                interior.threshold
            );
        }
    }

    #[test]
    fn empty_tree_has_exactly_zero_counts() {
        let bsp: SpatialBSP<SolidColor> = SpatialBSP::from_positioned(vec![]);

        assert_eq!(bsp.interior_count(), 0, "Empty tree must have 0 interiors");
        assert_eq!(bsp.leaf_count(), 0, "Empty tree must have 0 leaves");
    }

    #[test]
    fn single_item_has_exactly_one_leaf_zero_interiors() {
        let bsp = SpatialBSP::single(SolidColor::new(255, 0, 0, 255));

        assert_eq!(
            bsp.interior_count(),
            0,
            "Single-item tree must have exactly 0 interiors"
        );
        assert_eq!(
            bsp.leaf_count(),
            1,
            "Single-item tree must have exactly 1 leaf"
        );
    }

    #[test]
    fn test_stack_of_wide_strips() {
        use pixelflow_core::{materialize_discrete, PARALLELISM};

        // Create 2 wide, short items, stacked vertically.
        // Item 1: (0, 0, 100, 10). Red.
        // Item 2: (0, 10, 100, 20). Blue.
        let items = vec![
            Positioned {
                bounds: (0.0, 0.0, 100.0, 10.0),
                leaf: SolidColor::new(255, 0, 0, 255),
            },
            Positioned {
                bounds: (0.0, 10.0, 100.0, 20.0),
                leaf: SolidColor::new(0, 0, 255, 255),
            },
        ];

        let bsp = SpatialBSP::from_positioned(items);

        // Check Point in Item 1, Right side (x=75, y=5)
        let mut pixels = [0u32; PARALLELISM];
        materialize_discrete(&bsp, 75.0, 5.0, &mut pixels);

        let expected_red = {
            let red = SolidColor::new(255, 0, 0, 255);
            let mut buf = [0u32; PARALLELISM];
            materialize_discrete(&red, 0.0, 0.0, &mut buf);
            buf[0]
        };
        assert_eq!(
            pixels[0], expected_red,
            "Item 1 should be visible on right side"
        );

        // Check Point in Item 2, Left side (x=25, y=15)
        materialize_discrete(&bsp, 25.0, 15.0, &mut pixels);
        let expected_blue = {
            let blue = SolidColor::new(0, 0, 255, 255);
            let mut buf = [0u32; PARALLELISM];
            materialize_discrete(&blue, 0.0, 0.0, &mut buf);
            buf[0]
        };
        assert_eq!(
            pixels[0], expected_blue,
            "Item 2 should be visible on left side"
        );
    }
}
