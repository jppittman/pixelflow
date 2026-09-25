//! # A 3-D scene is four channel kernels of the screen coordinate
//!
//! Every geometry here is **analytic**: a ray's `t` is a closed form — the
//! sphere's quadratic, the plane's division — so a scene is one expression,
//! not a loop. There is no ray march and nothing is unrolled; if a scene ever
//! needs one, it is `n` steps in an ordinary Rust `for` at construction time,
//! and still a DAG when it gets here.
//!
//! The three layers the old jet tier named are still here, but they are
//! functions rather than types:
//!
//! 1. **Ray** — [`Ray::through_screen`] turns the pixel coordinate into a unit
//!    direction. The observer is fixed at the origin (CLAUDE.md's Fixed
//!    Observer), so a ray *is* its direction and a reflected ray is another
//!    direction from the same origin: the world seen in a mirror is the world
//!    seen along `R`, which is what the jet tier computed too.
//! 2. **Geometry** — [`Sphere::hit`] / [`Plane::hit`] solve for `t` and return
//!    a [`Hit`]: where the ray meets the surface, the outward normal there,
//!    and the mask saying whether it did at all.
//! 3. **Material** — [`checker`], [`sky`], [`Rgba::opaque_gray`]: four channel
//!    kernels in `[0, 1]`, which is all a colour ever is. [`Hit::select`]
//!    chooses material or background as ONE node over the whole colour, and
//!    nesting those choices is occlusion, exactly as before.
//!
//! ## Antialiasing is the calculus, not a heuristic
//!
//! A pixel's footprint on a surface is `∂P/∂X` and `∂P/∂Y`, and
//! [`Kernel::dx`]/[`Kernel::dy`] differentiate the whole expression
//! symbolically — through the screen mapping, the intersection, and the
//! reflection — so [`Hit::footprint`] is exact wherever the surface is
//! differentiable. The jet tier could not do that through a reflection (dual
//! numbers there were not trusted) and multiplied the normal's derivatives by
//! a hand-tuned `2/|cos θ|` instead; nothing here needs that, and the `dz`
//! term it carried drops out because a screen has two axes.
//!
//! ## Sharing
//!
//! `Kernel` has no let-binding: every combinator splices its operands, so
//! naming a subexpression twice writes it twice. Where that matters — a
//! material sampled at a hit point that is itself a large expression — the
//! piece is built over the coordinate slots and precomposed once with
//! [`Kernel::at`], which splices each argument exactly once and rewires every
//! use of it. What survives to the compiler as four *separate* channels is
//! shared back by the e-graph, which hash-conses the four copies of the
//! geometry into one — the four leaves of one [`Rgba`], never four copies of a
//! choice, which is the distinction the tree exists to keep.

use pixelflow_core::Kernel;

/// Past this the intersection is numerically meaningless (a ray parallel to
/// the plane gives `t = ±inf`), so it counts as a miss — the jet tier's
/// `t_max`, kept.
const MAX_T: f32 = 1.0e6;

/// Added under the sphere's discriminant so a grazing ray still has a hit
/// point to differentiate: the jet tier's epsilon, kept, because it is what
/// the silhouette's shape has always been.
const GRAZING_EPSILON: f32 = 1.0e-4;

/// Guards the division by a pixel footprint of zero — a surface seen exactly
/// edge-on, or a constant one.
const MIN_FOOTPRINT: f32 = 1.0e-3;

fn k(v: f32) -> Kernel {
    Kernel::constant(v)
}

/// `a · b`.
fn dot(a: &[Kernel; 3], b: &[Kernel; 3]) -> Kernel {
    a[0].mul(&b[0]).add(&a[1].mul(&b[1])).add(&a[2].mul(&b[2]))
}

/// `v · s`, componentwise.
fn scale(v: &[Kernel; 3], s: &Kernel) -> [Kernel; 3] {
    [v[0].mul(s), v[1].mul(s), v[2].mul(s)]
}

/// A colour: four channel kernels in `[0, 1]` — red, green, blue, alpha — or
/// a **choice** between two colours under a mask.
///
/// A tree, not an array, and that is the whole of stage S3b's language half.
/// A choice between colours is one thing that happens once, so it is one node
/// here and [`crate::render::scene::compile_packed_for`] lowers it to one
/// `If` on the packed words. Distributing it over the channels — four
/// `If`s sharing a mask, which is what an array of channels forces — is the
/// same picture and a different program: the emitter may skip a value under a
/// `If`'s arm only when *every* consumer of it lies inside that arm, and a
/// reflected world feeding four arms is, from each one's view, shared. It then
/// runs in every batch, for the 97% of the frame the sphere does not cover.
///
/// This is the same object `compile_packed_for` takes, so a scene is finished
/// when it is an `Rgba`: the pack to bytes is the compiler's, not the scene's.
#[derive(Clone)]
pub struct Rgba(Node);

/// The two ways a colour is written. Private: what a colour *is* is the four
/// channels at its leaves, and the only thing anyone outside may do with the
/// shape is [`Rgba::fold`] it.
#[derive(Clone)]
enum Node {
    /// Red, green, blue, alpha, each in `[0, 1]`.
    Channels([Kernel; 4]),
    /// `mask ? if_true : if_false` — one choice, whole colours.
    Choice {
        mask: Kernel,
        if_true: Box<Rgba>,
        if_false: Box<Rgba>,
    },
}

impl Rgba {
    /// A colour from its four channels.
    #[must_use]
    pub fn new(r: Kernel, g: Kernel, b: Kernel, a: Kernel) -> Self {
        Self(Node::Channels([r, g, b, a]))
    }

    /// An opaque flat grey — the matte material.
    #[must_use]
    pub fn opaque_gray(level: f32) -> Self {
        Self::new(k(level), k(level), k(level), k(1.0))
    }

    /// Lower this colour, leaves first: `pack` turns four channels into one
    /// value of the caller's choosing, `choose` blends two such values under a
    /// mask.
    ///
    /// The catamorphism is the whole interface to the tree's shape, and the
    /// only one — which is how "a choice is one `If`" stays a property of
    /// the type rather than of every caller's discipline. There is no way to
    /// reach the arms of a choice and pack them separately, so the packer
    /// ([`crate::render::scene::compile_packed_for`]) cannot accidentally be
    /// written as four `If`s again, and a caller that only wants to count
    /// nodes or size the code gets the same one traversal.
    pub fn fold<T>(
        &self,
        pack: &dyn Fn(&[Kernel; 4]) -> T,
        choose: &dyn Fn(&Kernel, T, T) -> T,
    ) -> T {
        match &self.0 {
            Node::Channels(channels) => pack(channels),
            Node::Choice {
                mask,
                if_true,
                if_false,
            } => choose(
                mask,
                if_true.fold(pack, choose),
                if_false.fold(pack, choose),
            ),
        }
    }

    /// `mask ? self : other` — one node, whatever the two colours are.
    ///
    /// `pub(crate)` because [`Hit::select`] is how a scene introduces a
    /// choice: a choice between colours *is* a surface being there or not.
    pub(crate) fn select(&self, mask: &Kernel, other: &Self) -> Self {
        Self(Node::Choice {
            mask: mask.clone(),
            if_true: Box::new(self.clone()),
            if_false: Box::new(other.clone()),
        })
    }

    /// The same colour with `f` applied to every leaf's four channels — a
    /// tint, or a scene rebuilt with one channel live. The choices and their
    /// masks are untouched, so the geometry is the same expression it was.
    #[must_use]
    pub fn map_channels(&self, f: &dyn Fn(&[Kernel; 4]) -> [Kernel; 4]) -> Self {
        match &self.0 {
            Node::Channels(channels) => Self(Node::Channels(f(channels))),
            Node::Choice {
                mask,
                if_true,
                if_false,
            } => Self(Node::Choice {
                mask: mask.clone(),
                if_true: Box::new(if_true.map_channels(f)),
                if_false: Box::new(if_false.map_channels(f)),
            }),
        }
    }
}

/// Four channel kernels are a colour with no choice in it — the leaf every
/// procedural shader already writes.
impl From<[Kernel; 4]> for Rgba {
    fn from(channels: [Kernel; 4]) -> Self {
        Self(Node::Channels(channels))
    }
}

impl From<&[Kernel; 4]> for Rgba {
    fn from(channels: &[Kernel; 4]) -> Self {
        Self(Node::Channels(channels.clone()))
    }
}

/// A ray from the fixed observer at the origin: a direction, as three kernels
/// of the screen coordinate (and of `W`, when the scene animates).
///
/// A reflected ray is another direction from that same origin, which makes a
/// mirror show the world along `R` — an environment reflection. That is the
/// denotation the jet tier had, and moving the origin to the hit point would
/// be a different (and more expensive) scene, not a bug fix here.
#[derive(Clone)]
pub struct Ray {
    dir: [Kernel; 3],
}

impl Ray {
    /// The pinhole camera's rays over a `width × height` pixel frame: the
    /// screen is `[-aspect, aspect] × [-1, 1]` at unit focal length, y up.
    ///
    /// This is the whole camera. Everything downstream is a function of the
    /// direction, so the frame's size is spent here and nowhere else.
    #[must_use]
    pub fn through_screen(width: f32, height: f32) -> Self {
        let scale = 2.0 / height;
        let sx = Kernel::x().sub(&k(width * 0.5)).mul(&k(scale));
        let sy = k(height * 0.5).sub(&Kernel::y()).mul(&k(scale));
        let sz = k(1.0);
        let len = sx.mul(&sx).add(&sy.mul(&sy)).add(&sz.mul(&sz)).sqrt();
        Self {
            dir: [sx.div(&len), sy.div(&len), sz.div(&len)],
        }
    }

    /// A ray in an arbitrary direction — normalized here, so callers hand over
    /// the direction they mean rather than remembering to scale it.
    #[must_use]
    pub fn towards(dir: [Kernel; 3]) -> Self {
        let len = dot(&dir, &dir).sqrt();
        Self {
            dir: [dir[0].div(&len), dir[1].div(&len), dir[2].div(&len)],
        }
    }

    /// The unit direction.
    #[must_use]
    pub fn direction(&self) -> &[Kernel; 3] {
        &self.dir
    }

    /// The Householder reflection `R = D − 2(D·N)N` about a unit normal.
    #[must_use]
    pub fn reflected(&self, normal: &[Kernel; 3]) -> Self {
        let two_d_dot_n = dot(&self.dir, normal).mul(&k(2.0));
        let scaled = scale(normal, &two_d_dot_n);
        Self {
            dir: core::array::from_fn(|i| self.dir[i].sub(&scaled[i])),
        }
    }
}

/// Where a ray meets a surface: the mask that says it did, the hit point, and
/// the outward unit normal there.
///
/// The mask is the geometry's own predicate — a sphere's discriminant, a
/// plane's finite positive `t` — and never a test on `t` alone: `t` is `NaN`
/// for a miss under a square root, and `NaN > 0` is *true* on x86 (CLAUDE.md's
/// unordered `Gt`), so a mask read off it would light up the whole frame on
/// one target and nothing on the other.
#[derive(Clone)]
pub struct Hit {
    mask: Kernel,
    point: [Kernel; 3],
    normal: [Kernel; 3],
}

impl Hit {
    /// True where the ray meets the surface.
    #[must_use]
    pub fn mask(&self) -> &Kernel {
        &self.mask
    }

    /// The hit point `P = D·t`.
    #[must_use]
    pub fn point(&self) -> &[Kernel; 3] {
        &self.point
    }

    /// The outward unit normal at the hit point.
    #[must_use]
    pub fn normal(&self) -> &[Kernel; 3] {
        &self.normal
    }

    /// How far the hit point moves for a one-pixel step — the width a
    /// material should filter its edges over.
    ///
    /// `max_i ‖(∂P_i/∂X, ∂P_i/∂Y)‖`, differentiated symbolically through
    /// everything between the pixel and the surface. A component that does not
    /// move (a floor's height) contributes zero and drops out of the maximum.
    #[must_use]
    pub fn footprint(&self) -> Kernel {
        let axis = |p: &Kernel| p.dx().hypot(&p.dy());
        axis(&self.point[0])
            .max(&axis(&self.point[1]))
            .max(&axis(&self.point[2]))
    }

    /// `material` where this hit happened, `background` where it did not —
    /// **one** choice over the whole colour, so a material's geometry is
    /// exclusive to its arm and the emitter can skip it where the mask says
    /// nothing in the batch needs it. Nesting these is occlusion.
    #[must_use]
    pub fn select(&self, material: &Rgba, background: &Rgba) -> Rgba {
        material.select(&self.mask, background)
    }
}

/// A sphere. Its centre and radius are kernels, so a scene that moves is the
/// same compiled program on a later `W` rather than a recompile.
pub struct Sphere {
    center: [Kernel; 3],
    radius: Kernel,
}

impl Sphere {
    /// A sphere at `center` with radius `radius`.
    #[must_use]
    pub fn new(center: [Kernel; 3], radius: Kernel) -> Self {
        Self { center, radius }
    }

    /// The near intersection of `ray` with this sphere.
    ///
    /// `|tD − C|² = r²` solved for the smaller root: `t = D·C − √((D·C)² −
    /// (|C|² − r²))`, with the observer at the origin. The mask is the
    /// discriminant's sign — the silhouette *is* `disc > 0` — and the root is
    /// taken of a clamped discriminant so the miss branch carries a number
    /// rather than a `NaN` into the blend.
    #[must_use]
    pub fn hit(&self, ray: &Ray) -> Hit {
        let d = ray.direction();
        let d_dot_c = dot(d, &self.center);
        let r_sq = self.radius.mul(&self.radius);
        let disc = d_dot_c
            .mul(&d_dot_c)
            .sub(&dot(&self.center, &self.center).sub(&r_sq))
            .add(&k(GRAZING_EPSILON));
        let t = d_dot_c.sub(&disc.max(&k(0.0)).sqrt());
        let point = scale(d, &t);
        Hit {
            mask: disc.gt(&k(0.0)).and(&t.gt(&k(0.0))).and(&t.lt(&k(MAX_T))),
            normal: core::array::from_fn(|i| point[i].sub(&self.center[i]).div(&self.radius)),
            point,
        }
    }
}

/// A horizontal plane at `y = height`.
pub struct Plane {
    height: Kernel,
}

impl Plane {
    /// The plane `y = height`.
    #[must_use]
    pub fn at_height(height: Kernel) -> Self {
        Self { height }
    }

    /// Where `ray` crosses this plane: `t = height / D.y`.
    ///
    /// A ray parallel to the plane gives `±inf`, which the finite-`t` half of
    /// the mask rejects; one pointing away gives a negative `t`.
    #[must_use]
    pub fn hit(&self, ray: &Ray) -> Hit {
        let d = ray.direction();
        let t = self.height.div(&d[1]);
        Hit {
            mask: t.gt(&k(0.0)).and(&t.lt(&k(MAX_T))),
            point: scale(d, &t),
            normal: [k(0.0), k(1.0), k(0.0)],
        }
    }
}

/// The sky: a blue gradient in the ray's elevation. A background, so it has no
/// hit point and no footprint.
#[must_use]
pub fn sky(ray: &Ray) -> Rgba {
    let t = ray.direction()[1]
        .mul(&k(0.5))
        .add(&k(0.5))
        .clamp(&k(0.0), &k(1.0));
    Rgba::new(
        k(0.7).sub(&t.mul(&k(0.5))),
        k(0.85).sub(&t.mul(&k(0.45))),
        k(1.0).sub(&t.mul(&k(0.2))),
        k(1.0),
    )
}

/// The warm half of the checker.
const CHECKER_LIGHT: [f32; 3] = [0.95, 0.9, 0.8];
/// The cool half.
const CHECKER_DARK: [f32; 3] = [0.2, 0.25, 0.3];

/// A unit checkerboard in the `x`/`z` plane, its edges filtered over
/// `footprint` — normally [`Hit::footprint`], which is one pixel wide.
///
/// The two *geometric* facts about where the hit lands — which square it is
/// in, and how far it is from the nearest edge — are functions of the hit's
/// `x` and `z` alone, so they are built over the two coordinate slots and
/// substituted there. The footprint is not a coordinate of the material: it is
/// a screen-space quantity the caller already has, and it enters where it is
/// used rather than as a third axis nothing varies along. (It was one, and it
/// was the last reader of `Z` outside a shader's clock.)
///
/// **What that costs, measured rather than assumed.** Substituting here — into
/// `light` and `to_edge`, before each is read by all three colour channels —
/// puts the hit point in the arena **twelve** times where the four-slot form
/// put it four: 2,404 nodes per channel against 1,201, and 7,213 for the
/// colour against 4,618. It is not a value bug (saturation absorbs it; the
/// `chrome` byte-identity table in the L1 CL is the evidence) but it doubles
/// the *input* to saturation for every scene calling this, and budgets here
/// are hard — `safety_ceiling` panics the build rather than truncating.
///
/// The four-slot form substituted once, last, over the finished blend, because
/// a four-coordinate `at` could carry the footprint through untouched as a
/// third slot. With two axes there is no way to say "substitute `x` and `z`,
/// leave `footprint` alone" in one operation: substitution rewrites every
/// `Var(0)`/`Var(1)`, and the footprint arrives already in the caller's space.
/// Expressing it would need `pixelflow-ir`'s placeholder machinery, which is
/// private to that crate and is not worth widening for this. So the cost is
/// recorded here rather than removed.
#[must_use]
pub fn checker(x: &Kernel, z: &Kernel, footprint: &Kernel) -> Rgba {
    let (cx, cz) = (Kernel::x(), Kernel::y());

    let cell_x = cx.floor();
    let cell_z = cz.floor();
    let half = cell_x.add(&cell_z).mul(&k(0.5));
    let light = half.sub(&half.floor()).abs().lt(&k(0.25));

    // How much of the pixel's footprint lands in the cell its centre is in:
    // a box filter of width `f` centred `d` from the nearest edge covers
    // `½ + d/f` of this cell, up to all of it. The two limits are the reason
    // for the `½`: a footprint much smaller than a cell gives 1 (a hard
    // edge), and one much larger gives ½ — the average of the two colours,
    // which is what a checkerboard washes out to when you cannot resolve it.
    // Without the `½` the same expression sends a pixel *on* an edge to the
    // neighbour's colour outright, so cells swap across every boundary and a
    // surface seen at a grazing angle flickers between whole cells.
    let edge = |f: &Kernel| k(0.5).sub(&f.sub(&k(0.5)).abs());
    let to_edge = edge(&cx.sub(&cell_x)).min(&edge(&cz.sub(&cell_z)));

    // Both facts, sampled at the hit point. Everything below is in the
    // caller's space already, so nothing else is substituted.
    let light = light.at(x, z);
    let to_edge = to_edge.at(x, z);
    let coverage = k(0.5).add(&to_edge.div(&footprint.add(&k(MIN_FOOTPRINT))).min(&k(0.5)));

    let blend = |c: usize| {
        let here = light.select(&k(CHECKER_LIGHT[c]), &k(CHECKER_DARK[c]));
        let over = light.select(&k(CHECKER_DARK[c]), &k(CHECKER_LIGHT[c]));
        here.mul(&coverage).add(&over.mul(&k(1.0).sub(&coverage)))
    };
    Rgba::new(blend(0), blend(1), blend(2), k(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use pixelflow_core::Lattice;

    /// The four channels of a colour that has no choice in it. `fold` is the
    /// only way into the tree, and a leaf's fold is exactly its channels.
    fn leaf_channels(color: &Rgba) -> [Kernel; 4] {
        color.fold(&|channels| channels.clone(), &|_, _, _| {
            panic!("this colour is a choice, not a leaf")
        })
    }

    /// Tabulate a kernel over a `w × h` integer grid. A mask has to be read
    /// through a `select` — it is a bit pattern, not a number.
    fn bake(kernel: &Kernel, w: usize, h: usize) -> Vec<f32> {
        Lattice::frame(w, h).bake(kernel).buffer().to_vec()
    }

    /// The camera's rays are unit length.
    ///
    /// To 1e-3, not to the last bit: the optimizer is free to answer `x/√y`
    /// with `x·rsqrt(y)`, and `rsqrt` is a ~12-bit estimate (CLAUDE.md's
    /// "precision is on the table"). What is *not* on the table is the range,
    /// and a direction that is not unit length would put the sphere's `t` on a
    /// different scale entirely.
    #[test]
    fn screen_rays_are_unit_length() {
        let ray = Ray::through_screen(64.0, 32.0);
        let len = dot(ray.direction(), ray.direction()).sqrt();
        for (i, v) in bake(&len, 64, 32).iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-3, "sample {i} has |D| = {v}");
        }
    }

    /// The near root is the near surface, and a sphere behind the observer is
    /// a miss — the mask is the discriminant's sign, so it does not depend on
    /// how a target orders a `NaN` comparison.
    #[test]
    fn a_sphere_is_hit_where_its_discriminant_is_positive() {
        let forward = Ray::towards([k(0.0), k(0.0), k(1.0)]);
        let hit = Sphere::new([k(0.0), k(0.0), k(4.0)], k(1.0)).hit(&forward);
        let z = hit.mask().select(&hit.point()[2], &k(-1.0));
        let z = bake(&z, 1, 1)[0];
        assert!(
            (2.99..3.01).contains(&z),
            "the near hit of a unit sphere at z = 4 is z = 3, got {z}"
        );

        let behind = Sphere::new([k(0.0), k(0.0), k(-4.0)], k(1.0)).hit(&forward);
        assert_eq!(bake(&behind.mask().select(&k(1.0), &k(0.0)), 1, 1)[0], 0.0);

        // Off-axis by more than the sphere subtends: also a miss.
        let away = Ray::towards([k(1.0), k(0.0), k(1.0)]);
        let side = Sphere::new([k(0.0), k(0.0), k(4.0)], k(1.0)).hit(&away);
        assert_eq!(bake(&side.mask().select(&k(1.0), &k(0.0)), 1, 1)[0], 0.0);
    }

    /// A mirror facing the observer sends the ray straight back.
    #[test]
    fn reflection_is_the_householder_form() {
        let ray = Ray::towards([k(0.0), k(0.0), k(1.0)]);
        let back = ray.reflected(&[k(0.0), k(0.0), k(-1.0)]);
        let z = bake(&back.direction()[2], 1, 1)[0];
        assert!((z + 1.0).abs() < 1e-6, "got {z}");
    }

    /// A reflection off a *sphere* is still a reflection: the reflected ray
    /// is unit length wherever the sphere is hit.
    ///
    /// This is the property the jet tier lost. Its normal came from the
    /// tangent frame's cross product and was scaled by
    /// `n_len_sq.sqrt().rsqrt()` — which is `|n|^-½`, not `|n|^-1` — so its
    /// "unit normal" had length `√|n|` and `D − 2(D·N)N` was a reflection
    /// only where the screen-to-surface map happened to have unit area
    /// scale. That is very likely why the reflected ray's derivatives needed
    /// a hand-tuned curvature factor to look right.
    #[test]
    fn a_reflection_off_a_sphere_is_a_unit_ray() {
        let ray = Ray::through_screen(64.0, 64.0);
        let hit = Sphere::new([k(0.0), k(0.0), k(4.0)], k(1.0)).hit(&ray);
        let mirrored = ray.reflected(hit.normal());
        let len_sq = dot(mirrored.direction(), mirrored.direction());
        // Off the sphere the normal is meaningless, so ask only where it is
        // not: a miss reports 1 so the sweep below is uniform.
        let guarded = hit.mask().select(&len_sq, &k(1.0));
        // The bound is loose (2.5%) and the reason is the silhouette: `t = b
        // − √(b² − c)` loses its leading digits where `b ≈ √(b² − c)`, which
        // is exactly the grazing rim, and the hit point's error carries into
        // the normal. Away from the rim it is the `rsqrt` estimates' ~1e-3.
        // What it forbids is the other kind of error — a normal of length
        // `√|n|`, which is off by whole factors and would fail this
        // everywhere at once.
        let worst = bake(&guarded, 64, 64)
            .into_iter()
            .fold(0.0f32, |w, v| w.max((v - 1.0).abs()));
        assert!(worst < 2.5e-2, "the reflected ray has |R|² off by {worst}");
    }

    /// The checker's two limits, which are what its filter *is*: resolved
    /// finer than a cell it is the cell's own colour, and coarser than one it
    /// is the average of the two — never the neighbour's colour, which is
    /// what a coverage without the `½` gives and what made a distant floor
    /// flicker between whole cells.
    #[test]
    fn a_checker_finer_than_its_footprint_washes_out_to_grey() {
        let cell_centre = |footprint: f32| {
            let rgba = checker(&Kernel::x(), &Kernel::y(), &k(footprint));
            let at = |c: &Kernel| Lattice::eval_at(c, 0.5, 0.5);
            let channels = leaf_channels(&rgba);
            [at(&channels[0]), at(&channels[1]), at(&channels[2])]
        };

        let sharp = cell_centre(1e-4);
        assert_eq!(
            sharp, CHECKER_LIGHT,
            "a cell far larger than the footprint is its own colour"
        );

        let washed = cell_centre(1e4);
        for c in 0..3 {
            let mean = 0.5 * (CHECKER_LIGHT[c] + CHECKER_DARK[c]);
            assert!(
                (washed[c] - mean).abs() < 1e-3,
                "channel {c} washes out to {mean}, got {}",
                washed[c]
            );
        }
    }

    /// The floor's screen footprint is what a material filters over: a
    /// pixel's worth of the surface, growing as the surface recedes. (The jet
    /// tier's footprint was in units of the *normalized* screen — half the
    /// frame height — which is why its distant floor showed whole flipped
    /// cells where its near floor was smeared over a hundred pixels.)
    #[test]
    fn the_floor_footprint_is_one_pixel_wide() {
        let ray = Ray::through_screen(64.0, 64.0);
        let floor = Plane::at_height(k(-1.0)).hit(&ray);
        let foot = bake(&floor.mask().select(&floor.footprint(), &k(0.0)), 64, 64);
        // Row 63 is the bottom of the frame (close), row 34 is just below the
        // horizon (far).
        let near = foot[63 * 64 + 32];
        let far = foot[34 * 64 + 32];
        assert!(
            near > 0.0 && near < 0.2,
            "a near floor pixel covers a fraction of a cell, got {near}"
        );
        assert!(
            far > near,
            "a distant floor pixel covers more, got {far} against {near}"
        );
    }
}
