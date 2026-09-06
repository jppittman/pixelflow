//! `F32x8`/`Mask8`/`U32x8` (AVX2) coverage.
//!
//! Mirrors `x86_backend_tests.rs`'s SSE2 coverage at 8 lanes. The whole
//! module is gated on `target_feature = "avx2"` because the types it
//! imports only exist under that cfg in `pixelflow_core::backend::x86` — a
//! plain `cargo test` (SSE2 baseline, no `-C target-feature`) compiles this
//! file down to nothing. Run with
//! `RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p pixelflow-core`
//! (or `cargo xtask isa-matrix`) on an AVX2+FMA host to actually execute it.
#[cfg(target_arch = "x86_64")]
#[cfg(target_feature = "avx2")]
#[cfg(test)]
mod tests {
    extern crate std;
    use pixelflow_core::backend::x86::{F32x8, Mask8, U32x8};
    use pixelflow_core::backend::{MaskOps, SimdOps, SimdU32Ops};
    use std::prelude::v1::*;

    #[test]
    fn add_sub_mul_and_div_should_operate_lanewise_on_all_eight_lanes() {
        let a = F32x8::splat(2.0);
        let b = F32x8::splat(3.0);

        let sum = a + b;
        let mut out = [0.0; 8];
        sum.store(&mut out);
        assert_eq!(out, [5.0; 8]);

        let diff = b - a;
        diff.store(&mut out);
        assert_eq!(out, [1.0; 8]);

        let prod = a * b;
        prod.store(&mut out);
        assert_eq!(out, [6.0; 8]);

        let quot = b / a;
        quot.store(&mut out);
        assert_eq!(out, [1.5; 8]);
    }

    #[test]
    fn sequential_should_produce_eight_consecutive_values_from_the_given_start() {
        let seq = F32x8::sequential(10.0);
        let mut out = [0.0; 8];
        seq.store(&mut out);
        assert_eq!(out, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]);
    }

    #[test]
    fn cmp_lt_and_simd_select_should_choose_lanes_by_the_comparison_result() {
        let a = F32x8::splat(1.0);
        let b = F32x8::splat(2.0);

        let lt = a.cmp_lt(b);
        assert!(lt.all());

        let t = F32x8::splat(10.0);
        let f = F32x8::splat(20.0);
        let sel = F32x8::simd_select(lt, t, f);
        let mut out = [0.0; 8];
        sel.store(&mut out);
        assert_eq!(out, [10.0; 8]);

        let gt = a.cmp_gt(b);
        assert!(!gt.any());
        let sel2 = F32x8::simd_select(gt, t, f);
        sel2.store(&mut out);
        assert_eq!(out, [20.0; 8]);
    }

    #[test]
    fn bitand_should_compute_the_lanewise_bitwise_and() {
        let a = F32x8::splat(1.0); // 1.0 is 0x3f800000
        let b = F32x8::splat(2.0); // 2.0 is 0x40000000
        let c = a & b;
        let mut out = [0.0; 8];
        c.store(&mut out);
        assert_eq!(out, [0.0; 8]);

        // The pair above ANDs to all-zero, which coincides with
        // `F32x8::default()` — a "replace bitand with Default::default()"
        // mutant survives it. 3.0 (0x40400000) & 2.0 (0x40000000) = 2.0,
        // distinguishing the real op from the default.
        let d = F32x8::splat(3.0) & F32x8::splat(2.0);
        d.store(&mut out);
        assert_eq!(out, [2.0; 8]);
    }

    #[test]
    fn simd_sqrt_simd_abs_and_simd_min_should_compute_correct_lane_results() {
        let a = F32x8::splat(4.0);
        let sqrt = a.simd_sqrt();
        let mut out = [0.0; 8];
        sqrt.store(&mut out);
        assert_eq!(out, [2.0; 8]);

        let b = F32x8::splat(-2.0);
        let abs = b.simd_abs();
        abs.store(&mut out);
        assert_eq!(out, [2.0; 8]);

        let min = a.simd_min(b);
        min.store(&mut out);
        assert_eq!(out, [-2.0; 8]);
    }

    #[test]
    fn any_and_all_should_report_whether_any_or_every_lane_is_true() {
        let zero = F32x8::splat(0.0);
        let zero_mask = zero.float_to_mask();
        assert!(!zero_mask.any());
        assert!(!zero_mask.all());

        let all_true = F32x8::splat(1.0).cmp_gt(F32x8::splat(0.0));
        assert!(all_true.any());
        assert!(all_true.all());

        // Lane 0 is false (0 > 0), lanes 1..8 are true.
        let mixed = F32x8::sequential(0.0).cmp_gt(F32x8::splat(0.0));
        assert!(mixed.any());
        assert!(!mixed.all());
    }

    #[test]
    #[should_panic]
    fn store_should_panic_when_the_output_slice_is_shorter_than_eight_lanes() {
        let a = F32x8::default();
        let mut out = [0.0; 7]; // Too small
        a.store(&mut out);
    }

    #[test]
    fn recip_and_simd_rsqrt_should_approximate_the_reciprocal_and_inverse_sqrt() {
        let a = F32x8::splat(4.0);
        let mut out = [0.0; 8];

        let recip = a.recip();
        recip.store(&mut out);
        for x in out.iter() {
            assert!(
                (x - 0.25).abs() < 1e-3,
                "recip(4.0) should be 0.25, got {}",
                x
            );
        }

        let rsqrt = a.simd_rsqrt();
        rsqrt.store(&mut out);
        for x in out.iter() {
            assert!(
                (x - 0.5).abs() < 1e-3,
                "rsqrt(4.0) should be 0.5, got {}",
                x
            );
        }
    }

    // ── SimdOps provided methods ──────────────────────────────────────────

    fn lanes(v: F32x8) -> [f32; 8] {
        let mut out = [0.0; 8];
        v.store(&mut out);
        out
    }

    #[test]
    fn simd_exp_matches_the_scalar_exponential() {
        let got = lanes(F32x8::splat(3.0).exp());
        let want = 3.0f32.exp();
        for x in got {
            assert!((x - want).abs() < 1e-2, "exp(3.0) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_ln_matches_the_scalar_natural_log() {
        let got = lanes(F32x8::splat(10.0).ln());
        let want = 10.0f32.ln();
        for x in got {
            assert!((x - want).abs() < 1e-2, "ln(10.0) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_log10_matches_the_scalar_base_10_log() {
        let got = lanes(F32x8::splat(100.0).log10());
        let want = 100.0f32.log10();
        for x in got {
            assert!((x - want).abs() < 1e-2, "log10(100.0) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_pow_matches_the_scalar_power_function() {
        let got = lanes(F32x8::splat(4.0).pow(F32x8::splat(0.5)));
        let want = 4.0f32.powf(0.5);
        for x in got {
            assert!((x - want).abs() < 1e-2, "pow(4.0, 0.5) = {x}, want {want}");
        }
    }

    #[test]
    fn simd_hypot_computes_the_euclidean_norm() {
        let got = lanes(F32x8::splat(3.0).hypot(F32x8::splat(4.0)));
        for x in got {
            assert!((x - 5.0).abs() < 1e-2, "hypot(3.0, 4.0) = {x}, want 5.0");
        }
    }

    #[test]
    fn simd_mul_rsqrt_divides_by_the_square_root() {
        let got = lanes(F32x8::splat(10.0).mul_rsqrt(F32x8::splat(4.0)));
        for x in got {
            assert!(
                (x - 5.0).abs() < 1e-2,
                "mul_rsqrt(10.0, 4.0) = {x}, want 5.0 (10.0 / sqrt(4.0))"
            );
        }
    }

    #[test]
    fn simd_ceil_rounds_toward_positive_infinity() {
        assert_eq!(lanes(F32x8::splat(1.2).ceil()), [2.0; 8]);
        assert_eq!(lanes(F32x8::splat(-1.2).ceil()), [-1.0; 8]);
        assert_eq!(lanes(F32x8::splat(3.0).ceil()), [3.0; 8]);
    }

    #[test]
    fn simd_round_rounds_to_the_nearest_integer() {
        assert_eq!(lanes(F32x8::splat(2.7).round()), [3.0; 8]);
        assert_eq!(lanes(F32x8::splat(2.2).round()), [2.0; 8]);
    }

    #[test]
    fn simd_fract_returns_the_value_past_the_decimal_point() {
        let got = lanes(F32x8::splat(2.75).fract());
        for x in got {
            assert!((x - 0.75).abs() < 1e-5, "fract(2.75) = {x}, want 0.75");
        }
    }

    #[test]
    fn simd_clamp_bounds_a_value_to_the_given_range() {
        assert_eq!(
            lanes(F32x8::splat(5.0).clamp(F32x8::splat(0.0), F32x8::splat(10.0))),
            [5.0; 8]
        );
        assert_eq!(
            lanes(F32x8::splat(-5.0).clamp(F32x8::splat(0.0), F32x8::splat(10.0))),
            [0.0; 8]
        );
        assert_eq!(
            lanes(F32x8::splat(15.0).clamp(F32x8::splat(0.0), F32x8::splat(10.0))),
            [10.0; 8]
        );
    }

    // ── Remaining SimdOps required methods, Debug impls, and U32x8 ─────────

    fn u32_lanes(v: U32x8) -> [u32; 8] {
        let mut out = [0u32; 8];
        v.store(&mut out);
        out
    }

    #[test]
    fn mask8_debug_format_shows_the_actual_lane_pattern() {
        // Lane 0 is false (0 > 0), lanes 1-7 are true — movemask reads
        // lane 7..0 MSB-first, so the mixed pattern prints as "11111110".
        let mixed = F32x8::sequential(0.0).cmp_gt(F32x8::splat(0.0));
        assert_eq!(format!("{:?}", mixed), "Mask8(11111110)");
    }

    #[test]
    fn f32x8_debug_format_shows_the_actual_lane_values() {
        assert_eq!(
            format!("{:?}", F32x8::splat(3.5)),
            "F32x8([3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5])"
        );
    }

    #[test]
    fn u32x8_debug_format_shows_the_actual_lane_values() {
        assert_eq!(
            format!("{:?}", <U32x8 as SimdU32Ops>::splat(42)),
            "U32x8([42, 42, 42, 42, 42, 42, 42, 42])"
        );
    }

    #[test]
    fn simd_gather_should_read_the_slice_element_at_each_lanes_own_index() {
        // A scrambled index vector (not the identity, not all-equal) is
        // needed to distinguish real per-lane gather from a mutant that
        // e.g. reads lane 0's index for every lane.
        let slice = [100.0f32, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0];
        let idx = F32x8::from_slice(&[3.0, 1.0, 4.0, 0.0, 2.0, 6.0, 5.0, 7.0]);
        let got = lanes(F32x8::gather(&slice, idx));
        assert_eq!(
            got,
            [400.0, 200.0, 500.0, 100.0, 300.0, 700.0, 600.0, 800.0]
        );
    }

    #[test]
    fn add_masked_should_add_only_in_the_lanes_where_the_mask_is_true() {
        let base = F32x8::splat(5.0);
        let val = F32x8::splat(10.0);

        let all_true = F32x8::splat(1.0).cmp_gt(F32x8::splat(0.0));
        assert_eq!(lanes(base.add_masked(val, all_true)), [15.0; 8]);

        let all_false = F32x8::splat(0.0).cmp_gt(F32x8::splat(1.0));
        assert_eq!(lanes(base.add_masked(val, all_false)), [5.0; 8]);

        // [0,1,2,3,4,5,6,7] > 1 is false,false,true,true,true,true,true,true
        let mixed = F32x8::sequential(0.0).cmp_gt(F32x8::splat(1.0));
        assert_eq!(
            lanes(base.add_masked(val, mixed)),
            [5.0, 5.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
        );
    }

    #[test]
    fn from_u32_bits_reinterprets_the_bit_pattern_as_f32() {
        let got = lanes(F32x8::from_u32_bits(1.0f32.to_bits()));
        assert_eq!(got, [1.0; 8]);
    }

    #[test]
    fn shr_u32_shifts_the_reinterpreted_bit_pattern_right() {
        let v = F32x8::from_u32_bits(128);
        let got = lanes(v.shr_u32(3));
        for x in got {
            assert_eq!(x.to_bits(), 16, "128 >> 3 should be 16");
        }
    }

    #[test]
    fn i32_to_f32_should_convert_the_reinterpreted_integer_not_the_bit_pattern() {
        let got = lanes(F32x8::from_u32_bits(5).i32_to_f32());
        assert_eq!(got, [5.0; 8]);
    }

    #[test]
    fn i32_to_f32_should_read_the_lane_as_signed_when_the_high_bit_is_set() {
        let got = lanes(F32x8::from_u32_bits(0xFFFF_FFFF).i32_to_f32());
        assert_eq!(got, [-1.0; 8]);
    }

    #[test]
    fn f32x8_bitor_combines_the_reinterpreted_bit_patterns() {
        let a = F32x8::from_u32_bits(0b1010);
        let b = F32x8::from_u32_bits(0b0101);
        let got = lanes(a | b);
        for x in got {
            assert_eq!(x.to_bits(), 0b1111);
        }
    }

    #[test]
    fn f32x8_not_flips_every_bit_of_the_pattern() {
        let got = lanes(!F32x8::from_u32_bits(0));
        for x in got {
            assert_eq!(x.to_bits(), u32::MAX);
        }
    }

    #[test]
    fn f32x8_neg_flips_the_sign_bit() {
        let got = lanes(-F32x8::splat(3.5));
        assert_eq!(got, [-3.5; 8]);
        let got_neg = lanes(-F32x8::splat(-3.5));
        assert_eq!(got_neg, [3.5; 8]);
    }

    #[test]
    fn u32x8_splat_then_store_round_trips_the_value() {
        assert_eq!(u32_lanes(<U32x8 as SimdU32Ops>::splat(7)), [7; 8]);
    }

    #[test]
    fn u32x8_bitand_masks_the_lanes() {
        let a = <U32x8 as SimdU32Ops>::splat(0b1010);
        let b = <U32x8 as SimdU32Ops>::splat(0b0110);
        assert_eq!(u32_lanes(a & b), [0b0010; 8]);
    }

    #[test]
    fn u32x8_bitor_combines_the_lanes() {
        let a = <U32x8 as SimdU32Ops>::splat(0b1010);
        let b = <U32x8 as SimdU32Ops>::splat(0b0101);
        assert_eq!(u32_lanes(a | b), [0b1111; 8]);
    }

    #[test]
    fn u32x8_not_flips_every_bit_of_every_lane() {
        assert_eq!(u32_lanes(!<U32x8 as SimdU32Ops>::splat(0)), [u32::MAX; 8]);
    }

    #[test]
    fn u32x8_shl_shifts_every_lane_left_by_the_given_count() {
        let v = <U32x8 as SimdU32Ops>::splat(1);
        assert_eq!(u32_lanes(v << 4), [16; 8]);
    }

    #[test]
    fn u32x8_shr_shifts_every_lane_right_by_the_given_count() {
        let v = <U32x8 as SimdU32Ops>::splat(16);
        assert_eq!(u32_lanes(v >> 4), [1; 8]);
    }

    #[test]
    fn pack_rgba_should_scale_in_range_channels_and_pack_them_one_u32_per_lane() {
        let r = F32x8::splat(1.0); // -> 255
        let g = F32x8::splat(0.5); // -> 127 (truncated, not rounded)
        let b = F32x8::splat(0.25); // -> 63
        let a = F32x8::splat(0.0); // -> 0

        let packed = u32_lanes(U32x8::pack_rgba(r, g, b, a));
        let expected: u32 = 255 | (127 << 8) | (63 << 16); // A channel is 0
        assert_eq!(packed, [expected; 8]);
    }

    #[test]
    fn pack_rgba_should_clamp_a_channel_outside_0_1_before_scaling_it() {
        let above = F32x8::splat(2.5); // clamps to 1.0 -> 255
        let below = F32x8::splat(-1.5); // clamps to 0.0 -> 0
        let mid = F32x8::splat(0.5); // -> 127

        let packed = u32_lanes(U32x8::pack_rgba(above, below, mid, above));
        let expected: u32 = 255 | (127 << 16) | (255 << 24);
        assert_eq!(packed, [expected; 8]);
    }

    fn mask_bits(m: Mask8) -> [u32; 8] {
        let mut out = [0.0f32; 8];
        F32x8::mask_to_float(m).store(&mut out);
        out.map(f32::to_bits)
    }

    #[test]
    fn float_to_mask_should_reinterpret_a_nonzero_bit_pattern_as_a_true_mask() {
        let ones = F32x8::from_u32_bits(u32::MAX);
        let mask = ones.float_to_mask();
        assert!(mask.all());
        assert_eq!(mask_bits(mask), [u32::MAX; 8]);
    }

    #[test]
    fn mask8_bitand_bitor_and_not_should_combine_masks_lanewise() {
        // Mask8's own BitAnd/BitOr/Not (as opposed to F32x8's) had no
        // direct coverage: every prior test only ever combined masks
        // indirectly via `any`/`all`, which a wrong-but-nonzero mask can
        // still satisfy.
        let a =
            F32x8::from_slice(&[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]).cmp_gt(F32x8::splat(0.0));
        let b =
            F32x8::from_slice(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]).cmp_gt(F32x8::splat(0.0));
        const T: u32 = u32::MAX;

        assert_eq!(mask_bits(a & b), [T, 0, 0, 0, T, 0, 0, 0]);
        assert_eq!(mask_bits(a | b), [T, T, T, 0, T, T, T, 0]);
        assert_eq!(mask_bits(!a), [0, 0, T, T, 0, 0, T, T]);
    }

    #[test]
    fn cmp_le_ge_eq_and_ne_should_each_produce_a_distinct_comparison_mask() {
        let a = F32x8::sequential(0.0); // [0, 1, 2, 3, 4, 5, 6, 7]
        let b = F32x8::splat(2.0);
        const T: u32 = u32::MAX;

        assert_eq!(
            mask_bits(a.cmp_le(b)),
            [T, T, T, 0, 0, 0, 0, 0],
            "0,1,2 <= 2; the rest are not"
        );
        assert_eq!(
            mask_bits(a.cmp_ge(b)),
            [0, 0, T, T, T, T, T, T],
            "2..7 >= 2; 0,1 are not"
        );
        assert_eq!(
            mask_bits(a.cmp_eq(b)),
            [0, 0, T, 0, 0, 0, 0, 0],
            "only lane 2 equals 2"
        );
        assert_eq!(
            mask_bits(a.cmp_ne(b)),
            [T, T, 0, T, T, T, T, T],
            "every lane but 2 differs"
        );
    }

    #[test]
    fn from_slice_should_load_eight_consecutive_values_starting_at_the_given_offset() {
        let data = [7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
        assert_eq!(
            lanes(F32x8::from_slice(&data[1..])),
            [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
        );
    }

    #[test]
    fn mul_add_should_compute_self_times_b_plus_c() {
        // self=2, b=3, c=4: `-`/`*` for `+`, and `+`/`/` for `*`, all disagree
        // with the correct 10.0 at these operands.
        let got = lanes(F32x8::splat(2.0).mul_add(F32x8::splat(3.0), F32x8::splat(4.0)));
        assert_eq!(got, [10.0; 8]);
    }

    #[test]
    fn u32x8_bitwise_operators_should_combine_lanes() {
        let a = U32x8::splat(0b1100);
        let b = U32x8::splat(0b1010);

        assert_eq!(u32_lanes(a & b), [0b1000; 8]);
        assert_eq!(u32_lanes(a | b), [0b1110; 8]);
        assert_eq!(u32_lanes(!a), [!0b1100u32; 8]);
    }

    #[test]
    fn u32x8_shift_operators_should_shift_every_lane() {
        let v = U32x8::splat(0b1000);
        assert_eq!(u32_lanes(v << 2), [0b10_0000; 8]);
        assert_eq!(u32_lanes(v >> 2), [0b10; 8]);
    }
}
