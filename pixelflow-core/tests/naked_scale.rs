#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::float32x4_t;
#[cfg(target_arch = "aarch64")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(target_arch = "aarch64")]
fn get_jit_mul_kernel() -> usize {
    // We emit raw AArch64 machine code for:
    // fmul v0.4s, v0.4s, v1.4s
    // ret
    let code: [u32; 2] = [
        0x6E21DC00, // fmul v0.4s, v0.4s, v1.4s
        0xD65F03C0, // ret
    ];
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(code.as_ptr() as *const u8, code.len() * 4) };

    let exec = unsafe {
        pixelflow_ir::backend::emit::executable::ExecutableCode::from_code(bytes).unwrap()
    };

    // Leak the executable so it lives forever (it's just a test)
    let ptr = unsafe { exec.as_fn::<extern "C" fn()>() as usize };
    std::mem::forget(exec);
    ptr
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn invoke_naked_kernel(
    func_ptr: usize,
    vx: float32x4_t,
    vy: float32x4_t,
    vz: float32x4_t,
    vw: float32x4_t,
) -> float32x4_t {
    let mut result: float32x4_t;
    // SAFETY: `func_ptr` is a valid JIT-emitted kernel following the documented
    // NEON ABI (v0..v3 in, v0 out); `clobber_abi("C")` clobbers every register
    // AAPCS64 marks caller-saved (including `lr`/x30, which `blr` overwrites
    // with the return address — a manual clobber list omitted that once and
    // the corrupted return address sent the function's epilogue to garbage,
    // hanging the test).
    unsafe {
        std::arch::asm!(
            "blr {func}",
            func = in(reg) func_ptr,
            inout("v0") vx => result,
            in("v1") vy,
            in("v2") vz,
            in("v3") vw,
            clobber_abi("C"),
            options(nostack)
        );
    }
    result
}

#[test]
fn naked_abi_multithreaded_scale() {
    #[cfg(target_arch = "aarch64")]
    {
        let num_threads = 16;
        let ops_per_thread = 1_000;
        let kernel_ptr = get_jit_mul_kernel();
        let total_successes = AtomicUsize::new(0);

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let successes = &total_successes;
                s.spawn(move || {
                    let mut local_successes = 0;

                    let x = [1.0f32, 2.0, 3.0, 4.0];
                    let y = [2.0f32, 2.0, 2.0, 2.0];
                    let zero = [0.0f32; 4];

                    for _ in 0..ops_per_thread {
                        unsafe {
                            let vx: float32x4_t = std::mem::transmute(x);
                            let vy: float32x4_t = std::mem::transmute(y);
                            let vz: float32x4_t = std::mem::transmute(zero);
                            let vw: float32x4_t = std::mem::transmute(zero);

                            let res = invoke_naked_kernel(kernel_ptr, vx, vy, vz, vw);
                            let out: [f32; 4] = std::mem::transmute(res);

                            // 1*2=2, 2*2=4, 3*2=6, 4*2=8
                            if out == [2.0, 4.0, 6.0, 8.0] {
                                local_successes += 1;
                            }
                        }
                    }
                    successes.fetch_add(local_successes, Ordering::Relaxed);
                });
            }
        });

        assert_eq!(
            total_successes.load(Ordering::SeqCst),
            num_threads * ops_per_thread
        );
        println!(
            "Successfully executed {} Naked ABI calls across {} threads without crashing.",
            num_threads * ops_per_thread,
            num_threads
        );
    }
}
