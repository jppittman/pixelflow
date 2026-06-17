
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::float32x4_t;
#[cfg(target_arch = "aarch64")]
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn naked_abi_multithreaded_scale_should_succeed() {
    #[cfg(target_arch = "aarch64")]
    {
        use pixelflow_ir::backend::emit::executable::{CodeBuffer, KernelFn};

        let num_threads = 16;
        let ops_per_thread = 10_000;

        let code: [u32; 2] = [
            0x4E21D800, // fmul v0.4s, v0.4s, v1.4s
            0xD65F03C0, // ret
        ];
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(code.as_ptr() as *const u8, code.len() * 4) };

        let mut buf = CodeBuffer::new(4096).unwrap();
        // write_code returns a KernelFn which safely encapsulates the JIT call and memory protections
        let kernel_fn: KernelFn = unsafe { buf.write_code(bytes).unwrap() };

        // We leak the buffer so it lives forever for the threads to use.
        // In a real app we'd share it via Arc.
        let kernel_fn_ptr = kernel_fn as usize;
        std::mem::forget(buf);

        let total_successes = AtomicUsize::new(0);

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let successes = &total_successes;
                s.spawn(move || {
                    let mut local_successes = 0;
                    // Safely cast back to KernelFn
                    let func: KernelFn = unsafe { std::mem::transmute(kernel_fn_ptr) };

                    let x = [1.0f32, 2.0, 3.0, 4.0];
                    let y = [2.0f32, 2.0, 2.0, 2.0];
                    let zero = [0.0f32; 4];

                    for _ in 0..ops_per_thread {
                        unsafe {
                            let vx: float32x4_t = std::mem::transmute(x);
                            let vy: float32x4_t = std::mem::transmute(y);
                            let vz: float32x4_t = std::mem::transmute(zero);
                            let vw: float32x4_t = std::mem::transmute(zero);

                            let res = func(vx, vy, vz, vw);
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
