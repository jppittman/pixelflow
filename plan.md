1. **Fix `test_naked_abi_multithreaded_scale` by initializing `pthread_jit_write_protect_np` in worker threads**
   - The test `test_naked_abi_multithreaded_scale` crashes with SIGBUS. I leaked `CodeBuffer` to share it across threads. `CodeBuffer::write_code` successfully mapped it as JIT memory and the main thread toggled it to executable via `pthread_jit_write_protect_np(1)`.
   - However, `pthread_jit_write_protect_np` is thread-local! The worker threads created via `std::thread::scope` have not explicitly enabled `pthread_jit_write_protect_np(1)`. The default state for new threads might be `Writable` (0) instead of `Executable` (1) on some Apple Silicon macOS versions for MAP_JIT regions, or it just requires explicit initialization.
   - Wait, `CodeBuffer::write_code` DOES call `sys_icache_invalidate` which operates on the memory region, BUT `pthread_jit_write_protect_np` is definitely thread-local.
   - I will modify `pixelflow-core/tests/naked_scale.rs` to call `pthread_jit_write_protect_np(1)` inside the worker thread loop OR at the beginning of each spawned thread.

cat << 'SCRIPT' > rewrite_test.py
with open('./pixelflow-core/tests/naked_scale.rs', 'r') as f:
    content = f.read()

content = content.replace("""                s.spawn(move || {
                    let mut local_successes = 0;""", """                s.spawn(move || {
                    #[cfg(target_os = "macos")]
                    unsafe {
                        extern "C" {
                            fn pthread_jit_write_protect_np(enabled: core::ffi::c_int);
                        }
                        pthread_jit_write_protect_np(1);
                    }
                    let mut local_successes = 0;""")

with open('./pixelflow-core/tests/naked_scale.rs', 'w') as f:
    f.write(content)
SCRIPT
python3 rewrite_test.py

2. **Verify Changes and Code Health**
    - Run `cargo test -p pixelflow-core --test naked_scale` to verify the test completes quickly and doesn't timeout or crash on macOS.
    - Run `rm rewrite_test.py` to clean up the temporary script.

3. **Complete Pre-Commit Steps**
    - Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.
