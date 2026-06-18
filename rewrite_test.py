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
