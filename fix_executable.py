import re

with open("pixelflow-ir/src/backend/emit/executable.rs", "r") as f:
    content = f.read()

# Let's fix ExecutableCode::from_code
search = """            // Instruction cache coherence on Apple Silicon.
            // sys_icache_invalidate is needed after writing code on ARM.
            #[cfg(target_os = "macos")]
            {
                extern "C" {
                    fn sys_icache_invalidate(start: *mut core::ffi::c_void, size: usize);
                }
                sys_icache_invalidate(ptr as *mut core::ffi::c_void, code.len());
            }

            // 3. Flip to read-execute (W^X)
            let result = mprotect(ptr as *mut libc::c_void, capacity, PROT_READ | PROT_EXEC);"""

replace = """            // 3. Flip to read-execute (W^X)
            let result = mprotect(ptr as *mut libc::c_void, capacity, PROT_READ | PROT_EXEC);

            // Instruction cache coherence on Apple Silicon.
            // sys_icache_invalidate is needed after writing code on ARM.
            #[cfg(target_os = "macos")]
            {
                extern "C" {
                    fn sys_icache_invalidate(start: *mut core::ffi::c_void, size: usize);
                }
                sys_icache_invalidate(ptr as *mut core::ffi::c_void, code.len());
            }"""

if search in content:
    content = content.replace(search, replace)
    with open("pixelflow-ir/src/backend/emit/executable.rs", "w") as f:
        f.write(content)
    print("Patched executable.rs")
else:
    print("Could not find search string in executable.rs")
