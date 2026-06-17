use std::arch::aarch64::float32x4_t;

#[inline(never)]
pub extern "C" fn dummy_jit_kernel(x: float32x4_t, y: float32x4_t, _z: float32x4_t, _w: float32x4_t) -> float32x4_t {
    // A completely unsafe dummy kernel to simulate our JIT payload.
    // It multiplies x and y and returns it.
    unsafe {
        std::arch::asm!(
            "fmul v0.4s, v0.4s, v1.4s",
            options(nostack)
        );
        std::mem::transmute([0.0f32; 4]) // Return value is physically in v0
    }
}

pub fn naked_call_should_succeed_when_executed() {
    let x_arr = [2.0f32, 3.0, 4.0, 5.0];
    let y_arr = [10.0f32, 10.0, 10.0, 10.0];
    
    unsafe {
        let vx: float32x4_t = std::mem::transmute(x_arr);
        let vy: float32x4_t = std::mem::transmute(y_arr);
        let vz: float32x4_t = std::mem::transmute([0.0f32; 4]);
        let vw: float32x4_t = std::mem::transmute([0.0f32; 4]);
        
        let func_ptr = dummy_jit_kernel as *const u8;
        let mut result: float32x4_t;

        std::arch::asm!(
            "blr {func}",
            func = in(reg) func_ptr,
            inout("v0") vx => result,
            in("v1") vy,
            in("v2") vz,
            in("v3") vw,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            out("x16") _, out("x17") _,
            options(nomem, nostack, preserves_flags)
        );

        let out_arr: [f32; 4] = std::mem::transmute(result);
        assert_eq!(out_arr, [20.0, 30.0, 40.0, 50.0]);
        println!("Naked call succeeded: {:?}", out_arr);
    }
}
