//! Raw System Bindings (The Engine Room)
//!
//! This module contains the raw `extern "C"` definitions and unsafe
//! typed helper functions for the Objective-C runtime.
//!
//! # Safety
//!
//! Everything here is unsafe. This is lowest-level FFI.

use std::ffi::c_void;

// --- Types ---

pub type Id = *mut c_void;
pub type Sel = *mut c_void;
pub type Class = *mut c_void;
pub type BOOL = i8;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CGSize {
    pub width: f64,
    pub height: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CGPoint {
    pub x: f64,
    pub y: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CGRect {
    pub origin: CGPoint,
    pub size: CGSize,
}

pub const YES: BOOL = 1;
pub const NO: BOOL = 0;

pub const NS_APP_DEFINED: u64 = 15;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLOrigin {
    pub x: usize, // NSUInteger
    pub y: usize,
    pub z: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLSize {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MTLRegion {
    pub origin: MTLOrigin,
    pub size: MTLSize,
}

impl MTLRegion {
    pub fn new_2d(x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            origin: MTLOrigin { x, y, z: 0 },
            size: MTLSize {
                width,
                height,
                depth: 1,
            },
        }
    }
}

// --- Raw FFI ---

#[link(name = "Cocoa", kind = "framework")]
extern "C" {
    pub fn objc_getClass(name: *const u8) -> Class;
    pub fn sel_registerName(name: *const u8) -> Sel;
    pub fn objc_msgSend(self_: Id, op: Sel, ...) -> Id;
    pub fn objc_allocateClassPair(superclass: Class, name: *const u8, extra_bytes: usize)
        -> Class;
    pub fn objc_registerClassPair(cls: Class);
    pub fn class_addMethod(cls: Class, name: Sel, imp: *const c_void, types: *const u8) -> BOOL;
}

// --- Inline Helpers ---

/// Get a class by name.
/// Use `b"ClassName\0"` for zero allocation.
#[inline(always)]
pub unsafe fn class(name: &[u8]) -> Class {
    objc_getClass(name.as_ptr())
}

/// Get a selector by name.
/// Use `b"selectorName\0"` for zero allocation.
#[inline(always)]
pub unsafe fn sel(name: &[u8]) -> Sel {
    sel_registerName(name.as_ptr())
}

/// Message Send (0 args, return R)
#[inline(always)]
pub unsafe fn send<R>(obj: Id, sel: Sel) -> R {
    let fn_ptr: unsafe extern "C" fn(Id, Sel) -> R =
        std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel)
}

/// Message Send (1 arg)
#[inline(always)]
pub unsafe fn send_1<R, A>(obj: Id, sel: Sel, a: A) -> R {
    let fn_ptr: unsafe extern "C" fn(Id, Sel, A) -> R =
        std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel, a)
}

/// Message Send (2 args)
#[inline(always)]
pub unsafe fn send_2<R, A, B>(obj: Id, sel: Sel, a: A, b: B) -> R {
    let fn_ptr: unsafe extern "C" fn(Id, Sel, A, B) -> R =
        std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel, a, b)
}

/// Message Send (3 args)
#[inline(always)]
pub unsafe fn send_3<R, A, B, C>(obj: Id, sel: Sel, a: A, b: B, c: C) -> R {
    let fn_ptr: unsafe extern "C" fn(Id, Sel, A, B, C) -> R =
        std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel, a, b, c)
}

/// Message Send (4 args)
#[inline(always)]
pub unsafe fn send_4<R, A, B, C, D>(obj: Id, sel: Sel, a: A, b: B, c: C, d: D) -> R {
    let fn_ptr: unsafe extern "C" fn(Id, Sel, A, B, C, D) -> R =
        std::mem::transmute(objc_msgSend as *const c_void);
    fn_ptr(obj, sel, a, b, c, d)
}
