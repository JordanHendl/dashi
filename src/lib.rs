pub mod gpu;
pub mod utils;

#[cfg(all(feature = "dashi-sdl2", feature = "dashi-minifb"))]
compile_error!("window backends are mutually exclusive; enable only one of `dashi-sdl2` or `dashi-minifb`");

pub use gpu::*;
