pub mod utils;
pub mod gpu;
pub use gpu::driver::types::{Handle, IndexType, ResourceUse, UsageBits};

#[cfg(
    any(
        all(feature = "dashi-sdl2", feature = "dashi-minifb"),
        all(feature = "dashi-sdl2", feature = "dashi-winit"),
        all(feature = "dashi-minifb", feature = "dashi-winit"),
    )
)]
compile_error!(
    "window backends are mutually exclusive; enable only one of `dashi-sdl2`, `dashi-minifb`, or `dashi-winit`"
);

#[macro_export]
macro_rules! fill {
    // Only fill: fill![..=> fill; N]
    (..=> $fill:expr ; $len:expr $(,)?) => {{
        [$fill; $len]
    }};

    // Prefix + fill: fill![a, b, ..=> fill; N]
    ($($val:expr),+ , ..=> $fill:expr ; $len:expr $(,)?) => {{
        let mut __arr = [$fill; $len];
        let mut __i = 0usize;
        $(
            __arr[__i] = $val;
            __i += 1;
        )+
        __arr
    }};
}


pub use gpu::*;
