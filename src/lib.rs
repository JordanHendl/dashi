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
    // Prefix values, then ".. fill", then "; N"
    ($($val:expr),+ , .. $fill:expr ; $len:expr $(,)?) => {{
        let mut arr = [$fill; $len]; // requires T: Clone if not Copy
        let mut i = 0usize;
        $(
            arr[i] = $val;
            i += 1;
        )+
        arr
    }};

    // Just fill: arrfill![..fill; N]
    (.. $fill:expr ; $len:expr $(,)?) => {{
        [$fill; $len]
    }};
}

pub use gpu::*;
