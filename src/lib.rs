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
    // Only fill: fill![..x; N]
    (.. $fill:expr ; $len:expr $(,)?) => {{
        [$fill; $len]
    }};

    // Prefix tokens up to `..`, then fill and length.
    ($($prefix:tt)+ .. $fill:expr ; $len:expr $(,)?) => {{
        let mut __arr = [$fill; $len];
        $crate::fill!(@set __arr, 0usize, $($prefix)+);
        __arr
    }};

    // Accept an empty prefix (optional convenience): fill![..x; N]
    (@set $arr:ident, $i:expr $(,)?) => {};

    // Munch `expr, expr, expr, ...` and assign into the array.
    (@set $arr:ident, $i:expr, $head:expr $(, $tail:expr)* $(,)?) => {{
        $arr[$i] = $head;
        $crate::fill!(@set $arr, $i + 1usize, $($tail),*);
    }};
}

pub use gpu::*;
