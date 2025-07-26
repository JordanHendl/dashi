pub mod gpu;
pub mod utils;

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

#[cfg(all(
    feature = "dashi-openxr",
    any(feature = "dashi-winit", feature = "dashi-sdl2", feature = "dashi-minifb"),
))]
compile_error!("`dashi-openxr` cannot be enabled alongside window backends");

#[cfg(all(feature = "dashi-vulkan", feature = "dashi-dx12"))]
compile_error!("GPU backends are mutually exclusive; enable only one of `dashi-vulkan` or `dashi-dx12`");

pub use gpu::*;
