pub mod utils;
pub mod driver;
pub mod ir;
pub mod sync;
pub mod gfx;
pub mod framegraph;

pub use driver::types::{BindTable, Handle, IndexType, UsageBits};
pub use gfx::cmd::{
    CommandBuffer,
    CommandBuilder,
    CommandBuilderExt,
    DebugLabelScope,
    EncodeTarget,
    PipelineBound,
    PipelineBuilder,
    RenderScope,
    DescriptorWriteBuilder,
    DynamicRenderingBuilder,
};
pub use driver::command::Recorder;

pub mod gpu;
#[cfg(feature = "dx12")]
pub mod gpu_dx12;
#[cfg(feature = "metal")]
pub mod gpu_metal;

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


pub use gpu::*;
#[cfg(feature = "dx12")]
#[allow(unused_imports)]
pub use gpu_dx12::*;
#[cfg(feature = "metal")]
#[allow(unused_imports)]
pub use gpu_metal::*;
