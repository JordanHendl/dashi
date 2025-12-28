/// Defines the interface that rendering backends must implement.
///
/// A backend wraps the low-level graphics API and exposes a
/// type-safe [`Context`] used to create and manage GPU resources.
///
/// # Examples
/// ```ignore
/// use dashi::gpu::Backend;
/// fn init<B: Backend>(ctx: B::Context) {
///     // store the backend-specific context
/// }
/// ```
pub trait Backend {
    /// Backend specific context type used for resource creation.
    type Context;
}

#[cfg(any(feature = "vulkan", feature = "webgpu"))]
pub mod context;
#[cfg(any(feature = "vulkan", feature = "webgpu"))]
pub use context::Context;

pub mod execution;
pub mod driver;
pub mod cmd;
pub use cmd::{
    CommandStream,
};
#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "vulkan")]
pub use vulkan::*;

#[cfg(feature = "webgpu")]
pub mod webgpu;

#[cfg(feature = "webgpu")]
pub use webgpu::*;
