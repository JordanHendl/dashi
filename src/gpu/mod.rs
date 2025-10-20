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
