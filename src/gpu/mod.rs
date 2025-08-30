use crate::Handle;

/// Abstraction over GPU backends.
///
/// Each backend implements this trait on its context type, allowing code to be
/// written generically over the supported graphics APIs.
pub trait Backend {
    /// Backend-specific description of a buffer to create.
    type BufferInfo<'a>;
    /// Raw buffer handle returned by the backend.
    type Buffer;
    /// Command list type used for submissions.
    type CommandList;
    /// Parameters controlling how work is submitted to the GPU.
    type SubmitInfo<'a>;
    /// Fence handle returned from submissions for synchronization.
    type Fence;
    /// Error type produced by the backend.
    type Error;

    /// Create a GPU buffer with the provided description.
    fn create_buffer<'a>(
        &mut self,
        info: &Self::BufferInfo<'a>,
    ) -> std::result::Result<Handle<Self::Buffer>, Self::Error>;

    /// Submit a command list for execution.
    fn submit<'a>(
        &mut self,
        cmd: &mut Self::CommandList,
        info: &Self::SubmitInfo<'a>,
    ) -> std::result::Result<Handle<Self::Fence>, Self::Error>;
}

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "vulkan")]
pub use vulkan::*;
