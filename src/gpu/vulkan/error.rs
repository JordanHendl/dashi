use super::structs::{Format, SampleCount, ShaderType, WindowMode};
use ash::vk;
use std::fmt;

#[derive(Debug)]
pub struct VulkanError {
    res: ash::vk::Result,
}

#[derive(Debug)]
pub struct SlotError {}

impl fmt::Display for SlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ran out of slots!")
    }
}
impl fmt::Display for VulkanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Vulkan Error: {}", self.res.to_string())
    }
}

#[derive(Debug)]
pub enum GPUError {
    VulkanError(VulkanError),
    LoadingError(ash::LoadingError),
    LibraryError(String),
    SlotError(),
    HeadlessDisplayNotSupported,
    UnsupportedWindowMode {
        backend: &'static str,
        mode: WindowMode,
    },
    DisplayNeedsRebuild,
    UnsupportedFormat(vk::Format),
    SwapchainConfigError(&'static str),
    UnsupportedShaderStage(ShaderType),
    Unimplemented(&'static str),
    InvalidBindTableBinding {
        binding: u32,
        reason: String,
    },
    BufferBarrierInsideRenderPass {
        buffer: String,
        old_usage: crate::UsageBits,
        new_usage: crate::UsageBits,
    },
    MismatchedAttachmentFormat {
        context: String,
        expected: Format,
        actual: Format,
    },
    MismatchedSampleCount {
        context: String,
        expected: SampleCount,
        actual: SampleCount,
    },
    InvalidSubpass {
        subpass: u32,
        available: u32,
    },
}

/// Convenient crate-wide result type.
pub type Result<T, E = GPUError> = std::result::Result<T, E>;

//impl From<SlotError> for GPUError {
//    fn from(res: ash::vk::Result) -> Self {
//        return GPUError::VulkanError(VulkanError{res});
//    }
//}
//

impl std::error::Error for GPUError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }

    fn description(&self) -> &str {
        "description() is deprecated; use Display"
    }

    fn cause(&self) -> Option<&dyn std::error::Error> {
        self.source()
    }
}

impl fmt::Display for GPUError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GPUError::VulkanError(e) => write!(f, "{}", e),
            GPUError::LoadingError(e) => write!(f, "{}", e),
            GPUError::LibraryError(msg) => write!(f, "Library failed to initialize. Error: {}", msg),
            GPUError::SlotError() => write!(f, "Slot Error"),
            GPUError::HeadlessDisplayNotSupported => write!(f, "Headless Display not supported"),
            GPUError::UnsupportedWindowMode { backend, mode } => write!(
                f,
                "Window mode {:?} is not supported by the {} backend",
                mode, backend
            ),
            GPUError::DisplayNeedsRebuild => write!(f, "Display needs rebuild"),
            GPUError::UnsupportedFormat(format) => write!(f, "Format {:?} not supported", format),
            GPUError::SwapchainConfigError(msg) => {
                write!(f, "Swapchain configuration error: {}", msg)
            }
            GPUError::UnsupportedShaderStage(stage) => write!(f, "Shader Stage {:?} not supported", stage),
            GPUError::Unimplemented(feature) => write!(f, "Unimplemented Feature: {}", feature),
            GPUError::InvalidBindTableBinding { binding, reason } => write!(
                f,
                "Invalid bind table update for binding {}: {}",
                binding, reason
            ),
            GPUError::BufferBarrierInsideRenderPass {
                buffer,
                old_usage,
                new_usage,
            } => write!(
                f,
                "Buffer barrier for '{}' is required inside an active render pass ({:?} -> {:?}); prepare the buffer before beginning the render pass",
                buffer, old_usage, new_usage
            ),
            GPUError::MismatchedAttachmentFormat {
                context,
                expected,
                actual,
            } => write!(
                f,
                "Attachment format mismatch for {}: expected {:?}, got {:?}.",
                context, expected, actual
            ),
            GPUError::MismatchedSampleCount {
                context,
                expected,
                actual,
            } => write!(
                f,
                "Multisample mismatch for {}: expected {:?}, got {:?}. Ensure image, render pass, and graphics pipeline sample counts match.",
                context, expected, actual
            ),
            GPUError::InvalidSubpass { subpass, available } => write!(
                f,
                "Invalid subpass index {} (render pass has {} subpasses)",
                subpass, available
            ),
        }
    }
}

impl From<anyhow::Error> for GPUError {
    fn from(_value: anyhow::Error) -> Self {
        todo!()
    }
}

#[cfg(feature = "dashi-serde")]
impl From<serde_yaml::Error> for GPUError {
    fn from(_value: serde_yaml::Error) -> Self {
        todo!()
    }
}

impl From<ash::vk::Result> for GPUError {
    fn from(res: ash::vk::Result) -> Self {
        return GPUError::VulkanError(VulkanError { res });
    }
}

impl<T> From<(T, ash::vk::Result)> for GPUError {
    fn from(res: (T, ash::vk::Result)) -> Self {
        return GPUError::VulkanError(VulkanError { res: res.1 });
    }
}

impl From<SlotError> for GPUError {
    fn from(_res: SlotError) -> Self {
        return GPUError::SlotError();
    }
}

impl From<ash::LoadingError> for GPUError {
    fn from(res: ash::LoadingError) -> Self {
        return GPUError::LoadingError(res);
    }
}
