use std::fmt;
use ash::vk;
use super::structs::ShaderType;

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
    LibraryError(),
    SlotError(),
    HeadlessDisplayNotSupported,
    UnsupportedFormat(vk::Format),
    UnsupportedShaderStage(ShaderType),
    Unimplemented(&'static str),
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
            GPUError::LibraryError() => write!(f, "Library could not be loaded"),
            GPUError::SlotError() => write!(f, "Slot Error"),
            GPUError::HeadlessDisplayNotSupported => write!(f, "Headless Display not supported"),
            GPUError::UnsupportedFormat(format) => write!(f, "Format {:?} not supported", format),
            GPUError::UnsupportedShaderStage(stage) => write!(f, "Shader Stage {:?} not supported", stage),
            GPUError::Unimplemented(feature) => write!(f, "Unimplemented Feature: {}", feature),
        }
    }
}

impl From<anyhow::Error> for GPUError {
    fn from(value: anyhow::Error) -> Self { 
        todo!()
    }
}

#[cfg(feature = "dashi-serde")]
impl From<serde_yaml::Error> for GPUError {
    fn from(value: serde_yaml::Error) -> Self { 
        todo!()
    }
}

impl From<ash::vk::Result> for GPUError {
    fn from(res: ash::vk::Result) -> Self {
        return GPUError::VulkanError(VulkanError { res });
    }
}

impl From<ash::LoadingError> for GPUError {
    fn from(res: ash::LoadingError) -> Self {
        return GPUError::LoadingError(res);
    }
}
