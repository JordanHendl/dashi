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
    LibraryError(),
    SlotError(),
}

//impl From<SlotError> for GPUError {
//    fn from(res: ash::vk::Result) -> Self {
//        return GPUError::VulkanError(VulkanError{res});
//    }
//}
//

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
