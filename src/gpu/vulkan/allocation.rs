use super::*;
use crate::gpu::external::ExternalMemoryInfo;

#[derive(Debug)]
pub(crate) enum Allocation {
    Vma(vk_mem::Allocation),
    External {
        memory: vk::DeviceMemory,
        info: ExternalMemoryInfo,
        exportable: bool,
    },
    Borrowed,
}

impl Clone for Allocation {
    fn clone(&self) -> Self {
        match self {
            Self::Vma(a) => Self::Vma(unsafe { std::mem::transmute_copy(a) }),
            Self::External {
                memory,
                info,
                exportable,
            } => Self::External {
                memory: *memory,
                info: *info,
                exportable: *exportable,
            },
            Self::Borrowed => Self::Borrowed,
        }
    }
}

impl Allocation {
    pub(crate) fn vma(&self) -> Result<&vk_mem::Allocation> {
        match self {
            Self::Vma(a) => Ok(a),
            _ => Err(GPUError::Unimplemented(
                "External memory is not host-mappable",
            )),
        }
    }
    pub(crate) unsafe fn destroy_image(
        &mut self,
        ctx_device: &ash::Device,
        allocator: &vk_mem::Allocator,
        image: vk::Image,
    ) {
        match self {
            Self::Vma(a) => unsafe { allocator.destroy_image(image, a) },
            Self::External { memory, .. } => unsafe {
                ctx_device.destroy_image(image, None);
                ctx_device.free_memory(*memory, None);
            },
            Self::Borrowed => {}
        }
    }
    pub(crate) unsafe fn destroy_buffer(
        &mut self,
        device: &ash::Device,
        allocator: &vk_mem::Allocator,
        buffer: vk::Buffer,
    ) {
        match self {
            Self::Vma(a) => unsafe { allocator.destroy_buffer(buffer, a) },
            Self::External { memory, .. } => unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(*memory, None);
            },
            Self::Borrowed => {}
        }
    }
}
