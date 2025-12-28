#[cfg(feature = "webgpu")]
use crate::gpu::webgpu::Context as WebGpuContext;
use crate::gpu::vulkan::{ContextInfo, VulkanContext};
use crate::{CommandQueue, Fence, QueueType, Result, SubmitInfo};
use super::execution::CommandRing;

#[cfg(any(feature = "vulkan", feature = "webgpu"))]
enum ContextBackend {
    #[cfg(feature = "vulkan")]
    Vulkan(VulkanContext),
    #[cfg(feature = "webgpu")]
    WebGpu(WebGpuContext),
}

/// Public GPU context facade that dispatches to the selected backend.
#[cfg(any(feature = "vulkan", feature = "webgpu"))]
pub struct Context {
    backend: ContextBackend,
}

#[cfg(any(feature = "vulkan", feature = "webgpu"))]
impl Context {
    /// Construct a [`Context`] with windowing support.
    pub fn new(info: &ContextInfo) -> Result<Self> {
        #[cfg(feature = "webgpu")]
        {
            return Ok(Self {
                backend: ContextBackend::WebGpu(WebGpuContext::new(info)?),
            });
        }
        #[cfg(feature = "vulkan")]
        {
            return Ok(Self {
                backend: ContextBackend::Vulkan(VulkanContext::new(info)?),
            });
        }
        #[allow(unreachable_code)]
        Err(crate::gpu::vulkan::GPUError::Unimplemented(
            "No GPU backend enabled",
        ))
    }

    /// Construct a [`Context`] without any windowing support.
    pub fn headless(info: &ContextInfo) -> Result<Self> {
        #[cfg(feature = "webgpu")]
        {
            return Ok(Self {
                backend: ContextBackend::WebGpu(WebGpuContext::headless(info)?),
            });
        }
        #[cfg(feature = "vulkan")]
        {
            return Ok(Self {
                backend: ContextBackend::Vulkan(VulkanContext::headless(info)?),
            });
        }
        #[allow(unreachable_code)]
        Err(crate::gpu::vulkan::GPUError::Unimplemented(
            "No GPU backend enabled",
        ))
    }

    /// Explicitly destroy the context and release backend resources.
    pub fn destroy(self) {
        match self.backend {
            #[cfg(feature = "vulkan")]
            ContextBackend::Vulkan(ctx) => ctx.destroy(),
            #[cfg(feature = "webgpu")]
            ContextBackend::WebGpu(ctx) => ctx.destroy(),
        }
    }

    pub(crate) fn vulkan(&self) -> Option<&VulkanContext> {
        match &self.backend {
            #[cfg(feature = "vulkan")]
            ContextBackend::Vulkan(ctx) => Some(ctx),
            #[cfg(feature = "webgpu")]
            ContextBackend::WebGpu(ctx) => Some(&*ctx),
        }
    }

    pub(crate) fn vulkan_mut(&mut self) -> Option<&mut VulkanContext> {
        match &mut self.backend {
            #[cfg(feature = "vulkan")]
            ContextBackend::Vulkan(ctx) => Some(ctx),
            #[cfg(feature = "webgpu")]
            ContextBackend::WebGpu(ctx) => Some(&mut *ctx),
        }
    }

    pub fn begin_command_queue(
        &mut self,
        queue_type: QueueType,
        debug_name: &str,
        is_secondary: bool,
    ) -> Result<CommandQueue> {
        let ctx_ptr = self.backend_mut_ptr();
        self.pool_mut(queue_type)
            .begin_raw(ctx_ptr, debug_name, is_secondary)
    }

    pub fn submit_command_queue(
        &mut self,
        queue: &mut CommandQueue,
        info: &SubmitInfo,
    ) -> Result<crate::Handle<Fence>> {
        match &mut self.backend {
            #[cfg(feature = "vulkan")]
            ContextBackend::Vulkan(ctx) => ctx.submit(queue, info),
            #[cfg(feature = "webgpu")]
            ContextBackend::WebGpu(ctx) => ctx.submit(queue, info),
        }
    }

    pub fn wait_fence(&mut self, fence: crate::Handle<Fence>) -> Result<()> {
        match &mut self.backend {
            #[cfg(feature = "vulkan")]
            ContextBackend::Vulkan(ctx) => ctx.wait(fence),
            #[cfg(feature = "webgpu")]
            ContextBackend::WebGpu(ctx) => ctx.wait(fence),
        }
    }

    pub fn destroy_command_queue(&mut self, queue: CommandQueue) {
        match &mut self.backend {
            #[cfg(feature = "vulkan")]
            ContextBackend::Vulkan(ctx) => ctx.destroy_cmd_queue(queue),
            #[cfg(feature = "webgpu")]
            ContextBackend::WebGpu(ctx) => ctx.destroy_cmd_queue(queue),
        }
    }

    pub fn make_command_ring(&mut self, info: &crate::CommandQueueInfo2) -> Result<CommandRing> {
        CommandRing::new(self, info.debug_name, 3, info.queue_type)
    }

    pub(crate) fn backend_mut_ptr(&mut self) -> *mut VulkanContext {
        match &mut self.backend {
            #[cfg(feature = "vulkan")]
            ContextBackend::Vulkan(ctx) => ctx as *mut VulkanContext,
            #[cfg(feature = "webgpu")]
            ContextBackend::WebGpu(ctx) => ctx.backend_mut_ptr(),
        }
    }
}

#[cfg(any(feature = "vulkan", feature = "webgpu"))]
impl std::ops::Deref for Context {
    type Target = VulkanContext;

    fn deref(&self) -> &Self::Target {
        self.vulkan()
            .expect("Vulkan backend not active")
    }
}

#[cfg(any(feature = "vulkan", feature = "webgpu"))]
impl std::ops::DerefMut for Context {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vulkan_mut()
            .expect("Vulkan backend not active")
    }
}
