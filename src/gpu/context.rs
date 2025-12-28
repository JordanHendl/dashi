#[cfg(feature = "vulkan")]
use crate::gpu::vulkan::{ContextInfo, VulkanContext};
use crate::{CommandQueue, Fence, QueueType, Result, SubmitInfo};
use super::execution::CommandRing;

#[cfg(feature = "vulkan")]
enum ContextBackend {
    Vulkan(VulkanContext),
}

/// Public GPU context facade that dispatches to the selected backend.
#[cfg(feature = "vulkan")]
pub struct Context {
    backend: ContextBackend,
}

#[cfg(feature = "vulkan")]
impl Context {
    /// Construct a [`Context`] with windowing support.
    pub fn new(info: &ContextInfo) -> Result<Self> {
        Ok(Self {
            backend: ContextBackend::Vulkan(VulkanContext::new(info)?),
        })
    }

    /// Construct a [`Context`] without any windowing support.
    pub fn headless(info: &ContextInfo) -> Result<Self> {
        Ok(Self {
            backend: ContextBackend::Vulkan(VulkanContext::headless(info)?),
        })
    }

    /// Explicitly destroy the context and release backend resources.
    pub fn destroy(self) {
        match self.backend {
            ContextBackend::Vulkan(ctx) => ctx.destroy(),
        }
    }

    pub(crate) fn vulkan(&self) -> Option<&VulkanContext> {
        match &self.backend {
            ContextBackend::Vulkan(ctx) => Some(ctx),
        }
    }

    pub(crate) fn vulkan_mut(&mut self) -> Option<&mut VulkanContext> {
        match &mut self.backend {
            ContextBackend::Vulkan(ctx) => Some(ctx),
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
            ContextBackend::Vulkan(ctx) => ctx.submit(queue, info),
        }
    }

    pub fn wait_fence(&mut self, fence: crate::Handle<Fence>) -> Result<()> {
        match &mut self.backend {
            ContextBackend::Vulkan(ctx) => ctx.wait(fence),
        }
    }

    pub fn destroy_command_queue(&mut self, queue: CommandQueue) {
        match &mut self.backend {
            ContextBackend::Vulkan(ctx) => ctx.destroy_cmd_queue(queue),
        }
    }

    pub fn make_command_ring(&mut self, info: &crate::CommandQueueInfo2) -> Result<CommandRing> {
        CommandRing::new(self, info.debug_name, 3, info.queue_type)
    }

    pub(crate) fn backend_mut_ptr(&mut self) -> *mut VulkanContext {
        self.vulkan_mut()
            .map(|ctx| ctx as *mut VulkanContext)
            .expect("Vulkan backend not active")
    }
}

#[cfg(feature = "vulkan")]
impl std::ops::Deref for Context {
    type Target = VulkanContext;

    fn deref(&self) -> &Self::Target {
        self.vulkan()
            .expect("Vulkan backend not active")
    }
}

#[cfg(feature = "vulkan")]
impl std::ops::DerefMut for Context {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vulkan_mut()
            .expect("Vulkan backend not active")
    }
}
