use ash::{vk, Device};
use std::{cell::UnsafeCell, marker::PhantomData, thread::ThreadId};

use crate::Result;

/// Thin wrapper around a Vulkan command pool.
///
/// Handles allocation and recycling of primary/secondary command buffers and
/// enforces single-threaded ownership. The pool may be moved to another thread
/// after creation but must not be shared across threads.
pub struct CommandPool {
    device: Device,
    raw: vk::CommandPool,
    free_primary: Vec<vk::CommandBuffer>,
    free_secondary: Vec<vk::CommandBuffer>,
    owner: ThreadId,
    // make !Sync
    _not_sync: PhantomData<UnsafeCell<()>>,
}

unsafe impl Send for CommandPool {}

impl CommandPool {
    /// Create a new command pool for the given queue family.
    pub fn new(device: Device, family: u32) -> Result<Self> {
        let ci = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .build();
        let raw = unsafe { device.create_command_pool(&ci, None)? };
        Ok(Self {
            device,
            raw,
            free_primary: Vec::new(),
            free_secondary: Vec::new(),
            owner: std::thread::current().id(),
            _not_sync: PhantomData,
        })
    }

    fn assert_owner(&self) {
        debug_assert_eq!(self.owner, std::thread::current().id(), "CommandPool used from wrong thread");
    }

    /// Allocate a primary command buffer, reusing one if available.
    pub fn alloc_primary(&mut self) -> Result<vk::CommandBuffer> {
        self.assert_owner();
        if let Some(buf) = self.free_primary.pop() {
            unsafe {
                self.device
                    .reset_command_buffer(buf, vk::CommandBufferResetFlags::empty())?;
            }
            Ok(buf)
        } else {
            let cmd = unsafe {
                self.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(self.raw)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1)
                        .build(),
                )?
            };
            Ok(cmd[0])
        }
    }

    /// Allocate a secondary command buffer, reusing one if available.
    pub fn alloc_secondary(&mut self) -> Result<vk::CommandBuffer> {
        self.assert_owner();
        if let Some(buf) = self.free_secondary.pop() {
            unsafe {
                self.device
                    .reset_command_buffer(buf, vk::CommandBufferResetFlags::empty())?;
            }
            Ok(buf)
        } else {
            let cmd = unsafe {
                self.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(self.raw)
                        .level(vk::CommandBufferLevel::SECONDARY)
                        .command_buffer_count(1)
                        .build(),
                )?
            };
            Ok(cmd[0])
        }
    }

    /// Recycle a primary command buffer after GPU completion.
    pub fn recycle_primary(&mut self, buf: vk::CommandBuffer) {
        self.assert_owner();
        self.free_primary.push(buf);
    }

    /// Recycle a secondary command buffer after GPU completion.
    pub fn recycle_secondary(&mut self, buf: vk::CommandBuffer) {
        self.assert_owner();
        self.free_secondary.push(buf);
    }

    /// Reset a specific command buffer for reuse.
    pub fn reset_cmd(&self, buf: vk::CommandBuffer) -> Result<()> {
        self.assert_owner();
        unsafe {
            self.device
                .reset_command_buffer(buf, vk::CommandBufferResetFlags::empty())?;
        }
        Ok(())
    }

    /// Destroy the underlying Vulkan command pool. Command buffers allocated
    /// from this pool become invalid after this call.
    pub fn destroy(&mut self) {
        self.assert_owner();
        unsafe {
            self.device.destroy_command_pool(self.raw, None);
        }
        self.raw = vk::CommandPool::null();
        self.free_primary.clear();
        self.free_secondary.clear();
    }

    /// Raw Vulkan command pool handle.
    pub fn raw(&self) -> vk::CommandPool { self.raw }
}
