use ash::{vk, Device};
use std::{cell::UnsafeCell, marker::PhantomData, thread::ThreadId};

use crate::{utils::Handle, Result};

use super::{CommandQueue, Context, Fence, QueueType};

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
    pub(super) fn new(device: Device, family: u32) -> Result<Self> {
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

    fn alloc(&mut self, level: vk::CommandBufferLevel) -> Result<vk::CommandBuffer> {
        self.assert_owner();
        let list = match level {
            vk::CommandBufferLevel::PRIMARY => &mut self.free_primary,
            vk::CommandBufferLevel::SECONDARY => &mut self.free_secondary,
            _ => unreachable!(),
        };
        if let Some(buf) = list.pop() {
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
                        .level(level)
                        .command_buffer_count(1)
                        .build(),
                )?
            };
            Ok(cmd[0])
        }
    }

    /// Begin recording a command queue from this pool.
    pub unsafe fn begin(
        &mut self,
        ctx: *mut Context,
        debug_name: &str,
        queue_type: QueueType,
        is_secondary: bool,
    ) -> Result<CommandQueue> {
        let level = if is_secondary {
            vk::CommandBufferLevel::SECONDARY
        } else {
            vk::CommandBufferLevel::PRIMARY
        };
        let cmd_buf = self.alloc(level)?;

        (*ctx).set_name(cmd_buf, debug_name, vk::ObjectType::COMMAND_BUFFER);

        let f = (*ctx)
            .device
            .create_fence(
                &vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::empty())
                    .build(),
                None,
            )?;
        (*ctx).set_name(
            f,
            format!("{}.fence", debug_name).as_str(),
            vk::ObjectType::FENCE,
        );

        (*ctx)
            .device
            .begin_command_buffer(cmd_buf, &vk::CommandBufferBeginInfo::builder().build())?;

        Ok(CommandQueue {
            cmd_buf,
            fence: (*ctx).fences.insert(Fence::new(f)).unwrap(),
            dirty: true,
            ctx,
            pool: self,
            queue_type,
            is_secondary,
            ..Default::default()
        })
    }

    /// Recycle a command queue after GPU completion, returning its fence handle.
    pub fn recycle(&mut self, list: CommandQueue) -> Handle<Fence> {
        self.assert_owner();
        let buf = list.cmd_buf;
        if list.is_secondary {
            self.free_secondary.push(buf);
        } else {
            self.free_primary.push(buf);
        }
        list.fence
    }

    /// Reset a specific command queue for reuse.
    pub fn reset(&self, queue: &CommandQueue) -> Result<()> {
        self.assert_owner();
        unsafe {
            self.device.reset_command_buffer(queue.cmd_buf, vk::CommandBufferResetFlags::empty())?;
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
