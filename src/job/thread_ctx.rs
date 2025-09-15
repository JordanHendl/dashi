use std::thread::ThreadId;

use crate::gpu::vulkan::{CommandPool, CommandQueue, Context};
use crate::Result;

/// Per-thread context for job execution.
///
/// Holds a Vulkan command pool and reusable scratch allocations owned by the
/// thread that created it. The context can be used by jobs to acquire
/// secondary command buffers which are safe to reset and recycle.
pub struct ThreadCtx {
    /// Command pool used to allocate command buffers.
    pool: Option<CommandPool>,
    /// Identifier of the owning thread.
    thread: ThreadId,
    /// Scratch allocations that can be reused between jobs.
    scratch: Vec<Vec<u8>>,
}

impl ThreadCtx {
    /// Create a new thread context with the provided command pool.
    pub fn new(pool: CommandPool) -> Self {
        Self {
            pool: Some(pool),
            thread: std::thread::current().id(),
            scratch: Vec::new(),
        }
    }

    fn assert_owner(&self) {
        debug_assert_eq!(
            self.thread,
            std::thread::current().id(),
            "ThreadCtx used from wrong thread",
        );
    }

    /// Acquire a secondary command buffer from the internal pool.
    pub fn acquire_secondary(
        &mut self,
        ctx: *mut Context,
        debug_name: &str,
    ) -> Result<CommandQueue> {
        self.assert_owner();
        self
            .pool
            .as_mut()
            .expect("CommandPool not initialized")
            .begin(ctx, debug_name, true)
    }

    /// Reset a previously acquired command queue for reuse.
    pub fn reset(&mut self, queue: &mut CommandQueue) -> Result<()> {
        self.assert_owner();
        self
            .pool
            .as_mut()
            .expect("CommandPool not initialized")
            .reset(queue)
    }
}

impl Default for ThreadCtx {
    fn default() -> Self {
        Self {
            pool: None,
            thread: std::thread::current().id(),
            scratch: Vec::new(),
        }
    }
}

