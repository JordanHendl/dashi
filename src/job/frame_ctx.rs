use std::ptr::NonNull;

use crate::{
    gpu::vulkan::{CommandQueue, Context, QueueType},
    utils::Handle,
    Fence,
    Result,
};

use super::{collector::{MergedPrimaries, PrimaryCmd}, Job, ThreadCtx};

/// Per-frame context holding jobs and GPU resources.
pub struct FrameCtx {
    /// Jobs queued for execution on this frame.
    pub jobs: Vec<Job>,
    /// Primary command buffer for this frame.
    pub primary: CommandQueue,
    /// Timeline value associated with the submitted work.
    pub timeline: u64,
    /// Completed thread contexts awaiting reset.
    pub completed: Vec<ThreadCtx>,
    /// Primaries recorded by all jobs in this frame.
    pub primaries: Vec<PrimaryCmd>,
    /// Merged primaries sorted by pass/bucket.
    pub merged: Vec<MergedPrimaries>,
    /// Fence signaled when GPU work for this frame completes.
    pub fence: Option<Handle<Fence>>,
}

impl FrameCtx {
    /// Create an empty frame context.
    pub fn new() -> Self {
        Self {
            jobs: Vec::new(),
            primary: CommandQueue::default(),
            timeline: 0,
            completed: Vec::new(),
            primaries: Vec::new(),
            merged: Vec::new(),
            fence: None,
        }
    }

    /// Record a finished thread context for later recycling.
    pub fn push_completed(&mut self, ctx: ThreadCtx) {
        self.completed.push(ctx);
    }
}

/// Ring allocator for `FrameCtx` instances.
pub struct FrameRing {
    frames: Vec<FrameCtx>,
    curr: usize,
    ctx: NonNull<Context>,
}

impl FrameRing {
    /// Create a new ring with the requested number of frames.
    pub fn new(
        ctx: &mut Context,
        frame_count: usize,
        queue_type: QueueType,
        debug_name: &str,
    ) -> Result<Self> {
        let mut frames = Vec::new();
        let raw_ctx = ctx as *mut Context;
        let ctx_ptr = NonNull::new(raw_ctx).unwrap();
        for i in 0..frame_count {
            let primary = ctx
                .pool_mut(queue_type)
                .begin(raw_ctx, &format!("{debug_name}.{i}"), false)?;
            frames.push(FrameCtx {
                jobs: Vec::new(),
                primary,
                timeline: 0,
                completed: Vec::new(),
                primaries: Vec::new(),
                merged: Vec::new(),
                fence: None,
            });
        }
        Ok(Self {
            frames,
            curr: 0,
            ctx: ctx_ptr,
        })
    }

    /// Acquire the next frame, resetting resources once GPU work has completed.
    pub fn acquire(&mut self, timeline: u64) -> Result<&mut FrameCtx> {
        let idx = self.curr;
        let len = self.frames.len();
        let frame = &mut self.frames[idx];

        if let Some(fence) = frame.fence.take() {
            unsafe {
                self.ctx.as_mut().wait(fence)?;
            }
            frame.primary.reset()?;
            frame.completed.clear();
            frame.jobs.clear();
            frame.primaries.clear();
            frame.merged.clear();
        }

        frame.timeline = timeline;
        self.curr = (self.curr + 1) % len;
        Ok(frame)
    }
}

impl Drop for FrameRing {
    fn drop(&mut self) {
        unsafe {
            let ctx = self.ctx.as_mut();
            for mut frame in self.frames.drain(..) {
                if let Some(fence) = frame.fence.take() {
                    let _ = ctx.wait(fence);
                }
                ctx.destroy_cmd_queue(frame.primary);
                for m in frame.merged.drain(..) {
                    for q in m.queues {
                        ctx.destroy_cmd_queue(q);
                    }
                }
                for p in frame.primaries.drain(..) {
                    ctx.destroy_cmd_queue(p.queue);
                }
            }
        }
    }
}

