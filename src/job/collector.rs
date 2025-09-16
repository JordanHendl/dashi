use std::thread::ThreadId;
use std::{collections::hash_map::DefaultHasher, hash::{Hash, Hasher}};

use crate::gpu::vulkan::CommandQueue;

use super::FrameCtx;

/// Metadata for a recorded primary command queue.
#[derive(Clone)]
pub struct PrimaryCmd {
    pub pass: u16,
    pub bucket: u16,
    pub order: u32,
    pub thread: ThreadId,
    pub queue: CommandQueue,
}

impl PrimaryCmd {
    fn key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.thread.hash(&mut hasher);
        let thread_bits = hasher.finish() & 0xFF;
        ((self.pass as u64) << 48)
            | ((self.bucket as u64) << 32)
            | ((self.order as u64) << 8)
            | thread_bits
    }
}

/// A merged set of primaries for a given bucket/pass pair.
#[derive(Default)]
pub struct MergedPrimaries {
    pub pass: u16,
    pub bucket: u16,
    pub queues: Vec<CommandQueue>,
}

/// Collects recorded primaries from a frame, sorting and grouping them.
///
/// The sort key is constructed as `(pass<<48)|(bucket<<32)|(order<<8)|(thread_id&0xFF)`.
/// Primaries are grouped by `(pass, bucket)` while preserving the relative ordering
/// of `(order, thread_id)` within each group. The grouped primaries are written
/// to `frame.merged` for later submission.
pub fn collect(frame: &mut FrameCtx) {
    // Stable sort by the computed key.
    frame.primaries.sort_by_key(|p| p.key());

    // Group by pass/bucket.
    let mut merged: Vec<MergedPrimaries> = Vec::new();
    for prim in frame.primaries.drain(..) {
        if let Some(last) = merged.last_mut() {
            if last.pass == prim.pass && last.bucket == prim.bucket {
                last.queues.push(prim.queue);
                continue;
            }
        }
        merged.push(MergedPrimaries {
            pass: prim.pass,
            bucket: prim.bucket,
            queues: vec![prim.queue],
        });
    }
    frame.merged = merged;
}
