//! subpass_dispatcher_rayon.rs
//! Fixed-size Rayon thread pool for parallel subpass rendering.
//
// Add to Cargo.toml:
// [dependencies]
// rayon = "1.10"

use rayon::{prelude::*, ThreadPool, ThreadPoolBuilder};
use std::{
    ptr::{self, NonNull},
    sync::Arc,
};

use crate::{
    cmd::{Executable, Graphics}, driver::command::BeginDrawing, CommandQueueInfo2, CommandRing, CommandStream, Context, Handle, QueueType, RenderPass, SubmitInfo, SubmitInfo2
};

pub struct RendererDispatch<'a> {
    pub draw: BeginDrawing,
    pub submit: SubmitInfo<'a>,
}

/// Context handed to each render callback.
#[derive(Clone, Copy)]
pub struct SubpassCtx {
    pub render_pass: Handle<RenderPass>,
    pub subpass_index: u32,
    pub thread_index: usize,
    pub thread_count: usize,
}

/// Signature for subpass render callbacks.
/// Return a `CommandStream` for this (subpass, thread) slice.
pub type RenderCallback =
    Arc<dyn Fn(SubpassCtx) -> CommandStream<Graphics> + Send + Sync + 'static>;

/// Holds the render pass, registry of callbacks per subpass, and a fixed-size Rayon pool.
pub struct Renderer {
    ctx: NonNull<Context>,
    render_pass: Handle<RenderPass>,
    ring: CommandRing,
    subpasses: Vec<Vec<RenderCallback>>,
    pool: Arc<ThreadPool>,
    worker_threads: usize,
}

impl Renderer {
    /// Create with a fixed-size Rayon pool.
    pub fn new_with_pool_size(
        ctx: &mut Context,
        render_pass: Handle<RenderPass>,
        subpass_count: u32,
        worker_threads: usize,
    ) -> Self {
        assert!(worker_threads > 0, "worker_threads must be > 0");
        let pool = ThreadPoolBuilder::new()
            .num_threads(worker_threads)
            .build()
            .expect("failed to build Rayon thread pool");

        let mut subpasses = Vec::with_capacity(subpass_count as usize);
        subpasses.resize_with(subpass_count as usize, Vec::new);

        Self {
            render_pass,
            ring: ctx
                .make_command_ring(&CommandQueueInfo2 {
                    debug_name: "dashi-renderer-ring",
                    parent: None,
                    queue_type: QueueType::Graphics,
                })
                .unwrap(),
            subpasses,
            pool: Arc::new(pool),
            worker_threads,
            ctx: NonNull::new(ctx as *mut Context).unwrap(),
        }
    }

    /// Create using an existing Rayon pool.
    pub fn new_with_pool(
        ctx: &mut Context,
        render_pass: Handle<RenderPass>,
        subpass_count: u32,
        pool: Arc<ThreadPool>,
        worker_threads: usize,
    ) -> Self {
        let mut subpasses = Vec::with_capacity(subpass_count as usize);
        subpasses.resize_with(subpass_count as usize, Vec::new);
        Self {
            ctx: NonNull::new(ctx as *mut Context).unwrap(),
            ring: ctx
                .make_command_ring(&CommandQueueInfo2 {
                    debug_name: "dashi-renderer-ring",
                    parent: None,
                    queue_type: QueueType::Graphics,
                })
                .unwrap(),

            render_pass,
            subpasses,
            pool,
            worker_threads: worker_threads.max(1),
        }
    }

    /// Register a callback for a specific subpass (multiple allowed).
    pub fn on_subpass(&mut self, subpass_index: u32, cb: RenderCallback) {
        let i = subpass_index as usize;
        assert!(i < self.subpasses.len(), "subpass_index out of range");
        self.subpasses[i].push(cb);
    }

    /// Register the same callback over [start, end) subpasses.
    pub fn on_subpass_range_arc(&mut self, start: u32, end_exclusive: u32, cb: RenderCallback) {
        assert!(start <= end_exclusive);
        for sp in start..end_exclusive {
            self.on_subpass(sp, cb.clone());
        }
    }

    /// Run all registered callbacks for each subpass in parallel across the fixed pool.
    ///
    /// Returns one `CommandStream` per worker thread (index = thread_index).
    pub fn dispatch(&mut self, info: &RendererDispatch) {
        let render_pass = self.render_pass;
        let subpasses = &self.subpasses;
        let threads = self.worker_threads;

        self.ring.record(|cmd| {
            let mut stream = CommandStream::new().begin();
            let mut draw = stream.begin_drawing(&info.draw);
            let s = self.pool.install(|| {
                let streams = (0..threads)
                    .into_par_iter()
                    .map(|thread_index| {
                        let mut stream = CommandStream::<Graphics>::new_gfx();
                        for (sp_idx, callbacks) in subpasses.iter().enumerate() {
                            if callbacks.is_empty() {
                                continue;
                            }
                            let ctx = SubpassCtx {
                                render_pass,
                                subpass_index: sp_idx as u32,
                                thread_index,
                                thread_count: threads,
                            };
                            for cb in callbacks {
                                let mut s = (cb)(ctx);
                                stream.append(s);
                            }
                        }
                        return stream;
                    })
                    .collect::<Vec<_>>();

                return streams;
            });

            for c in s {
                draw.append(c);
            }

            // End drawing.
            stream = draw.stop_drawing();
            stream.end().append(cmd);
        });
        // Submit our recorded commands
        self.ring.submit(&info.submit)
        .unwrap();
    }
}

///// Main-thread helper: record each per-thread stream into a secondary CB.
/////
///// Pass a closure that takes a mutable stream and records it into a secondary CB
///// with the inheritance you need (render pass, framebuffer, subpass, etc.).
//pub fn fold_to_secondary_per_thread<F>(
//    mut record_secondary: F,
//    mut per_thread_streams: Vec<CommandStream>,
//) -> Vec<CommandStream>
//where
//    F: FnMut(&mut CommandStream) -> (),
//{
//    for s in per_thread_streams.iter_mut() {
//        record_secondary(s);
//    }
//    per_thread_streams
//}
//
//// --- Minimal glue so this compiles against your existing API shape -----------
//// If your `CommandStream` already has an extend/append method, delete this trait
//// and call your real method at the two call sites above.
//
//trait CommandStreamExt {
//    fn append_from(&mut self, other: &mut CommandStream);
//}
//
//impl CommandStreamExt for CommandStream {
//    fn append_from(&mut self, _other: &mut CommandStream) {
//        // Replace with your efficient splice/extend:
//        // e.g., self.extend(other.drain_all());
//    }
//}
