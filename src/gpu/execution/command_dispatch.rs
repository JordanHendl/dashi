use crate::{
    cmd::{CommandStream, Executable},
    CommandQueue,
    Context,
    Fence,
    GPUError,
    Result,
    SubmitInfo2,
};
use std::{
    ptr::NonNull,
    sync::{Mutex, OnceLock},
    time::{Duration, Instant},
};

pub struct CommandDispatchConfig {
    pub debug_name: String,
    pub wait_delay: Duration,
}

impl Default for CommandDispatchConfig {
    fn default() -> Self {
        Self {
            debug_name: "command_dispatch".to_string(),
            wait_delay: Duration::from_millis(0),
        }
    }
}

struct PendingDispatch {
    queue: CommandQueue,
    fence: crate::Handle<Fence>,
    ready_at: Instant,
}

struct CommandDispatchBackend {
    ctx: NonNull<Context>,
    debug_name: String,
    wait_delay: Duration,
    pending: Vec<PendingDispatch>,
}

unsafe impl Send for CommandDispatchBackend {}

impl CommandDispatchBackend {
    fn new(ctx: &mut Context, config: CommandDispatchConfig) -> Self {
        Self {
            ctx: NonNull::from(ctx),
            debug_name: config.debug_name,
            wait_delay: config.wait_delay,
            pending: Vec::new(),
        }
    }

    fn dispatch(&mut self, stream: CommandStream<Executable>, submit: &SubmitInfo2) -> Result<()> {
        let queue_type = stream.queue_type();
        let ctx_ptr = unsafe { self.ctx.as_mut().vulkan_mut_ptr() };
        let ctx_ref = unsafe { &mut *ctx_ptr };
        let mut queue = ctx_ref
            .pool_mut(queue_type)
            .begin(ctx_ptr, &self.debug_name, false)?;
        if !submit
            .wait_sems
            .iter()
            .chain(submit.signal_sems.iter())
            .all(|sem| sem.valid() || sem == &crate::Handle::default())
        {
            return Err(GPUError::Unimplemented(
                "CommandDispatch received invalid semaphore handle",
            ));
        }
        let (_pending, fence) = stream.submit(&mut queue, submit);
        if let Some(fence) = fence {
            self.pending.push(PendingDispatch {
                queue,
                fence,
                ready_at: Instant::now() + self.wait_delay,
            });
            let _ = self.tick();
            return Ok(());
        }
        ctx_ref.destroy_cmd_queue(queue);
        Ok(())
    }

    fn tick(&mut self) -> Result<usize> {
        let ctx = unsafe { self.ctx.as_mut().vulkan_mut_ptr() };
        let ctx_ref = unsafe { &mut *ctx };
        let now = Instant::now();
        let mut completed = 0;
        let mut idx = 0;
        while idx < self.pending.len() {
            if self.pending[idx].ready_at > now {
                idx += 1;
                continue;
            }
            let pending = self.pending.swap_remove(idx);
            ctx_ref.wait(pending.fence)?;
            ctx_ref.destroy_cmd_queue(pending.queue);
            completed += 1;
        }
        Ok(completed)
    }
}

static COMMAND_DISPATCH: OnceLock<Mutex<CommandDispatchBackend>> = OnceLock::new();

pub struct CommandDispatch;

impl CommandDispatch {
    /// Initialize the global dispatch backend.
    pub fn init(ctx: &mut Context) -> Result<()> {
        Self::init_with_config(ctx, CommandDispatchConfig::default())
    }

    /// Initialize the global dispatch backend with a custom debug name.
    pub fn init_with_name(ctx: &mut Context, debug_name: impl Into<String>) -> Result<()> {
        Self::init_with_config(
            ctx,
            CommandDispatchConfig {
                debug_name: debug_name.into(),
                ..CommandDispatchConfig::default()
            },
        )
    }

    /// Initialize the global dispatch backend with configuration.
    pub fn init_with_config(ctx: &mut Context, config: CommandDispatchConfig) -> Result<()> {
        COMMAND_DISPATCH
            .set(Mutex::new(CommandDispatchBackend::new(ctx, config)))
            .map_err(|_| GPUError::Unimplemented("CommandDispatch backend already initialized"))?;
        Ok(())
    }

    /// Dispatch a command stream using the global backend.
    pub fn dispatch(stream: CommandStream<Executable>, submit: &SubmitInfo2) -> Result<()> {
        let backend = COMMAND_DISPATCH
            .get()
            .ok_or(GPUError::Unimplemented("CommandDispatch backend not initialized"))?;
        let mut backend = backend.lock().expect("CommandDispatch backend lock poisoned");
        backend.dispatch(stream, submit)
    }

    /// Process completed submissions and release command queues.
    pub fn tick() -> Result<usize> {
        let backend = COMMAND_DISPATCH
            .get()
            .ok_or(GPUError::Unimplemented("CommandDispatch backend not initialized"))?;
        let mut backend = backend.lock().expect("CommandDispatch backend lock poisoned");
        backend.tick()
    }
}
