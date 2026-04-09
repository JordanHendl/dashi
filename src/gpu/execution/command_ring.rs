use crate::{utils::Handle, CommandQueue, Context, Fence, QueueType, Result, SubmitInfo};
use std::ptr::NonNull;

pub struct CommandRing {
    cmds: Vec<CommandQueue>,
    fences: Vec<Option<Handle<Fence>>>,
    ctx: NonNull<Context>,
    curr: u16,
}

impl CommandRing {
    pub fn new(
        ctx: &mut Context,
        name: &str,
        frame_count: usize,
        queue_type: QueueType,
    ) -> Result<Self> {
        let mut cmds = Vec::new();
        for _i in 0..frame_count {
            let cmd = ctx.begin_command_queue(queue_type, name, false)?;
            cmds.push(cmd);
        }

        let fences: Vec<Option<Handle<Fence>>> = vec![None; frame_count];

        Ok(Self {
            cmds,
            fences,
            curr: 0,
            ctx: NonNull::from(ctx),
        })
    }

    /// Returns the index of the command buffer currently being recorded.
    pub fn current_index(&self) -> usize {
        self.curr as usize
    }

    pub fn append<T>(&mut self, mut record_func: T) -> Result<()>
    where
        T: FnMut(&mut CommandQueue),
    {
        if let Some(fence) = self.fences[self.curr as usize].as_mut() {
            unsafe {
                self.ctx.as_mut().wait_fence(fence.clone())?;
            }
            self.fences[self.curr as usize] = None;
        }

        record_func(&mut self.cmds[self.curr as usize]);
        Ok(())
    }

    pub fn record_enumerated<T>(&mut self, mut record_func: T) -> Result<()>
    where
        T: FnMut(&mut CommandQueue, u16),
    {
        if let Some(fence) = self.fences[self.curr as usize].as_mut() {
            unsafe {
                self.ctx.as_mut().wait_fence(fence.clone())?;
            }
            self.fences[self.curr as usize] = None;
        }

        self.cmds[self.curr as usize].reset()?;
        record_func(&mut self.cmds[self.curr as usize], self.curr);
        Ok(())
    }

    pub fn record<T>(&mut self, mut record_func: T) -> Result<()>
    where
        T: FnMut(&mut CommandQueue),
    {
        let trace = std::env::var_os("MESHI_TRACE_COMMAND_RING").is_some();
        let slot = self.curr as usize;
        let mut wait_ms = 0.0;
        if let Some(fence) = self.fences[self.curr as usize].as_mut() {
            let wait_start = std::time::Instant::now();
            unsafe {
                self.ctx.as_mut().wait_fence(fence.clone())?;
            }
            wait_ms = wait_start.elapsed().as_secs_f64() * 1000.0;
            self.fences[self.curr as usize] = None;
        }

        let reset_start = std::time::Instant::now();
        self.cmds[self.curr as usize].reset()?;
        let reset_ms = reset_start.elapsed().as_secs_f64() * 1000.0;
        let record_start = std::time::Instant::now();
        record_func(&mut self.cmds[self.curr as usize]);
        let record_ms = record_start.elapsed().as_secs_f64() * 1000.0;
        if trace {
            eprintln!(
                "[meshi-command-ring] slot={} wait_ms={:.3} reset_ms={:.3} record_ms={:.3}",
                slot, wait_ms, reset_ms, record_ms
            );
        }
        Ok(())
    }

    pub fn submit(&mut self, info: &SubmitInfo) -> Result<()> {
        self.submit_with_fence(info).map(|_| ())
    }

    pub fn submit_with_fence(&mut self, info: &SubmitInfo) -> Result<Handle<Fence>> {
        let fence = unsafe {
            self.ctx
                .as_mut()
                .submit_command_queue(&mut self.cmds[self.curr as usize], info)?
        };
        self.fences[self.curr as usize] = Some(fence);
        self.advance();
        Ok(fence)
    }

    /// Poll a single in-flight slot and clear its fence when the GPU work completes.
    pub fn poll_slot(&mut self, idx: usize) -> Result<bool> {
        if idx >= self.fences.len() {
            return Ok(false);
        }

        let Some(fence) = self.fences[idx].clone() else {
            return Ok(false);
        };

        let signaled = unsafe { self.ctx.as_mut().poll_fence(fence)? };
        if signaled {
            self.fences[idx] = None;
        }
        Ok(signaled)
    }

    fn advance(&mut self) {
        self.curr = (self.curr + 1) % self.cmds.len() as u16;
    }

    /// Waits on all in-flight fences and clears them.
    pub fn wait_all(&mut self) -> Result<()> {
        for slot in self.fences.iter_mut() {
            if let Some(fence) = slot.take() {
                unsafe {
                    self.ctx.as_mut().wait_fence(fence)?;
                }
            }
        }
        Ok(())
    }

    /// Record on every frame without advancing, running the closure on each CommandQueue
    pub fn record_all<T>(&mut self, mut record_func: T) -> Result<()>
    where
        T: FnMut(&mut CommandQueue),
    {
        let count = self.cmds.len();
        for idx in 0..count {
            if let Some(fence) = self.fences[idx].take() {
                unsafe {
                    self.ctx.as_mut().wait_fence(fence)?;
                }
            }
            self.cmds[idx].reset()?;
            record_func(&mut self.cmds[idx]);
        }
        Ok(())
    }

    /// Record on every frame with index, running the closure on each CommandQueue and its index
    pub fn record_all_enumerated<T>(&mut self, mut record_func: T) -> Result<()>
    where
        T: FnMut(&mut CommandQueue, u16),
    {
        let count = self.cmds.len() as u16;
        for idx in 0..count {
            let ui = idx as usize;
            if let Some(fence) = self.fences[ui].take() {
                unsafe {
                    self.ctx.as_mut().wait_fence(fence)?;
                }
            }
            self.cmds[ui].reset()?;
            record_func(&mut self.cmds[ui], idx);
        }
        Ok(())
    }

    /// Waits on all in-flight fences, frees all command lists, and clears internal state.
    fn destroy_all(&mut self) {
        // wait on any outstanding work
        let _ = self.wait_all();
        // free command buffers and fences
        unsafe {
            let ctx = self.ctx.as_mut();
            for cmd in self.cmds.drain(..) {
                ctx.destroy_command_queue(cmd);
            }
        }
    }
}

impl Drop for CommandRing {
    fn drop(&mut self) {
        // ensure all GPU work is finished and command lists freed
        self.destroy_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, ContextInfo, SubmitInfo};
    use serial_test::serial;

    /// Build a 3-frame CommandRing, do a few cycles of append+submit
    /// and make sure nothing panics.
    #[test]
    #[serial]
    fn basic_append_and_submit_cycles() {
        // create a headless context
        let mut ctx = Context::headless(&ContextInfo::default()).expect("headless context");
        // 3 frames
        let mut fcl = CommandRing::new(&mut ctx, "basic", 3, QueueType::Graphics).unwrap();

        // do 10 cycles of append + submit
        for _ in 0..10 {
            fcl.append(|cmd| {
                // we can even record a no-op dispatch or whatever
                // here we just begin/end an empty command list
                let _ = cmd;
            })
            .unwrap();

            // no semaphores or waits
            let submit_info = SubmitInfo::default();
            fcl.submit(&submit_info).unwrap();
        }
        fcl.wait_all().unwrap();
        // cleanup
        drop(fcl);
        ctx.destroy();
    }

    /// Test that record() resets the CommandQueue and that record_enumerated()
    /// passes the current frame index correctly into the closure.
    #[test]
    #[serial]
    fn record_and_record_enumerated_indices() {
        let mut ctx = Context::headless(&ContextInfo::default()).expect("headless context");
        // choose 3 frames so we see at least two distinct indices
        let mut fcl = CommandRing::new(&mut ctx, "enum", 3, QueueType::Graphics).unwrap();

        let mut seen_plain = Vec::new();
        let mut seen_enum = Vec::new();

        // record() should always invoke our closure with no index
        fcl.record(|_cmd| {
            seen_plain.push(());
        })
        .unwrap();
        fcl.submit(&SubmitInfo::default()).unwrap();

        fcl.record(|_cmd| {
            seen_plain.push(());
        })
        .unwrap();
        fcl.submit(&SubmitInfo::default()).unwrap();
        fcl.wait_all().unwrap();

        assert_eq!(seen_plain.len(), 2);

        // record_enumerated() gives us the current frame index before we submit
        fcl.record_enumerated(|_cmd, idx| {
            seen_enum.push(idx);
        })
        .unwrap();
        fcl.submit(&SubmitInfo::default()).unwrap();
        fcl.wait_all().unwrap();

        fcl.record_enumerated(|_cmd, idx| {
            seen_enum.push(idx);
        })
        .unwrap();
        fcl.submit(&SubmitInfo::default()).unwrap();
        fcl.wait_all().unwrap();
        // with 3 frames and the current advance logic, we should have seen [2, 0]
        assert_eq!(seen_enum, vec![2, 0]);

        drop(fcl);
        ctx.destroy();
    }
}
