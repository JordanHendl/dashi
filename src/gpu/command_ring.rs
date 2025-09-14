use crate::{
    utils::Handle,
    CommandList,
    CommandListInfo,
    Context,
    Fence,
    SubmitInfo,
    Result,
};
use std::ptr::NonNull;

pub struct CommandRing {
    cmds: Vec<CommandList>,
    fences: Vec<Option<Handle<Fence>>>,
    ctx: NonNull<Context>,
    curr: u16,
}

impl CommandRing {
    pub fn new(
        ctx: &mut Context,
        name: &str,
        frame_count: usize,
    ) -> Result<Self> {
        let mut cmds = Vec::new();
        for _i in 0..frame_count {
            cmds.push(ctx.begin_command_list(&CommandListInfo {
                debug_name: name,
                should_cleanup: false,
            })?);
        }

        let fences: Vec<Option<Handle<Fence>>> = vec![None; frame_count];

        Ok(Self {
            cmds,
            fences,
            curr: 0,
            ctx: NonNull::from(ctx),
        })
    }

    pub fn append<T>(&mut self, mut record_func: T) -> Result<()>
    where
        T: FnMut(&mut CommandList),
    {
        if let Some(fence) = self.fences[self.curr as usize].as_mut() {
            unsafe {
                self.ctx.as_mut().wait(fence.clone())?;
            }
            self.fences[self.curr as usize] = None;
        }

        record_func(&mut self.cmds[self.curr as usize]);
        Ok(())
    }

    pub fn record_enumerated<T>(&mut self, mut record_func: T) -> Result<()>
    where
        T: FnMut(&mut CommandList, u16),
    {
        if let Some(fence) = self.fences[self.curr as usize].as_mut() {
            unsafe {
                self.ctx.as_mut().wait(fence.clone())?;
            }
            self.fences[self.curr as usize] = None;
        }

        self.cmds[self.curr as usize].reset()?;
        record_func(&mut self.cmds[self.curr as usize], self.curr);
        Ok(())
    }

    pub fn record<T>(&mut self, mut record_func: T) -> Result<()>
    where
        T: FnMut(&mut CommandList),
    {
        if let Some(fence) = self.fences[self.curr as usize].as_mut() {
            unsafe {
                self.ctx.as_mut().wait(fence.clone())?;
            }
            self.fences[self.curr as usize] = None;
        }

        self.cmds[self.curr as usize].reset()?;
        record_func(&mut self.cmds[self.curr as usize]);
        Ok(())
    }

    pub fn submit(&mut self, info: &SubmitInfo) -> Result<()> {
        self.fences[self.curr as usize] = Some(unsafe {
            self.ctx
                .as_mut()
                .submit(&mut self.cmds[self.curr as usize], info)?
        });

        self.advance();
        Ok(())
    }

    fn advance(&mut self) {
        self.curr = (self.curr + 1) % self.cmds.len() as u16;
    }

    /// Waits on all in-flight fences and clears them.
    pub fn wait_all(&mut self) -> Result<()> {
        for slot in self.fences.iter_mut() {
            if let Some(fence) = slot.take() {
                unsafe {
                    self.ctx.as_mut().wait(fence)?;
                }
            }
        }
        Ok(())
    }

    /// Record on every frame without advancing, running the closure on each CommandList
    pub fn record_all<T>(&mut self, mut record_func: T) -> Result<()>
    where
        T: FnMut(&mut CommandList),
    {
        let count = self.cmds.len();
        for idx in 0..count {
            if let Some(fence) = self.fences[idx].take() {
                unsafe {
                    self.ctx.as_mut().wait(fence)?;
                }
            }
            self.cmds[idx].reset()?;
            record_func(&mut self.cmds[idx]);
        }
        Ok(())
    }

    /// Record on every frame with index, running the closure on each CommandList and its index
    pub fn record_all_enumerated<T>(&mut self, mut record_func: T) -> Result<()>
    where
        T: FnMut(&mut CommandList, u16),
    {
        let count = self.cmds.len() as u16;
        for idx in 0..count {
            let ui = idx as usize;
            if let Some(fence) = self.fences[ui].take() {
                unsafe {
                    self.ctx.as_mut().wait(fence)?;
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
                ctx.destroy_cmd_list(cmd);
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
        let mut fcl = CommandRing::new(&mut ctx, "basic", 3).unwrap();

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

    /// Test that record() resets the CommandList and that record_enumerated()
    /// passes the current frame index correctly into the closure.
    #[test]
    #[serial]
    fn record_and_record_enumerated_indices() {
        let mut ctx = Context::headless(&ContextInfo::default()).expect("headless context");
        // choose 3 frames so we see at least two distinct indices
        let mut fcl = CommandRing::new(&mut ctx, "enum", 3).unwrap();

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
