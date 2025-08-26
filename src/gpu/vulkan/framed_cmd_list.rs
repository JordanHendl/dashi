use crate::{utils::Handle, CommandList, CommandListInfo, Context, Fence, SubmitInfo};

pub struct FramedCommandList {
    cmds: Vec<CommandList>,
    fences: Vec<Option<Handle<Fence>>>,
    ctx: *mut Context,
    curr: u16,
}

impl FramedCommandList {
    pub fn new(ctx: &mut Context, name: &str, frame_count: usize) -> Self {
        let mut cmds = Vec::new();
        for _i in 0..frame_count {
            cmds.push(
                ctx.begin_command_list(&CommandListInfo {
                    debug_name: name,
                    should_cleanup: false,
                })
                .unwrap(),
            );
        }

        let fences: Vec<Option<Handle<Fence>>> = vec![None; frame_count];

        Self {
            cmds,
            fences,
            curr: 0,
            ctx,
        }
    }

    pub fn append<T>(&mut self, mut record_func: T)
    where
        T: FnMut(&mut CommandList),
    {
        if let Some(fence) = self.fences[self.curr as usize].as_mut() {
            unsafe { (*self.ctx).wait(fence.clone()).unwrap() };
            self.fences[self.curr as usize] = None;
        }

        record_func(&mut self.cmds[self.curr as usize]);
    }

    pub fn record_enumerated<T>(&mut self, mut record_func: T)
    where
        T: FnMut(&mut CommandList, u16),
    {
        if let Some(fence) = self.fences[self.curr as usize].as_mut() {
            unsafe { (*self.ctx).wait(fence.clone()).unwrap() };
            self.fences[self.curr as usize] = None;
        }

        self.cmds[self.curr as usize].reset().unwrap();
        record_func(&mut self.cmds[self.curr as usize], self.curr);
    }

    pub fn record<T>(&mut self, mut record_func: T)
    where
        T: FnMut(&mut CommandList),
    {
        if let Some(fence) = self.fences[self.curr as usize].as_mut() {
            unsafe { (*self.ctx).wait(fence.clone()).unwrap() };
            self.fences[self.curr as usize] = None;
        }

        self.cmds[self.curr as usize].reset().unwrap();
        record_func(&mut self.cmds[self.curr as usize]);
    }

    pub fn submit(&mut self, info: &SubmitInfo) {
        self.fences[self.curr as usize] = Some(unsafe {
            (*self.ctx)
                .submit(&mut self.cmds[self.curr as usize], info)
                .unwrap()
        });

        self.advance();
    }

    fn advance(&mut self) {
        self.curr = (self.curr + 1) % (self.cmds.len() - 1) as u16;
    }

    /// Waits on all in-flight fences and clears them.
    pub fn wait_all(&mut self) {
        for slot in self.fences.iter_mut() {
            if let Some(fence) = slot.take() {
                unsafe {
                    (*self.ctx).wait(fence).unwrap();
                }
            }
        }
    }

    /// Record on every frame without advancing, running the closure on each CommandList
    pub fn record_all<T>(&mut self, mut record_func: T)
    where
        T: FnMut(&mut CommandList),
    {
        let count = self.cmds.len();
        for idx in 0..count {
            if let Some(fence) = self.fences[idx].take() {
                unsafe {
                    (*self.ctx).wait(fence).unwrap();
                }
            }
            self.cmds[idx].reset().unwrap();
            record_func(&mut self.cmds[idx]);
        }
    }

    /// Record on every frame with index, running the closure on each CommandList and its index
    pub fn record_all_enumerated<T>(&mut self, mut record_func: T)
    where
        T: FnMut(&mut CommandList, u16),
    {
        let count = self.cmds.len() as u16;
        for idx in 0..count {
            let ui = idx as usize;
            if let Some(fence) = self.fences[ui].take() {
                unsafe {
                    (*self.ctx).wait(fence).unwrap();
                }
            }
            self.cmds[ui].reset().unwrap();
            record_func(&mut self.cmds[ui], idx);
        }
    }

    /// Waits on all in-flight fences, frees all command lists, and clears internal state.
    fn destroy_all(&mut self) {
        // wait on any outstanding work
        self.wait_all();
        // free command buffers and fences
        unsafe {
            for cmd in self.cmds.drain(..) {
                (*self.ctx).destroy_cmd_list(cmd);
            }
        }
    }
}

impl Drop for FramedCommandList {
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

    /// Build a 3-frame FramedCommandList, do a few cycles of append+submit
    /// and make sure nothing panics.
    #[test]
    #[serial]
    fn basic_append_and_submit_cycles() {
        // create a headless context
        let mut ctx = Context::headless(&ContextInfo::default()).expect("headless context");
        // 3 frames
        let mut fcl = FramedCommandList::new(&mut ctx, "basic", 3);

        // do 10 cycles of append + submit
        for _ in 0..10 {
            fcl.append(|cmd| {
                // we can even record a no-op dispatch or whatever
                // here we just begin/end an empty command list
                let _ = cmd;
            });

            // no semaphores or waits
            let submit_info = SubmitInfo::default();
            fcl.submit(&submit_info);
        }
        fcl.wait_all();
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
        let mut fcl = FramedCommandList::new(&mut ctx, "enum", 3);

        let mut seen_plain = Vec::new();
        let mut seen_enum = Vec::new();

        // record() should always invoke our closure with no index
        fcl.record(|_cmd| {
            seen_plain.push(());
        });
        fcl.submit(&SubmitInfo::default());

        fcl.record(|_cmd| {
            seen_plain.push(());
        });
        fcl.submit(&SubmitInfo::default());
        fcl.wait_all();

        assert_eq!(seen_plain.len(), 2);

        // record_enumerated() gives us the current frame index before we submit
        fcl.record_enumerated(|_cmd, idx| {
            seen_enum.push(idx);
        });
        fcl.submit(&SubmitInfo::default());
        fcl.wait_all();

        fcl.record_enumerated(|_cmd, idx| {
            seen_enum.push(idx);
        });
        fcl.submit(&SubmitInfo::default());
        fcl.wait_all();
        // with 3 frames and the current advance logic, we should have seen [0, 1]
        assert_eq!(seen_enum, vec![0, 1]);

        drop(fcl);
        ctx.destroy();
    }
}
