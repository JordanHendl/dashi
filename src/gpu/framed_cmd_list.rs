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

    pub fn record<T>(&mut self, mut record_func: T)
    where
        T: FnMut(&mut CommandList),
    {
        if let Some(fence) = self.fences[self.curr as usize].as_mut() {
            unsafe { (*self.ctx).wait(fence.clone()).unwrap() };
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
        self.curr = (self.curr + 1) % self.cmds.len() as u16;
    }
}
