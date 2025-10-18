use crate::{utils::Handle, CommandQueue, Context, Fence, QueueType, Result, SubmitInfo};
use std::ptr::NonNull;

use super::{cmd::Graphics, CommandStream, RenderPass};

pub struct SubpassContext {
    subpass_id: u16,
    thread_id: u16,
}

pub struct RendererInfo<'a> {
    pub debug_name: &'a str,
    pub render_pass: Handle<RenderPass>,
    pub num_subpasses: u16,
    pub num_threads: u16,
}

pub struct Renderer {
    ctx: NonNull<Context>,
    rp: Handle<RenderPass>,
    subpass_fns: Vec<
        Vec<Box<dyn FnMut(SubpassContext, CommandStream<Graphics>) -> CommandStream<Graphics>>>,
    >,
}

impl Renderer {
    pub fn new(ctx: &mut Context, info: &RendererInfo) -> Result<Self> {
        let mut s = Self {
            ctx: NonNull::new(ctx as *mut Context).unwrap(),
            rp: info.render_pass,
            subpass_fns: Vec::new(),
        };

        for it in 0..info.num_subpasses {
            s.subpass_fns.push(Vec::new());
        }

        Ok(s)
    }

    pub fn add_render_function<F>(&mut self, f: F, subpass_id: u16)
    where
        F: FnMut(SubpassContext, CommandStream<Graphics>) -> CommandStream<Graphics> + 'static,
    {
        self.subpass_fns[subpass_id as usize].push(Box::new(f));
    }

    pub fn add_render_function_to_all_subpasses<F>(&mut self, f: F)
    where
        F: FnMut(SubpassContext, CommandStream<Graphics>) -> CommandStream<Graphics> + 'static,
    {
        todo!()
    }

    pub fn dispatch(&mut self) {

    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Context, ContextInfo, SubmitInfo};
    use serial_test::serial;
}
