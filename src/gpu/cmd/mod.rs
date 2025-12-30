use std::marker::PhantomData;

use crate::gpu::driver::command::{
    BeginDrawing, BeginRenderPass, BlitImage, Dispatch, Draw, DrawIndexed,
    GraphicsPipelineStateUpdate, MSImageResolve,
};
use crate::gpu::driver::command::{
    CommandEncoder, CommandSink, CopyBuffer, CopyBufferImage, CopyImageBuffer,
};
use crate::gpu::driver::types::Handle;
use crate::{
    Buffer, Fence, GraphicsPipeline, Image, QueueType, ResourceUse, Result, SubmitInfo2, UsageBits,
    Viewport,
};

/// Generic command buffer with type-state tracking.
pub struct CommandStream<S> {
    enc: CommandEncoder,
    _state: PhantomData<S>,
}

pub struct Initial;
pub struct Recording;
pub struct Executable;
pub struct Pending;
pub struct Graphics;
pub struct PendingGraphics;
pub struct Compute;

impl<T> CommandStream<T> {
    pub fn combine<G>(mut self, sink: CommandStream<G>) -> Self {
        self.enc.combine(&sink.enc);
        self
    }

    /// Returns the queue type this command stream targets.
    pub fn queue_type(&self) -> QueueType {
        self.enc.queue_type()
    }
}

impl CommandStream<Initial> {
    /// Create a new command buffer in the initial state.
    pub fn new() -> Self {
        Self::new_with_queue(QueueType::Graphics)
    }

    /// Create a new command buffer targeting a specific queue type.
    pub fn new_with_queue(queue: QueueType) -> Self {
        Self {
            enc: CommandEncoder::new(queue),
            _state: PhantomData,
        }
    }

    /// Begin recording commands.
    pub fn begin(self) -> CommandStream<Recording> {
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }
}

impl CommandStream<Recording> {
    /// Finish recording and transition to an executable state.
    pub fn end(self) -> CommandStream<Executable> {
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }

    /// Explicitly transition a buffer to a desired usage when automatic inference
    /// is insufficient (for example, after staging uploads or when handing the
    /// buffer across queues).
    pub fn prepare_buffer(mut self, buffer: Handle<Buffer>, usage: UsageBits) -> Self {
        self.enc.prepare_buffer(buffer, usage, None);
        self
    }

    pub fn prepare_buffer_for(&mut self, buffer: Handle<Buffer>, usage: ResourceUse) {
        self.enc.prepare_buffer_for(buffer, usage);
    }

    pub fn prepare_image_for(&mut self, image: Handle<Image>, usage: ResourceUse) {
        self.enc.prepare_image_for(image, usage);
    }

    /// Specify the queue that should own the buffer after the transition. This
    /// is useful for queue family ownership transfers.
    pub fn prepare_buffer_for_queue(
        mut self,
        buffer: Handle<Buffer>,
        usage: UsageBits,
        queue: QueueType,
    ) -> Self {
        self.enc.prepare_buffer(buffer, usage, Some(queue));
        self
    }

    pub fn copy_buffers(mut self, cmd: &CopyBuffer) -> Self {
        self.enc.copy_buffer(cmd);
        self
    }

    pub fn copy_buffer_to_image(mut self, cmd: &CopyBufferImage) -> Self {
        self.enc.copy_buffer_to_image(cmd);
        self
    }

    pub fn copy_image_to_buffer(mut self, cmd: &CopyImageBuffer) -> Self {
        self.enc.copy_image_to_buffer(cmd);
        self
    }

    pub fn blit_images(mut self, cmd: &BlitImage) -> Self {
        self.enc.blit_image(cmd);
        self
    }

    pub fn resolve_images(mut self, cmd: &MSImageResolve) -> Self {
        self.enc.resolve_image(cmd);
        self
    }

    pub fn prepare_for_presentation(mut self, image: Handle<Image>) -> Self {
        self.enc.prepare_for_presentation(image);
        self
    }

    pub fn begin_render_pass(mut self, cmd: &BeginRenderPass) -> CommandStream<PendingGraphics> {
        self.enc.begin_render_pass(cmd);
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }

    pub fn begin_drawing(mut self, cmd: &BeginDrawing) -> CommandStream<Graphics> {
        self.enc.begin_drawing(cmd);
        let new = CommandStream {
            enc: self.enc,
            _state: PhantomData,
        };

        new.update_viewport(&cmd.viewport)
    }

    pub fn dispatch(mut self, cmd: &Dispatch) -> CommandStream<Compute> {
        self.enc.dispatch(cmd);
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }
}

impl CommandStream<Compute> {
    pub fn prepare_buffer(mut self, buffer: Handle<Buffer>, usage: UsageBits) -> Self {
        self.enc.prepare_buffer(buffer, usage, None);
        self
    }

    pub fn prepare_buffer_for(&mut self, buffer: Handle<Buffer>, usage: ResourceUse) {
        self.enc.prepare_buffer_for(buffer, usage);
    }

    pub fn prepare_image_for(&mut self, image: Handle<Image>, usage: ResourceUse) {
        self.enc.prepare_image_for(image, usage);
    }

    pub fn prepare_buffer_for_queue(
        mut self,
        buffer: Handle<Buffer>,
        usage: UsageBits,
        queue: QueueType,
    ) -> Self {
        self.enc.prepare_buffer(buffer, usage, Some(queue));
        self
    }

    pub fn copy_buffers(mut self, cmd: &CopyBuffer) -> Self {
        self.enc.copy_buffer(cmd);
        self
    }

    pub fn copy_buffer_to_image(mut self, cmd: &CopyBufferImage) -> Self {
        self.enc.copy_buffer_to_image(cmd);
        self
    }

    pub fn copy_image_to_buffer(mut self, cmd: &CopyImageBuffer) -> Self {
        self.enc.copy_image_to_buffer(cmd);
        self
    }

    pub fn blit_images(mut self, cmd: &BlitImage) -> Self {
        self.enc.blit_image(cmd);
        self
    }

    pub fn resolve_images(mut self, cmd: &MSImageResolve) -> Self {
        self.enc.resolve_image(cmd);
        self
    }

    pub fn prepare_for_presentation(mut self, image: Handle<Image>) -> Self {
        self.enc.prepare_for_presentation(image);
        self
    }

    pub fn dispatch(mut self, cmd: &Dispatch) -> Self {
        self.enc.dispatch(cmd);
        self
    }

    pub fn unbind_pipeline(self) -> CommandStream<Recording> {
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }

    pub fn begin_drawing(mut self, cmd: &BeginDrawing) -> CommandStream<Graphics> {
        self.enc.begin_drawing(cmd);
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }

    pub fn end(self) -> CommandStream<Executable> {
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }
}

impl CommandStream<PendingGraphics> {
    pub(crate) fn new_pending_graphics() -> Self {
        Self {
            enc: CommandEncoder::new(QueueType::Graphics),
            _state: PhantomData,
        }
    }

    pub fn bind_graphics_pipeline(
        mut self,
        pipeline: Handle<GraphicsPipeline>,
    ) -> CommandStream<Graphics> {
        self.enc.bind_graphics_pipeline(pipeline);
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }

    pub fn next_subpass(mut self) -> Self {
        self.enc.next_subpass();
        self
    }

    pub fn stop_drawing(mut self) -> CommandStream<Recording> {
        self.enc.end_drawing();
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }
}

impl CommandStream<Graphics> {
    pub fn bind_graphics_pipeline(mut self, pipeline: Handle<GraphicsPipeline>) -> Self {
        self.enc.bind_graphics_pipeline(pipeline);
        self
    }

    pub fn unbind_graphics_pipeline(self) -> CommandStream<PendingGraphics> {
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }

    pub fn next_subpass(mut self) -> Self {
        self.enc.next_subpass();
        self
    }

    pub fn draw(mut self, cmd: &Draw) -> Self {
        self.enc.draw(cmd);
        self
    }

    pub fn draw_indexed(mut self, cmd: &DrawIndexed) -> Self {
        self.enc.draw_indexed(cmd);
        self
    }

    pub fn update_viewport(mut self, viewport: &Viewport) -> Self {
        let mut update = GraphicsPipelineStateUpdate::default();
        update.viewport = Some(*viewport);
        self.enc.update_graphics_pipeline_state(&update);
        self
    }

    pub fn stop_drawing(mut self) -> CommandStream<Recording> {
        self.enc.end_drawing();
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }

    pub fn end(self) -> CommandStream<Executable> {
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }
}

impl CommandStream<Executable> {
    /// Submit the recorded commands to a sink and transition to pending.
    pub fn submit<S: CommandSink>(
        self,
        sink: &mut S,
        submit: &SubmitInfo2,
    ) -> Result<(CommandStream<Pending>, Option<Handle<Fence>>)> {
        let f = self.enc.submit(sink, submit)?;
        Ok((
            CommandStream {
                enc: self.enc,
                _state: PhantomData,
            },
            f,
        ))
    }

    pub fn debug_print_commands(&self) {
        for c in self.enc.iter() {
            println!("Op: {:?}", c.op);
        }
    }
    /// Submit the recorded commands to a sink and transition to pending.
    pub fn append<S: CommandSink>(self, sink: &mut S) -> Result<CommandStream<Pending>> {
        self.enc.append(sink)?;
        Ok(CommandStream {
            enc: self.enc,
            _state: PhantomData,
        })
    }
}
