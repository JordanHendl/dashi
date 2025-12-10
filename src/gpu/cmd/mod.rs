use std::marker::PhantomData;

use crate::gpu::driver::command::{
    BeginDrawing, BeginRenderPass, BlitImage, Dispatch, Draw, DrawIndexed,
    GraphicsPipelineStateUpdate, MSImageResolve,
};
use crate::gpu::driver::command::{
    CommandEncoder, CommandSink, CopyBuffer, CopyBufferImage, CopyImageBuffer,
};
use crate::gpu::driver::types::Handle;
use crate::{Fence, GraphicsPipeline, Image, QueueType, SubmitInfo2, Viewport};

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
    pub fn combine<G>(&mut self, sink: CommandStream<G>) {
        self.enc.combine(&sink.enc);
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

    pub fn copy_buffers(&mut self, cmd: &CopyBuffer) {
        self.enc.copy_buffer(cmd);
    }

    pub fn copy_buffer_to_image(&mut self, cmd: &CopyBufferImage) {
        self.enc.copy_buffer_to_image(cmd);
    }

    pub fn copy_image_to_buffer(&mut self, cmd: &CopyImageBuffer) {
        self.enc.copy_image_to_buffer(cmd);
    }

    pub fn blit_images(&mut self, cmd: &BlitImage) {
        self.enc.blit_image(cmd);
    }

    pub fn resolve_images(&mut self, cmd: &MSImageResolve) {
        self.enc.resolve_image(cmd);
    }

    pub fn prepare_for_presentation(&mut self, image: Handle<Image>) {
        self.enc.prepare_for_presentation(image);
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
        let mut new = CommandStream {
            enc: self.enc,
            _state: PhantomData,
        };

        new.update_viewport(&cmd.viewport);
        new
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
    pub fn copy_buffers(&mut self, cmd: &CopyBuffer) {
        self.enc.copy_buffer(cmd);
    }

    pub fn copy_buffer_to_image(&mut self, cmd: &CopyBufferImage) {
        self.enc.copy_buffer_to_image(cmd);
    }

    pub fn copy_image_to_buffer(&mut self, cmd: &CopyImageBuffer) {
        self.enc.copy_image_to_buffer(cmd);
    }

    pub fn blit_images(&mut self, cmd: &BlitImage) {
        self.enc.blit_image(cmd);
    }

    pub fn resolve_images(&mut self, cmd: &MSImageResolve) {
        self.enc.resolve_image(cmd);
    }

    pub fn prepare_for_presentation(&mut self, image: Handle<Image>) {
        self.enc.prepare_for_presentation(image);
    }

    pub fn dispatch(&mut self, cmd: &Dispatch) {
        self.enc.dispatch(cmd);
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

    pub fn copy_buffers(&mut self, cmd: &CopyBuffer) {
        self.enc.copy_buffer(cmd);
    }

    pub fn copy_buffer_to_image(&mut self, cmd: &CopyBufferImage) {
        self.enc.copy_buffer_to_image(cmd);
    }

    pub fn copy_image_to_buffer(&mut self, cmd: &CopyImageBuffer) {
        self.enc.copy_image_to_buffer(cmd);
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

    pub fn blit_images(&mut self, cmd: &BlitImage) {
        self.enc.blit_image(cmd);
    }

    pub fn resolve_images(&mut self, cmd: &MSImageResolve) {
        self.enc.resolve_image(cmd);
    }

    pub fn next_subpass(&mut self) {
        self.enc.next_subpass();
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
    pub fn copy_buffers(&mut self, cmd: &CopyBuffer) {
        self.enc.copy_buffer(cmd);
    }

    pub fn copy_buffer_to_image(&mut self, cmd: &CopyBufferImage) {
        self.enc.copy_buffer_to_image(cmd);
    }

    pub fn copy_image_to_buffer(&mut self, cmd: &CopyImageBuffer) {
        self.enc.copy_image_to_buffer(cmd);
    }

    pub fn bind_graphics_pipeline(mut self, pipeline: Handle<GraphicsPipeline>) {
        self.enc.bind_graphics_pipeline(pipeline);
    }

    pub fn unbind_graphics_pipeline(self) -> CommandStream<PendingGraphics> {
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }

    pub fn blit_images(&mut self, cmd: &BlitImage) {
        self.enc.blit_image(cmd);
    }

    pub fn resolve_images(&mut self, cmd: &MSImageResolve) {
        self.enc.resolve_image(cmd);
    }

    pub fn next_subpass(&mut self) {
        self.enc.next_subpass();
    }

    pub fn prepare_for_presentation(&mut self, image: Handle<Image>) {
        self.enc.prepare_for_presentation(image);
    }

    pub fn draw(&mut self, cmd: &Draw) {
        self.enc.draw(cmd);
    }

    pub fn draw_indexed(&mut self, cmd: &DrawIndexed) {
        self.enc.draw_indexed(cmd);
    }

    pub fn update_viewport(&mut self, viewport: &Viewport) {
        let mut update = GraphicsPipelineStateUpdate::default();
        update.viewport = Some(*viewport);
        self.enc.update_graphics_pipeline_state(&update);
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
    ) -> (CommandStream<Pending>, Option<Handle<Fence>>) {
        let f = self.enc.submit(sink, submit);
        (
            CommandStream {
                enc: self.enc,
                _state: PhantomData,
            },
            f,
        )
    }

    /// Submit the recorded commands to a sink and transition to pending.
    pub fn append<S: CommandSink>(self, sink: &mut S) -> CommandStream<Pending> {
        self.enc.append(sink);
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
    }
}
