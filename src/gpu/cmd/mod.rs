use std::marker::PhantomData;

use crate::driver::command::{BeginDrawing, BlitImage, Dispatch, Draw, DrawIndexed};
use crate::driver::types::Handle;
use crate::gpu::driver::command::{
    CommandEncoder, CommandSink, CopyBuffer, CopyBufferImage, CopyImageBuffer,
};
use crate::{BindTable, Fence, GraphicsPipeline, SubmitInfo2};

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
pub struct Compute;

impl CommandStream<Initial> {
    /// Create a new command buffer in the initial state.
    pub fn new() -> Self {
        Self {
            enc: CommandEncoder::new(),
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

    pub fn begin_drawing(mut self, cmd: &BeginDrawing) -> CommandStream<Graphics> {
        self.enc.begin_drawing(cmd);
        CommandStream {
            enc: self.enc,
            _state: PhantomData,
        }
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

    pub fn dispatch(&mut self, cmd: &Dispatch) {
        self.enc.dispatch(cmd);
    }

    pub fn begin_drawing(mut self, cmd: &BeginDrawing) -> Self {
        self.enc.begin_drawing(cmd);
        self
    }

    pub fn next_subpass(mut self) -> Self {
        todo!()
    }

    pub fn end(self) -> CommandStream<Executable> {
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
    pub fn bind_graphics_pipeline(mut self, pipeline: Handle<GraphicsPipeline>) -> Self {
        self.enc.bind_graphics_pipeline(pipeline);
        self
    }

    pub fn blit_images(&mut self, cmd: &BlitImage) {
        self.enc.blit_image(cmd);
    }

    pub fn draw(&mut self, cmd: &Draw) {
        self.enc.draw(cmd);
    }

    pub fn draw_indexed(&mut self, cmd: &DrawIndexed) {
        self.enc.draw_indexed(cmd);
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
