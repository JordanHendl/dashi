use std::marker::PhantomData;

pub use crate::gpu::driver::command::{Scope, SyncPoint};
use crate::gpu::driver::command::{
    BeginDrawing, BeginRenderPass, BlitImage, CommandEncoder, CommandSink, CopyBuffer,
    CopyBufferImage, CopyImageBuffer, Dispatch, Draw, DrawIndexed, DrawIndexedIndirect,
    DrawIndirect, GraphicsPipelineStateUpdate, MSImageResolve,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferUsage {
    IndirectRead,
    VertexRead,
    IndexRead,
    UniformRead,
    StorageRead,
    StorageWrite,
    CopySrc,
    CopyDst,
    HostRead,
    HostWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageUsage {
    SampledRead,
    StorageRead,
    StorageWrite,
    ColorAttachmentWrite,
    DepthAttachmentWrite,
    DepthRead,
    CopySrc,
    CopyDst,
    Present,
}

impl BufferUsage {
    fn to_resource_use(self) -> ResourceUse {
        match self {
            BufferUsage::IndirectRead => ResourceUse::IndirectRead,
            BufferUsage::VertexRead => ResourceUse::VertexRead,
            BufferUsage::IndexRead => ResourceUse::IndexRead,
            BufferUsage::UniformRead => ResourceUse::UniformRead,
            BufferUsage::StorageRead => ResourceUse::StorageRead,
            BufferUsage::StorageWrite => ResourceUse::StorageWrite,
            BufferUsage::CopySrc => ResourceUse::CopySrc,
            BufferUsage::CopyDst => ResourceUse::CopyDst,
            BufferUsage::HostRead => ResourceUse::HostRead,
            BufferUsage::HostWrite => ResourceUse::HostWrite,
        }
    }
}

impl ImageUsage {
    fn to_resource_use(self) -> ResourceUse {
        match self {
            ImageUsage::SampledRead => ResourceUse::Sampled,
            ImageUsage::StorageRead => ResourceUse::StorageRead,
            ImageUsage::StorageWrite => ResourceUse::StorageWrite,
            ImageUsage::ColorAttachmentWrite => ResourceUse::ColorAttachment,
            ImageUsage::DepthAttachmentWrite => ResourceUse::DepthAttachment,
            ImageUsage::DepthRead => ResourceUse::DepthRead,
            ImageUsage::CopySrc => ResourceUse::CopySrc,
            ImageUsage::CopyDst => ResourceUse::CopyDst,
            ImageUsage::Present => ResourceUse::Present,
        }
    }
}

pub trait PassResource {
    type Usage;
    fn apply_read(enc: &mut CommandEncoder, resource: Self, usage: Self::Usage);
    fn apply_write(enc: &mut CommandEncoder, resource: Self, usage: Self::Usage);
}

impl PassResource for Handle<Buffer> {
    type Usage = BufferUsage;

    fn apply_read(enc: &mut CommandEncoder, resource: Self, usage: Self::Usage) {
        enc.prepare_buffer_for(resource, usage.to_resource_use());
    }

    fn apply_write(enc: &mut CommandEncoder, resource: Self, usage: Self::Usage) {
        enc.prepare_buffer_for(resource, usage.to_resource_use());
    }
}

impl PassResource for Handle<Image> {
    type Usage = ImageUsage;

    fn apply_read(enc: &mut CommandEncoder, resource: Self, usage: Self::Usage) {
        enc.prepare_image_for(resource, usage.to_resource_use());
    }

    fn apply_write(enc: &mut CommandEncoder, resource: Self, usage: Self::Usage) {
        enc.prepare_image_for(resource, usage.to_resource_use());
    }
}

pub struct Pass<'a> {
    enc: &'a mut CommandEncoder,
}

impl<'a> Pass<'a> {
    pub fn read<R: PassResource>(&mut self, resource: R, usage: R::Usage) {
        R::apply_read(self.enc, resource, usage);
    }

    pub fn write<R: PassResource>(&mut self, resource: R, usage: R::Usage) {
        R::apply_write(self.enc, resource, usage);
    }
}

/// Command stream for recording subpass/subdraw work inside an active render pass.
pub type SubdrawStream = CommandStream<PendingGraphics>;

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

    /// Begin a render-graph-lite pass and declare resource dependencies.
    pub fn pass(&mut self) -> Pass<'_> {
        Pass { enc: &mut self.enc }
    }

    /// Insert an explicit global sync point between pipeline stages.
    pub fn sync(mut self, point: SyncPoint, scope: Scope) -> Self {
        self.enc.sync_point(point, scope);
        self
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

    /// Begin a GPU timer region for the specified frame index.
    pub fn gpu_timer_begin(mut self, frame: u32) -> Self {
        self.enc.gpu_timer_begin(frame);
        self
    }

    /// End a GPU timer region for the specified frame index.
    pub fn gpu_timer_end(mut self, frame: u32) -> Self {
        self.enc.gpu_timer_end(frame);
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

    /// Begin a GPU timer region for the specified frame index.
    pub fn gpu_timer_begin(mut self, frame: u32) -> Self {
        self.enc.gpu_timer_begin(frame);
        self
    }

    /// End a GPU timer region for the specified frame index.
    pub fn gpu_timer_end(mut self, frame: u32) -> Self {
        self.enc.gpu_timer_end(frame);
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
    /// Create a command stream for subdraw recording inside an active render pass.
    pub fn subdraw() -> SubdrawStream {
        Self::new_pending_graphics()
    }

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

    /// Begin a GPU timer region for the specified frame index.
    pub fn gpu_timer_begin(mut self, frame: u32) -> Self {
        self.enc.gpu_timer_begin(frame);
        self
    }

    /// End a GPU timer region for the specified frame index.
    pub fn gpu_timer_end(mut self, frame: u32) -> Self {
        self.enc.gpu_timer_end(frame);
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

    pub fn draw_indirect(mut self, cmd: &DrawIndirect) -> Self {
        self.enc.draw_indirect(cmd);
        self
    }

    pub fn draw_indexed_indirect(mut self, cmd: &DrawIndexedIndirect) -> Self {
        self.enc.draw_indexed_indirect(cmd);
        self
    }

    /// Begin a GPU timer region for the specified frame index.
    pub fn gpu_timer_begin(mut self, frame: u32) -> Self {
        self.enc.gpu_timer_begin(frame);
        self
    }

    /// End a GPU timer region for the specified frame index.
    pub fn gpu_timer_end(mut self, frame: u32) -> Self {
        self.enc.gpu_timer_end(frame);
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
