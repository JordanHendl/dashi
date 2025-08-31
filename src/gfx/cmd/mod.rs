use std::marker::PhantomData;

use crate::{Buffer, Image};
use crate::driver::command::{
    ColorAttachment, CommandEncoder, CommandSink, DepthAttachment, RenderPassDesc,
};
use crate::driver::state::SubresourceRange;
use crate::driver::types::{BindTable as BindTableRes, Handle, Pipeline};

/// Generic command buffer with type-state tracking.
pub struct CommandBuffer<S> {
    enc: CommandEncoder,
    _state: PhantomData<S>,
}

pub struct Initial;
pub struct Recording;
pub struct Executable;
pub struct Pending;
pub struct PipelineBound;

impl CommandBuffer<Initial> {
    /// Create a new command buffer in the initial state.
    pub fn new() -> Self {
        Self { enc: CommandEncoder::new(), _state: PhantomData }
    }

    /// Begin recording commands.
    pub fn begin(self) -> CommandBuffer<Recording> {
        CommandBuffer { enc: self.enc, _state: PhantomData }
    }
}

impl CommandBuffer<Recording> {
    /// Finish recording and transition to an executable state.
    pub fn end(self) -> CommandBuffer<Executable> {
        CommandBuffer { enc: self.enc, _state: PhantomData }
    }

    pub fn bind_pipeline(mut self, pipeline: Handle<Pipeline>) -> CommandBuffer<PipelineBound> {
        self.enc.bind_pipeline(pipeline);
        CommandBuffer { enc: self.enc, _state: PhantomData }
    }
}

impl CommandBuffer<PipelineBound> {
    pub fn bind_pipeline(mut self, pipeline: Handle<Pipeline>) -> Self {
        self.enc.bind_pipeline(pipeline);
        self
    }

    pub fn draw(&mut self, vertices: u32, instances: u32) {
        self.enc.draw(vertices, instances);
    }

    pub fn bind_table(&mut self, table: Handle<BindTableRes>) {
        self.enc.bind_table(table);
    }

    pub fn end(self) -> CommandBuffer<Executable> {
        CommandBuffer { enc: self.enc, _state: PhantomData }
    }
}

impl CommandBuffer<Executable> {
    /// Submit the recorded commands to a sink and transition to pending.
    pub fn submit<S: CommandSink>(self, sink: &mut S) -> CommandBuffer<Pending> {
        self.enc.submit(sink);
        CommandBuffer { enc: self.enc, _state: PhantomData }
    }
}

/// RAII scope for a render pass. Ends the pass when dropped.
pub struct RenderScope<'a, T: CommandBuilder + ?Sized> {
    cmd: &'a mut T,
}

impl<'a, T: CommandBuilder + ?Sized> Drop for RenderScope<'a, T> {
    fn drop(&mut self) {
        self.cmd.end_render_pass();
    }
}

/// RAII scope for a debug label. Ends the label when dropped.
pub struct DebugLabelScope<'a, T: CommandBuilder + ?Sized> {
    cmd: &'a mut T,
}

impl<'a, T: CommandBuilder + ?Sized> Drop for DebugLabelScope<'a, T> {
    fn drop(&mut self) {
        self.cmd.end_debug_label();
    }
}

/// A trait describing common command-buffer style operations. Both the IR
/// [`CommandEncoder`] and direct [`CommandBuffer`] implement this so callers can
/// use a single ergonomic front end regardless of the underlying recording
/// mechanism.
pub trait CommandBuilder {
    fn begin_render_pass<'a>(&mut self, desc: RenderPassDesc<'a>);
    fn end_render_pass(&mut self);
    fn begin_debug_label(&mut self, label: &str);
    fn end_debug_label(&mut self);
    fn texture_barrier(&mut self, image: Handle<Image>, range: SubresourceRange);
    fn buffer_barrier(&mut self, buffer: Handle<Buffer>);
}

/// Extension methods providing RAII helpers over [`CommandBuilder`] objects.
pub trait CommandBuilderExt: CommandBuilder {
    fn with_label<'a>(&'a mut self, label: &str) -> DebugLabelScope<'a, Self>
    where
        Self: Sized,
    {
        self.begin_debug_label(label);
        DebugLabelScope { cmd: self }
    }

    fn begin_targets<'a>(&'a mut self, desc: RenderPassDesc<'a>) -> RenderScope<'a, Self>
    where
        Self: Sized,
    {
        self.begin_render_pass(desc);
        RenderScope { cmd: self }
    }
}

impl<T: CommandBuilder + ?Sized> CommandBuilderExt for T {}

/// Target for encoding commands. This can either record into an IR
/// [`CommandEncoder`] or directly into an immediate [`CommandBuffer`].
pub enum EncodeTarget<'a> {
    /// Record to the intermediate representation encoder.
    IR(&'a mut CommandEncoder),
    /// Record directly to a command buffer in recording state.
    CB(&'a mut CommandBuffer<Recording>),
}

impl<'a> From<&'a mut CommandEncoder> for EncodeTarget<'a> {
    fn from(enc: &'a mut CommandEncoder) -> Self { Self::IR(enc) }
}

impl<'a> From<&'a mut CommandBuffer<Recording>> for EncodeTarget<'a> {
    fn from(cb: &'a mut CommandBuffer<Recording>) -> Self { Self::CB(cb) }
}

impl<'a> CommandBuilder for EncodeTarget<'a> {
    fn begin_render_pass<'b>(&mut self, desc: RenderPassDesc<'b>) {
        match self {
            EncodeTarget::IR(enc) => enc.begin_render_pass(desc),
            EncodeTarget::CB(cb) => cb.begin_render_pass(desc),
        }
    }

    fn end_render_pass(&mut self) {
        match self {
            EncodeTarget::IR(enc) => enc.end_render_pass(),
            EncodeTarget::CB(cb) => cb.end_render_pass(),
        }
    }

    fn begin_debug_label(&mut self, label: &str) {
        match self {
            EncodeTarget::IR(enc) => enc.begin_debug_label(label),
            EncodeTarget::CB(cb) => cb.begin_debug_label(label),
        }
    }

    fn end_debug_label(&mut self) {
        match self {
            EncodeTarget::IR(enc) => enc.end_debug_label(),
            EncodeTarget::CB(cb) => cb.end_debug_label(),
        }
    }

    fn texture_barrier(&mut self, image: Handle<Image>, range: SubresourceRange) {
        match self {
            EncodeTarget::IR(enc) => enc.texture_barrier(image, range),
            EncodeTarget::CB(cb) => cb.texture_barrier(image, range),
        }
    }

    fn buffer_barrier(&mut self, buffer: Handle<Buffer>) {
        match self {
            EncodeTarget::IR(enc) => enc.buffer_barrier(buffer),
            EncodeTarget::CB(cb) => cb.buffer_barrier(buffer),
        }
    }
}

impl CommandBuilder for CommandBuffer<Recording> {
    fn begin_render_pass<'a>(&mut self, desc: RenderPassDesc<'a>) {
        self.enc.begin_render_pass(desc);
    }

    fn end_render_pass(&mut self) {
        self.enc.end_render_pass();
    }

    fn begin_debug_label(&mut self, _label: &str) {
        self.enc.begin_debug_marker();
    }

    fn end_debug_label(&mut self) {
        self.enc.end_debug_marker();
    }

    fn texture_barrier(&mut self, image: Handle<Image>, range: SubresourceRange) {
        self.enc.texture_barrier(image, range);
    }

    fn buffer_barrier(&mut self, buffer: Handle<Buffer>) {
        self.enc.buffer_barrier(buffer);
    }
}

impl CommandBuilder for CommandBuffer<PipelineBound> {
    fn begin_render_pass<'a>(&mut self, desc: RenderPassDesc<'a>) {
        self.enc.begin_render_pass(desc);
    }

    fn end_render_pass(&mut self) {
        self.enc.end_render_pass();
    }

    fn begin_debug_label(&mut self, _label: &str) {
        self.enc.begin_debug_marker();
    }

    fn end_debug_label(&mut self) {
        self.enc.end_debug_marker();
    }

    fn texture_barrier(&mut self, image: Handle<Image>, range: SubresourceRange) {
        self.enc.texture_barrier(image, range);
    }

    fn buffer_barrier(&mut self, buffer: Handle<Buffer>) {
        self.enc.buffer_barrier(buffer);
    }
}

pub trait PipelineBuilder {
    fn bind_table(&mut self, table: Handle<BindTableRes>);
}

impl PipelineBuilder for CommandBuffer<PipelineBound> {
    fn bind_table(&mut self, table: Handle<BindTableRes>) {
        self.enc.bind_table(table);
    }
}

pub struct DescriptorWriteBuilder {
    table: Option<Handle<BindTableRes>>,
}

impl DescriptorWriteBuilder {
    pub fn new() -> Self {
        Self { table: None }
    }

    pub fn table(mut self, table: Handle<BindTableRes>) -> Self {
        self.table = Some(table);
        self
    }

    pub fn build(self, cmd: &mut impl PipelineBuilder) {
        if let Some(t) = self.table {
            cmd.bind_table(t);
        }
    }
}

pub struct DynamicRenderingBuilder<'a> {
    colors: Vec<ColorAttachment>,
    depth: Option<DepthAttachment>,
    _phantom: PhantomData<&'a ()>,
}

impl<'a> DynamicRenderingBuilder<'a> {
    pub fn new() -> Self {
        Self { colors: Vec::new(), depth: None, _phantom: PhantomData }
    }

    pub fn color(mut self, color: ColorAttachment) -> Self {
        self.colors.push(color);
        self
    }

    pub fn depth(mut self, depth: DepthAttachment) -> Self {
        self.depth = Some(depth);
        self
    }

    pub fn begin<T: CommandBuilder + ?Sized>(self, cmd: &mut T) -> RenderScope<'_, T> {
        let desc = RenderPassDesc { colors: &self.colors, depth: self.depth };
        cmd.begin_render_pass(desc);
        RenderScope { cmd }
    }
}
