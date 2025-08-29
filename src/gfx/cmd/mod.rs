use std::marker::PhantomData;
use std::cell::Cell;

use crate::driver::command::{ColorAttachment, CommandEncoder, DepthAttachment, RenderPassDesc};
use crate::driver::types::{BindTable as BindTableRes, Handle, Pipeline};

/// Generic command buffer with type-state tracking.
pub struct CommandBuffer<S> {
    render_pass_active: bool,
    label_depth: u32,
    bound_pipeline: Option<Handle<Pipeline>>,
    _state: PhantomData<S>,
}

pub struct Initial;
pub struct Recording;
pub struct Executable;
pub struct Pending;

impl CommandBuffer<Initial> {
    /// Create a new command buffer in the initial state.
    pub fn new() -> Self {
        Self { render_pass_active: false, label_depth: 0, bound_pipeline: None, _state: PhantomData }
    }

    /// Begin recording commands.
    pub fn begin(self) -> CommandBuffer<Recording> {
        CommandBuffer { render_pass_active: false, label_depth: 0, bound_pipeline: None, _state: PhantomData }
    }
}

impl CommandBuffer<Recording> {
    /// Finish recording and transition to an executable state.
    pub fn end(self) -> CommandBuffer<Executable> {
        debug_assert!(!self.render_pass_active, "render pass still active");
        debug_assert_eq!(self.label_depth, 0, "debug labels still active");
        CommandBuffer { render_pass_active: false, label_depth: 0, bound_pipeline: None, _state: PhantomData }
    }
}

impl CommandBuffer<Executable> {
    /// Submit the recorded commands. This is a stub that simply transitions the
    /// state to pending.
    pub fn submit(self) -> CommandBuffer<Pending> {
        CommandBuffer { render_pass_active: false, label_depth: 0, bound_pipeline: None, _state: PhantomData }
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
    fn bind_pipeline(&mut self, pipeline: Handle<Pipeline>);
    fn bind_table(&mut self, table: Handle<BindTableRes>);
    fn draw(&mut self, vertices: u32, instances: u32);
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

impl CommandBuilder for CommandBuffer<Recording> {
    fn begin_render_pass<'a>(&mut self, _desc: RenderPassDesc<'a>) {
        debug_assert!(!self.render_pass_active, "render pass already active");
        self.render_pass_active = true;
    }

    fn end_render_pass(&mut self) {
        debug_assert!(self.render_pass_active, "no render pass active");
        self.render_pass_active = false;
    }

    fn begin_debug_label(&mut self, _label: &str) {
        self.label_depth += 1;
    }

    fn end_debug_label(&mut self) {
        debug_assert!(self.label_depth > 0, "no debug label active");
        self.label_depth -= 1;
    }

    fn bind_pipeline(&mut self, pipeline: Handle<Pipeline>) {
        self.bound_pipeline = Some(pipeline);
    }

    fn bind_table(&mut self, table: Handle<BindTableRes>) {
        assert_layout_compat(self.bound_pipeline, table);
    }

    fn draw(&mut self, _vertices: u32, _instances: u32) {
        assert_pipeline_bound(self.bound_pipeline);
        debug_assert!(self.render_pass_active, "draw outside render pass");
    }
}

impl CommandBuilder for CommandEncoder {
    fn begin_render_pass<'a>(&mut self, desc: RenderPassDesc<'a>) {
        CommandEncoder::begin_render_pass(self, desc);
    }

    fn end_render_pass(&mut self) {
        CommandEncoder::end_render_pass(self);
    }

    fn begin_debug_label(&mut self, _label: &str) {
        self.begin_debug_marker();
    }

    fn end_debug_label(&mut self) {
        self.end_debug_marker();
    }

    fn bind_pipeline(&mut self, pipeline: Handle<Pipeline>) {
        ENCODER_PIPELINE.with(|p| p.set(Some(pipeline)));
        CommandEncoder::bind_pipeline(self, pipeline);
    }

    fn bind_table(&mut self, table: Handle<BindTableRes>) {
        let pipeline = ENCODER_PIPELINE.with(|p| p.get());
        assert_layout_compat(pipeline, table);
        CommandEncoder::bind_table(self, table);
    }

    fn draw(&mut self, vertices: u32, instances: u32) {
        let pipeline = ENCODER_PIPELINE.with(|p| p.get());
        assert_pipeline_bound(pipeline);
        CommandEncoder::draw(self, vertices, instances);
    }
}

thread_local! {
    static ENCODER_PIPELINE: Cell<Option<Handle<Pipeline>>> = Cell::new(None);
}

#[inline]
fn assert_pipeline_bound(pipeline: Option<Handle<Pipeline>>) {
    debug_assert!(pipeline.is_some(), "pipeline not bound");
}

#[inline]
fn assert_layout_compat(pipeline: Option<Handle<Pipeline>>, _table: Handle<BindTableRes>) {
    assert_pipeline_bound(pipeline);
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

    pub fn build(self, cmd: &mut impl CommandBuilder) {
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
