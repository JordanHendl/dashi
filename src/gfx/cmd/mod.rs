use std::marker::PhantomData;

use crate::driver::command::{CommandEncoder, RenderPassDesc};
use crate::driver::types::{BindTable as BindTableRes, Handle, Pipeline};

/// Generic command buffer with type-state tracking.
pub struct CommandBuffer<S> {
    render_pass_active: bool,
    label_depth: u32,
    _state: PhantomData<S>,
}

pub struct Initial;
pub struct Recording;
pub struct Executable;
pub struct Pending;

impl CommandBuffer<Initial> {
    /// Create a new command buffer in the initial state.
    pub fn new() -> Self {
        Self { render_pass_active: false, label_depth: 0, _state: PhantomData }
    }

    /// Begin recording commands.
    pub fn begin(self) -> CommandBuffer<Recording> {
        CommandBuffer { render_pass_active: false, label_depth: 0, _state: PhantomData }
    }
}

impl CommandBuffer<Recording> {
    /// Finish recording and transition to an executable state.
    pub fn end(self) -> CommandBuffer<Executable> {
        debug_assert!(!self.render_pass_active, "render pass still active");
        debug_assert_eq!(self.label_depth, 0, "debug labels still active");
        CommandBuffer { render_pass_active: false, label_depth: 0, _state: PhantomData }
    }
}

impl CommandBuffer<Executable> {
    /// Submit the recorded commands. This is a stub that simply transitions the
    /// state to pending.
    pub fn submit(self) -> CommandBuffer<Pending> {
        CommandBuffer { render_pass_active: false, label_depth: 0, _state: PhantomData }
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

    fn bind_pipeline(&mut self, _pipeline: Handle<Pipeline>) {}

    fn bind_table(&mut self, _table: Handle<BindTableRes>) {}

    fn draw(&mut self, _vertices: u32, _instances: u32) {
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
        CommandEncoder::bind_pipeline(self, pipeline);
    }

    fn bind_table(&mut self, table: Handle<BindTableRes>) {
        CommandEncoder::bind_table(self, table);
    }

    fn draw(&mut self, vertices: u32, instances: u32) {
        CommandEncoder::draw(self, vertices, instances);
    }
}
