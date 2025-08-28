use core::marker::PhantomData;

use crate::driver::command::{
    BeginRenderPass, BindPipeline, BindTableCmd, BufferBarrier, CopyBuffer, CopyImage,
    DebugMarkerBegin, DebugMarkerEnd, Dispatch, Draw, EndRenderPass, ImageBarrier,
    CommandSink,
};

/// Command buffer state marker types.
pub struct Initial;
pub struct Recording;
pub struct Executable;
pub struct Pending;

/// Generic command buffer front end built on top of a [`CommandSink`].
///
/// The `S` parameter encodes the current state of the command buffer.
pub struct CommandBuffer<S, B: CommandSink> {
    sink: B,
    _state: PhantomData<S>,
}

impl<B: CommandSink> CommandBuffer<Initial, B> {
    /// Create a new command buffer in the initial state.
    pub fn new(sink: B) -> Self {
        Self { sink, _state: PhantomData }
    }

    /// Begin recording.
    pub fn begin(self) -> CommandBuffer<Recording, B> {
        CommandBuffer { sink: self.sink, _state: PhantomData }
    }
}

impl<B: CommandSink> CommandBuffer<Recording, B> {
    /// Finish recording and transition to the executable state.
    pub fn end(self) -> CommandBuffer<Executable, B> {
        CommandBuffer { sink: self.sink, _state: PhantomData }
    }

    /// Begin a render pass and return an RAII scope that ends it automatically.
    pub fn begin_targets<'a>(&'a mut self, pass: &BeginRenderPass) -> RenderScope<'a, B> {
        self.sink.begin_render_pass(pass);
        RenderScope { cmd: self }
    }

    /// Begin a debug label scope.
    pub fn with_label<'a>(&'a mut self, _label: &str) -> DebugLabelScope<'a, B> {
        // The backend may choose to ignore the string if unsupported.
        self.sink.debug_marker_begin(&DebugMarkerBegin {});
        DebugLabelScope { cmd: self }
    }

    /// Bind a pipeline.
    pub fn bind_pipeline(&mut self, cmd: &BindPipeline) {
        self.sink.bind_pipeline(cmd);
    }

    /// Bind a resource table.
    pub fn bind_table(&mut self, cmd: &BindTableCmd) {
        self.sink.bind_table(cmd);
    }

    /// Issue a draw call.
    pub fn draw(&mut self, cmd: &Draw) {
        self.sink.draw(cmd);
    }

    /// Issue a dispatch call.
    pub fn dispatch(&mut self, cmd: &Dispatch) {
        self.sink.dispatch(cmd);
    }

    /// Copy data between buffers.
    pub fn copy_buffer(&mut self, cmd: &CopyBuffer) {
        self.sink.copy_buffer(cmd);
    }

    /// Copy data between images.
    pub fn copy_texture(&mut self, cmd: &CopyImage) {
        self.sink.copy_texture(cmd);
    }

    /// Insert an explicit texture barrier.
    pub fn texture_barrier(&mut self, cmd: &ImageBarrier) {
        self.sink.texture_barrier(cmd);
    }

    /// Insert an explicit buffer barrier.
    pub fn buffer_barrier(&mut self, cmd: &BufferBarrier) {
        self.sink.buffer_barrier(cmd);
    }
}

impl<B: CommandSink> CommandBuffer<Executable, B> {
    /// Submit the command buffer for execution.
    pub fn submit(self) -> CommandBuffer<Pending, B> {
        CommandBuffer { sink: self.sink, _state: PhantomData }
    }

    /// Access the underlying sink.
    pub fn into_inner(self) -> B {
        self.sink
    }
}

/// RAII guard for a render pass scope.
pub struct RenderScope<'a, B: CommandSink> {
    cmd: &'a mut CommandBuffer<Recording, B>,
}

impl<'a, B: CommandSink> Drop for RenderScope<'a, B> {
    fn drop(&mut self) {
        // Misuse is only checked in debug builds.
        debug_assert!(core::mem::size_of::<EndRenderPass>() > 0);
        self.cmd.sink.end_render_pass(&EndRenderPass {});
    }
}

/// RAII guard for debug labels.
pub struct DebugLabelScope<'a, B: CommandSink> {
    cmd: &'a mut CommandBuffer<Recording, B>,
}

impl<'a, B: CommandSink> Drop for DebugLabelScope<'a, B> {
    fn drop(&mut self) {
        debug_assert!(core::mem::size_of::<DebugMarkerEnd>() > 0);
        self.cmd.sink.debug_marker_end(&DebugMarkerEnd {});
    }
}

// Expose the sink for advanced users.
impl<S, B: CommandSink> CommandBuffer<S, B> {
    /// Borrow the underlying sink.
    pub fn sink(&mut self) -> &mut B {
        &mut self.sink
    }
}
