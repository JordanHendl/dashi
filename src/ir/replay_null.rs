use crate::driver::command::{
    BeginRenderPass, BindPipeline, BindTableCmd, BufferBarrier, CommandEncoder, CommandSink,
    CopyBuffer, CopyImage, DebugMarkerBegin, DebugMarkerEnd, Dispatch, Draw, EndRenderPass,
    ImageBarrier, Op as OpTag,
};

/// Replays an encoded command stream on any [`CommandSink`].
///
/// This is useful for tests or headless backends where commands are
/// interpreted without performing real GPU work.
pub struct NullReplayer<'a, S: CommandSink> {
    sink: &'a mut S,
}

impl<'a, S: CommandSink> NullReplayer<'a, S> {
    /// Create a new replayer targeting the given sink.
    pub fn new(sink: &'a mut S) -> Self {
        Self { sink }
    }

    /// Iterate over the encoded command stream and forward each operation to
    /// the underlying sink.
    pub fn replay(&mut self, encoder: &CommandEncoder) {
        for cmd in encoder.iter() {
            match cmd.op {
                OpTag::BeginRenderPass => {
                    CommandSink::begin_render_pass(self.sink, cmd.payload::<BeginRenderPass>())
                }
                OpTag::EndRenderPass => {
                    CommandSink::end_render_pass(self.sink, cmd.payload::<EndRenderPass>())
                }
                OpTag::BindPipeline => {
                    CommandSink::bind_pipeline(self.sink, cmd.payload::<BindPipeline>())
                }
                OpTag::BindTable => {
                    CommandSink::bind_table(self.sink, cmd.payload::<BindTableCmd>())
                }
                OpTag::Draw => CommandSink::draw(self.sink, cmd.payload::<Draw>()),
                OpTag::Dispatch => CommandSink::dispatch(self.sink, cmd.payload::<Dispatch>()),
                OpTag::CopyBuffer => {
                    CommandSink::copy_buffer(self.sink, cmd.payload::<CopyBuffer>())
                }
                OpTag::CopyImage => {
                    CommandSink::copy_texture(self.sink, cmd.payload::<CopyImage>())
                }
                OpTag::ImageBarrier => {
                    CommandSink::texture_barrier(self.sink, cmd.payload::<ImageBarrier>())
                }
                OpTag::BufferBarrier => {
                    CommandSink::buffer_barrier(self.sink, cmd.payload::<BufferBarrier>())
                }
                OpTag::DebugMarkerBegin => {
                    CommandSink::debug_marker_begin(self.sink, cmd.payload::<DebugMarkerBegin>())
                }
                OpTag::DebugMarkerEnd => {
                    CommandSink::debug_marker_end(self.sink, cmd.payload::<DebugMarkerEnd>())
                }
            }
        }
    }
}

