use crate::driver::command::{
    BeginRenderPass, BindPipeline, BindTableCmd, BufferBarrier, CommandEncoder, CommandSink,
    CopyBuffer, CopyImage, DebugMarkerBegin, DebugMarkerEnd, Dispatch, Draw, EndRenderPass,
    ImageBarrier, Op as OpTag,
};

/// Trait for types that can replay a [`CommandEncoder`] on a [`CommandSink`].
pub trait Replayer<S: CommandSink> {
    /// Obtain the underlying command sink.
    fn sink(&mut self) -> &mut S;

    /// Iterate over the encoded command stream and forward each operation to
    /// the underlying sink.
    fn replay(&mut self, encoder: &CommandEncoder) {
        let sink = self.sink();
        for cmd in encoder.iter() {
            match cmd.op {
                OpTag::BeginRenderPass => {
                    CommandSink::begin_render_pass(sink, cmd.payload::<BeginRenderPass>())
                }
                OpTag::EndRenderPass => {
                    CommandSink::end_render_pass(sink, cmd.payload::<EndRenderPass>())
                }
                OpTag::BindPipeline => {
                    CommandSink::bind_pipeline(sink, cmd.payload::<BindPipeline>())
                }
                OpTag::BindTable => {
                    CommandSink::bind_table(sink, cmd.payload::<BindTableCmd>())
                }
                OpTag::Draw => CommandSink::draw(sink, cmd.payload::<Draw>()),
                OpTag::Dispatch => CommandSink::dispatch(sink, cmd.payload::<Dispatch>()),
                OpTag::CopyBuffer => {
                    CommandSink::copy_buffer(sink, cmd.payload::<CopyBuffer>())
                }
                OpTag::CopyImage => {
                    CommandSink::copy_texture(sink, cmd.payload::<CopyImage>())
                }
                OpTag::ImageBarrier => {
                    CommandSink::texture_barrier(sink, cmd.payload::<ImageBarrier>())
                }
                OpTag::BufferBarrier => {
                    CommandSink::buffer_barrier(sink, cmd.payload::<BufferBarrier>())
                }
                OpTag::DebugMarkerBegin => {
                    CommandSink::debug_marker_begin(sink, cmd.payload::<DebugMarkerBegin>())
                }
                OpTag::DebugMarkerEnd => {
                    CommandSink::debug_marker_end(sink, cmd.payload::<DebugMarkerEnd>())
                }
            }
        }
    }
}

/// Generic replayer that forwards commands to any [`CommandSink`].
pub struct CommandReplayer<'a, S: CommandSink> {
    sink: &'a mut S,
}

impl<'a, S: CommandSink> CommandReplayer<'a, S> {
    /// Create a new replayer targeting the given sink.
    pub fn new(sink: &'a mut S) -> Self {
        Self { sink }
    }
}

impl<'a, S: CommandSink> Replayer<S> for CommandReplayer<'a, S> {
    fn sink(&mut self) -> &mut S {
        &mut *self.sink
    }
}

