use crate::driver::command::{
    BeginRenderPass, BindPipeline, BindTableCmd, BufferBarrier, CommandEncoder, CommandSink,
    CopyBuffer, CopyImage, DebugMarkerBegin, DebugMarkerEnd, Dispatch, Draw, EndRenderPass,
    ImageBarrier, Op as OpTag,
};
use crate::gpu::vulkan::CommandList;

/// Replays an encoded command stream on a Vulkan [`CommandList`].
///
/// The replayer operates purely on [`Handle`](crate::driver::types::Handle)
/// identifiers and resolves them to Vulkan objects through the existing
/// registries maintained by the `CommandList`'s context.
pub struct VkReplayer<'a> {
    list: &'a mut CommandList,
}

impl<'a> VkReplayer<'a> {
    /// Create a new Vulkan replayer targeting the given command list.
    pub fn new(list: &'a mut CommandList) -> Self {
        Self { list }
    }

    /// Iterate over the encoded command stream and issue the corresponding
    /// Vulkan calls via the underlying command list.
    pub fn replay(&mut self, encoder: &CommandEncoder) {
        for cmd in encoder.iter() {
            match cmd.op {
                OpTag::BeginRenderPass => {
                    CommandSink::begin_render_pass(self.list, cmd.payload::<BeginRenderPass>())
                }
                OpTag::EndRenderPass => {
                    CommandSink::end_render_pass(self.list, cmd.payload::<EndRenderPass>())
                }
                OpTag::BindPipeline => {
                    CommandSink::bind_pipeline(self.list, cmd.payload::<BindPipeline>())
                }
                OpTag::BindTable => {
                    CommandSink::bind_table(self.list, cmd.payload::<BindTableCmd>())
                }
                OpTag::Draw => CommandSink::draw(self.list, cmd.payload::<Draw>()),
                OpTag::Dispatch => CommandSink::dispatch(self.list, cmd.payload::<Dispatch>()),
                OpTag::CopyBuffer => {
                    CommandSink::copy_buffer(self.list, cmd.payload::<CopyBuffer>())
                }
                OpTag::CopyImage => {
                    CommandSink::copy_texture(self.list, cmd.payload::<CopyImage>())
                }
                OpTag::ImageBarrier => {
                    CommandSink::texture_barrier(self.list, cmd.payload::<ImageBarrier>())
                }
                OpTag::BufferBarrier => {
                    CommandSink::buffer_barrier(self.list, cmd.payload::<BufferBarrier>())
                }
                OpTag::DebugMarkerBegin => {
                    CommandSink::debug_marker_begin(self.list, cmd.payload::<DebugMarkerBegin>())
                }
                OpTag::DebugMarkerEnd => {
                    CommandSink::debug_marker_end(self.list, cmd.payload::<DebugMarkerEnd>())
                }
            }
        }
    }
}
