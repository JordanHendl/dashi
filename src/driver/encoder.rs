use super::{
    ir::{
        CommandStream, Op, BeginRenderPass, EndRenderPass, BindPipeline, BindTable, Draw, Dispatch,
        CopyBuffer, CopyTexture, DebugMarkerBegin, DebugMarkerEnd,
    },
    state::{StateTracker, SubresourceRange},
    types::{BindTableHandle, BufferHandle, PipelineHandle, TextureHandle, UsageBits},
};

/// Encoder records high level operations into a [`CommandStream`].
/// It also uses [`StateTracker`] to automatically emit resource barriers
/// whenever resource usage changes.
pub struct Encoder {
    stream: CommandStream,
    state: StateTracker,
}

impl Encoder {
    /// Create a new empty encoder.
    pub fn new() -> Self {
        Self { stream: CommandStream::new(), state: StateTracker::new() }
    }

    /// Finish recording and return the underlying command stream.
    pub fn finish(self) -> CommandStream {
        self.stream
    }

    /// Begin a render pass with a single color attachment.
    pub fn begin_render_pass(&mut self, color: TextureHandle, range: SubresourceRange) {
        if let Some(bar) = self.state.request_texture_state(color, range, UsageBits::RT_WRITE) {
            self.stream.push(Op::TextureBarrier, &bar);
        }
        let payload = BeginRenderPass { color_attachments: color.0 };
        self.stream.push(Op::BeginRenderPass, &payload);
    }

    /// End the current render pass.
    pub fn end_render_pass(&mut self) {
        self.stream.push(Op::EndRenderPass, &EndRenderPass {});
    }

    /// Bind a graphics or compute pipeline.
    pub fn bind_pipeline(&mut self, pipe: PipelineHandle) {
        let payload = BindPipeline { pipeline_id: pipe.0 };
        self.stream.push(Op::BindPipeline, &payload);
    }

    /// Bind a table of resources.
    pub fn bind_table(&mut self, table: BindTableHandle) {
        let payload = BindTable { table_id: table.0 };
        self.stream.push(Op::BindTable, &payload);
    }

    /// Issue a draw call.
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32) {
        let payload = Draw { vertex_count, instance_count };
        self.stream.push(Op::Draw, &payload);
    }

    /// Issue a dispatch call.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        let payload = Dispatch { x, y, z };
        self.stream.push(Op::Dispatch, &payload);
    }

    /// Copy data between buffers. Barriers for source and destination
    /// are emitted automatically.
    pub fn copy_buffer(&mut self, src: BufferHandle, dst: BufferHandle) {
        if let Some(bar) = self.state.request_buffer_state(src, UsageBits::COPY_SRC) {
            self.stream.push(Op::BufferBarrier, &bar);
        }
        if let Some(bar) = self.state.request_buffer_state(dst, UsageBits::COPY_DST) {
            self.stream.push(Op::BufferBarrier, &bar);
        }
        let payload = CopyBuffer { src: src.0, dst: dst.0 };
        self.stream.push(Op::CopyBuffer, &payload);
    }

    /// Copy data between textures, emitting required barriers.
    pub fn copy_texture(&mut self, src: TextureHandle, dst: TextureHandle, range: SubresourceRange) {
        if let Some(bar) = self.state.request_texture_state(src, range, UsageBits::COPY_SRC) {
            self.stream.push(Op::TextureBarrier, &bar);
        }
        if let Some(bar) = self.state.request_texture_state(dst, range, UsageBits::COPY_DST) {
            self.stream.push(Op::TextureBarrier, &bar);
        }
        let payload = CopyTexture { src: src.0, dst: dst.0 };
        self.stream.push(Op::CopyTexture, &payload);
    }

    /// Begin a debug marker region.
    pub fn begin_debug_marker(&mut self) {
        self.stream.push(Op::DebugMarkerBegin, &DebugMarkerBegin {});
    }

    /// End a debug marker region.
    pub fn end_debug_marker(&mut self) {
        self.stream.push(Op::DebugMarkerEnd, &DebugMarkerEnd {});
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn barriers_are_emitted() {
        let mut enc = Encoder::new();
        let tex = TextureHandle(1);
        let range = SubresourceRange::new(0, 1, 0, 1);
        enc.begin_render_pass(tex, range);
        enc.end_render_pass();
        let stream = enc.finish();
        let cmds: Vec<_> = stream.iter().collect();
        assert_eq!(cmds[0].op, Op::TextureBarrier);
        assert_eq!(cmds[1].op, Op::BeginRenderPass);
        assert_eq!(cmds[2].op, Op::EndRenderPass);
    }

    #[test]
    fn copy_buffer_emits_barriers() {
        let mut enc = Encoder::new();
        let src = BufferHandle(2);
        let dst = BufferHandle(3);
        enc.copy_buffer(src, dst);
        let stream = enc.finish();
        let cmds: Vec<_> = stream.iter().collect();
        assert_eq!(cmds[0].op, Op::BufferBarrier);
        assert_eq!(cmds[1].op, Op::BufferBarrier);
        assert_eq!(cmds[2].op, Op::CopyBuffer);
    }
}
