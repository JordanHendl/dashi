use bytemuck::{bytes_of, from_bytes, Pod, Zeroable};

use crate::{Image, Buffer};

use super::{
    state::{StateTracker, SubresourceRange},
    types::{BindTable as BindTableRes, Pipeline, UsageBits, Handle},
};

//===----------------------------------------------------------------------===//
// Command definitions
//===----------------------------------------------------------------------===//

#[repr(u16)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Op {
    BeginRenderPass = 0,
    EndRenderPass = 1,
    BindPipeline = 2,
    BindTable = 3,
    Draw = 4,
    Dispatch = 5,
    CopyBuffer = 6,
    CopyImage = 7,
    ImageBarrier = 8,
    BufferBarrier = 9,
    DebugMarkerBegin = 10,
    DebugMarkerEnd = 11,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadOp {
    Load = 0,
    Clear = 1,
    DontCare = 2,
}

unsafe impl Zeroable for LoadOp {}
unsafe impl Pod for LoadOp {}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreOp {
    Store = 0,
    DontCare = 1,
}

unsafe impl Zeroable for StoreOp {}
unsafe impl Pod for StoreOp {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct ColorAttachment {
    pub handle: Handle<Image>,
    pub range: SubresourceRange,
    pub clear: [f32; 4],
    pub load: LoadOp,
    pub store: StoreOp,
    pub _pad: [u8; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct DepthAttachment {
    pub handle: Handle<Image>,
    pub range: SubresourceRange,
    pub clear: f32,
    pub load: LoadOp,
    pub store: StoreOp,
    pub _pad: [u8; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct BeginRenderPass {
    pub colors: [ColorAttachment; 4],
    pub color_count: u32,
    pub depth: DepthAttachment,
    pub has_depth: u32,
}

pub struct RenderPassDesc<'a> {
    pub colors: &'a [ColorAttachment],
    pub depth: Option<DepthAttachment>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct EndRenderPass {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct BindPipeline {
    pub pipeline: Handle<Pipeline>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct BindTableCmd {
    pub table: Handle<BindTableRes>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct Draw {
    pub vertex_count: u32,
    pub instance_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct Dispatch {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct CopyBuffer {
    pub src: Handle<Buffer>,
    pub dst: Handle<Buffer>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct CopyImage {
    pub src: Handle<Image>,
    pub dst: Handle<Image>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct ImageBarrier {
    pub image: Handle<Image>,
    pub range: SubresourceRange,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct BufferBarrier {
    pub buffer: Handle<Buffer>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct DebugMarkerBegin {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct DebugMarkerEnd {}

//===----------------------------------------------------------------------===//
// Command encoder & stream
//===----------------------------------------------------------------------===//

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CmdHeader {
    op: u16,
    size: u16,
}

/// A single structure that both records commands and stores them in a raw stream.
///
/// The encoder keeps track of resource state and writes a compact byte stream that
/// can be iterated over later.  The implementation is intentionally low level and
/// strives to avoid unnecessary overhead in order to be "blazingly fast".
pub struct CommandEncoder {
    data: Vec<u8>,
    state: StateTracker,
}

impl CommandEncoder {
    /// Create a new empty encoder. Some initial capacity is reserved to avoid
    /// frequent reallocations when recording many commands.
    pub fn new() -> Self {
        Self { data: Vec::with_capacity(1024), state: StateTracker::new() }
    }

    /// Push a command payload into the internal byte stream. The method is marked
    /// `inline(always)` so that callers can be optimized tightly.
    #[inline(always)]
    fn push<T: Pod>(&mut self, op: Op, payload: &T) {
        let header = CmdHeader { op: op as u16, size: core::mem::size_of::<T>() as u16 };
        self.data.extend_from_slice(bytes_of(&header));
        self.data.extend_from_slice(bytes_of(payload));
    }

    /// Begin a render pass with the provided attachments.
    pub fn begin_render_pass(&mut self, desc: RenderPassDesc) {
        let mut payload = BeginRenderPass {
            colors: [ColorAttachment::zeroed(); 4],
            color_count: desc.colors.len() as u32,
            depth: DepthAttachment::zeroed(),
            has_depth: desc.depth.is_some() as u32,
        };
        for (i, att) in desc.colors.iter().enumerate() {
            if let Some(bar) =
                self.state.request_texture_state(att.handle, att.range, UsageBits::RT_WRITE)
            {
                self.push(Op::ImageBarrier, &bar);
            }
            payload.colors[i] = *att;
        }
        if let Some(depth) = desc.depth {
            if let Some(bar) =
                self.state.request_texture_state(depth.handle, depth.range, UsageBits::DEPTH_WRITE)
            {
                self.push(Op::ImageBarrier, &bar);
            }
            payload.depth = depth;
        }
        self.push(Op::BeginRenderPass, &payload);
    }

    /// End the current render pass.
    pub fn end_render_pass(&mut self) {
        self.push(Op::EndRenderPass, &EndRenderPass {});
    }

    /// Bind a graphics or compute pipeline.
    pub fn bind_pipeline(&mut self, pipe: Handle<Pipeline>) {
        let payload = BindPipeline { pipeline: pipe };
        self.push(Op::BindPipeline, &payload);
    }

    /// Bind a table of resources.
    pub fn bind_table(&mut self, table: Handle<BindTableRes>) {
        let payload = BindTableCmd { table };
        self.push(Op::BindTable, &payload);
    }

    /// Issue a draw call.
    pub fn draw(&mut self, vertex_count: u32, instance_count: u32) {
        let payload = Draw { vertex_count, instance_count };
        self.push(Op::Draw, &payload);
    }

    /// Issue a dispatch call.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) {
        let payload = Dispatch { x, y, z };
        self.push(Op::Dispatch, &payload);
    }

    /// Copy data between buffers. Barriers for source and destination are emitted automatically.
    pub fn copy_buffer(&mut self, src: Handle<Buffer>, dst: Handle<Buffer>) {
        if let Some(bar) = self.state.request_buffer_state(src, UsageBits::COPY_SRC) {
            self.push(Op::BufferBarrier, &bar);
        }
        if let Some(bar) = self.state.request_buffer_state(dst, UsageBits::COPY_DST) {
            self.push(Op::BufferBarrier, &bar);
        }
        let payload = CopyBuffer { src, dst };
        self.push(Op::CopyBuffer, &payload);
    }

    /// Copy data between images, emitting required barriers.
    pub fn copy_texture(
        &mut self,
        src: Handle<Image>,
        dst: Handle<Image>,
        range: SubresourceRange,
    ) {
        if let Some(bar) = self.state.request_texture_state(src, range, UsageBits::COPY_SRC) {
            self.push(Op::ImageBarrier, &bar);
        }
        if let Some(bar) = self.state.request_texture_state(dst, range, UsageBits::COPY_DST) {
            self.push(Op::ImageBarrier, &bar);
        }
        let payload = CopyImage { src, dst };
        self.push(Op::CopyImage, &payload);
    }

    /// Begin a debug marker region.
    pub fn begin_debug_marker(&mut self) {
        self.push(Op::DebugMarkerBegin, &DebugMarkerBegin {});
    }

    /// End a debug marker region.
    pub fn end_debug_marker(&mut self) {
        self.push(Op::DebugMarkerEnd, &DebugMarkerEnd {});
    }

    /// Submit the recorded commands to a backend context implementing [`CommandSink`].
    pub fn submit<S: CommandSink>(&self, sink: &mut S) {
        for cmd in self.iter() {
            match cmd.op {
                Op::BeginRenderPass => sink.begin_render_pass(cmd.payload()),
                Op::EndRenderPass => sink.end_render_pass(cmd.payload()),
                Op::BindPipeline => sink.bind_pipeline(cmd.payload()),
                Op::BindTable => sink.bind_table(cmd.payload()),
                Op::Draw => sink.draw(cmd.payload()),
                Op::Dispatch => sink.dispatch(cmd.payload()),
                Op::CopyBuffer => sink.copy_buffer(cmd.payload()),
                Op::CopyImage => sink.copy_texture(cmd.payload()),
                Op::ImageBarrier => sink.texture_barrier(cmd.payload()),
                Op::BufferBarrier => sink.buffer_barrier(cmd.payload()),
                Op::DebugMarkerBegin => sink.debug_marker_begin(cmd.payload()),
                Op::DebugMarkerEnd => sink.debug_marker_end(cmd.payload()),
            }
        }
    }

    /// Iterate over recorded commands.
    pub fn iter(&self) -> CommandIter {
        CommandIter { data: &self.data }
    }
}

/// Allow using [`CommandEncoder`] anywhere a [`CommandSink`] is expected. This enables
/// the new `gfx::cmd::CommandBuffer` front end to target both immediate command lists
/// and the intermediate representation encoded by `CommandEncoder`.
impl CommandSink for CommandEncoder {
    fn begin_render_pass(&mut self, pass: &BeginRenderPass) {
        let colors = &pass.colors[..pass.color_count as usize];
        let depth = if pass.has_depth != 0 { Some(pass.depth) } else { None };
        self.begin_render_pass(RenderPassDesc { colors, depth });
    }

    fn end_render_pass(&mut self, _pass: &EndRenderPass) {
        self.end_render_pass();
    }

    fn bind_pipeline(&mut self, cmd: &BindPipeline) {
        self.bind_pipeline(cmd.pipeline);
    }

    fn bind_table(&mut self, cmd: &BindTableCmd) {
        self.bind_table(cmd.table);
    }

    fn draw(&mut self, cmd: &Draw) {
        self.draw(cmd.vertex_count, cmd.instance_count);
    }

    fn dispatch(&mut self, cmd: &Dispatch) {
        self.dispatch(cmd.x, cmd.y, cmd.z);
    }

    fn copy_buffer(&mut self, cmd: &CopyBuffer) {
        self.copy_buffer(cmd.src, cmd.dst);
    }

    fn copy_texture(&mut self, cmd: &CopyImage) {
        // Without range information default to a single subresource.
        let range = SubresourceRange::new(0, 1, 0, 1);
        self.copy_texture(cmd.src, cmd.dst, range);
    }

    fn texture_barrier(&mut self, cmd: &ImageBarrier) {
        self.push(Op::ImageBarrier, cmd);
    }

    fn buffer_barrier(&mut self, cmd: &BufferBarrier) {
        self.push(Op::BufferBarrier, cmd);
    }

    fn debug_marker_begin(&mut self, _cmd: &DebugMarkerBegin) {
        self.begin_debug_marker();
    }

    fn debug_marker_end(&mut self, _cmd: &DebugMarkerEnd) {
        self.end_debug_marker();
    }
}

impl Default for CommandEncoder {
    fn default() -> Self {
        Self::new()
    }
}

//===----------------------------------------------------------------------===//
// Iteration
//===----------------------------------------------------------------------===//

pub struct Command<'a> {
    pub op: Op,
    bytes: &'a [u8],
}

impl<'a> Command<'a> {
    pub fn payload<T: Pod>(&self) -> &T {
        from_bytes(self.bytes)
    }
}

pub struct CommandIter<'a> {
    data: &'a [u8],
}

impl<'a> Iterator for CommandIter<'a> {
    type Item = Command<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        use core::mem::size_of;
        if self.data.len() < size_of::<CmdHeader>() {
            return None;
        }
        let (head_bytes, rest) = self.data.split_at(size_of::<CmdHeader>());
        let header: CmdHeader = *from_bytes(head_bytes);
        if rest.len() < header.size as usize {
            return None;
        }
        let (payload, remaining) = rest.split_at(header.size as usize);
        self.data = remaining;
        Some(Command { op: Op::from_u16(header.op).unwrap(), bytes: payload })
    }
}

impl Op {
    fn from_u16(v: u16) -> Option<Self> {
        match v {
            x if x == Op::BeginRenderPass as u16 => Some(Op::BeginRenderPass),
            x if x == Op::EndRenderPass as u16 => Some(Op::EndRenderPass),
            x if x == Op::BindPipeline as u16 => Some(Op::BindPipeline),
            x if x == Op::BindTable as u16 => Some(Op::BindTable),
            x if x == Op::Draw as u16 => Some(Op::Draw),
            x if x == Op::Dispatch as u16 => Some(Op::Dispatch),
            x if x == Op::CopyBuffer as u16 => Some(Op::CopyBuffer),
            x if x == Op::CopyImage as u16 => Some(Op::CopyImage),
            x if x == Op::ImageBarrier as u16 => Some(Op::ImageBarrier),
            x if x == Op::BufferBarrier as u16 => Some(Op::BufferBarrier),
            x if x == Op::DebugMarkerBegin as u16 => Some(Op::DebugMarkerBegin),
            x if x == Op::DebugMarkerEnd as u16 => Some(Op::DebugMarkerEnd),
            _ => None,
        }
    }
}

pub trait CommandSink {
    fn begin_render_pass(&mut self, pass: &BeginRenderPass);
    fn end_render_pass(&mut self, pass: &EndRenderPass);
    fn bind_pipeline(&mut self, cmd: &BindPipeline);
    fn bind_table(&mut self, cmd: &BindTableCmd);
    fn draw(&mut self, cmd: &Draw);
    fn dispatch(&mut self, cmd: &Dispatch);
    fn copy_buffer(&mut self, cmd: &CopyBuffer);
    fn copy_texture(&mut self, cmd: &CopyImage);
    fn texture_barrier(&mut self, cmd: &ImageBarrier);
    fn buffer_barrier(&mut self, cmd: &BufferBarrier);
    fn debug_marker_begin(&mut self, cmd: &DebugMarkerBegin);
    fn debug_marker_end(&mut self, cmd: &DebugMarkerEnd);
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

#[cfg(test)]
mod tests {
    use crate::Image;

    use super::*;

    #[test]
    fn barriers_are_emitted() {
        let mut enc = CommandEncoder::new();
        let tex = Handle::<Image>::new(1, 0);
        let range = SubresourceRange::new(0, 1, 0, 1);
        let color = ColorAttachment {
            handle: tex,
            range,
            clear: [0.0; 4],
            load: LoadOp::Load,
            store: StoreOp::Store,
            _pad: [0; 2],
        };
        enc.begin_render_pass(RenderPassDesc { colors: &[color], depth: None });
        enc.end_render_pass();
        let cmds: Vec<_> = enc.iter().collect();
        assert_eq!(cmds[0].op, Op::ImageBarrier);
        assert_eq!(cmds[1].op, Op::BeginRenderPass);
        assert_eq!(cmds[2].op, Op::EndRenderPass);
    }

    #[test]
    fn copy_buffer_emits_barriers() {
        let mut enc = CommandEncoder::new();
        let src = Handle::<Buffer>::new(2, 0);
        let dst = Handle::<Buffer>::new(3, 0);
        enc.copy_buffer(src, dst);
        let cmds: Vec<_> = enc.iter().collect();
        assert_eq!(cmds[0].op, Op::BufferBarrier);
        assert_eq!(cmds[1].op, Op::BufferBarrier);
        assert_eq!(cmds[2].op, Op::CopyBuffer);
    }

    #[test]
    fn round_trip() {
        let mut enc = CommandEncoder::new();
        let begin = BeginRenderPass { colors: [ColorAttachment::zeroed(); 4], color_count: 0, depth: DepthAttachment::zeroed(), has_depth: 0 };
        let end = EndRenderPass {};
        let bind = BindPipeline { pipeline: Handle::<Pipeline>::new(2, 0) };
        let draw = Draw { vertex_count: 3, instance_count: 1 };
        let dispatch = Dispatch { x: 1, y: 2, z: 3 };
        let copy = CopyBuffer { src: Handle::<Buffer>::new(4, 0), dst: Handle::<Buffer>::new(5, 0) };
        let img_barrier = ImageBarrier { image: Handle::<Image>::new(7, 0), range: SubresourceRange::new(0,1,0,1) };
        let buf_barrier = BufferBarrier { buffer: Handle::<Buffer>::new(9, 0) };
        let marker_begin = DebugMarkerBegin {};
        let marker_end = DebugMarkerEnd {};

        enc.push(Op::BeginRenderPass, &begin);
        enc.push(Op::EndRenderPass, &end);
        enc.push(Op::BindPipeline, &bind);
        enc.push(Op::Draw, &draw);
        enc.push(Op::Dispatch, &dispatch);
        enc.push(Op::CopyBuffer, &copy);
        enc.push(Op::ImageBarrier, &img_barrier);
        enc.push(Op::BufferBarrier, &buf_barrier);
        enc.push(Op::DebugMarkerBegin, &marker_begin);
        enc.push(Op::DebugMarkerEnd, &marker_end);

        let mut iter = enc.iter();

        let cmd1 = iter.next().unwrap();
        assert_eq!(cmd1.op, Op::BeginRenderPass);
        assert_eq!(*cmd1.payload::<BeginRenderPass>(), begin);

        let cmd2 = iter.next().unwrap();
        assert_eq!(cmd2.op, Op::EndRenderPass);
        assert_eq!(*cmd2.payload::<EndRenderPass>(), end);

        let cmd3 = iter.next().unwrap();
        assert_eq!(cmd3.op, Op::BindPipeline);
        assert_eq!(*cmd3.payload::<BindPipeline>(), bind);

        let cmd4 = iter.next().unwrap();
        assert_eq!(cmd4.op, Op::Draw);
        assert_eq!(*cmd4.payload::<Draw>(), draw);

        let cmd5 = iter.next().unwrap();
        assert_eq!(cmd5.op, Op::Dispatch);
        assert_eq!(*cmd5.payload::<Dispatch>(), dispatch);

        let cmd6 = iter.next().unwrap();
        assert_eq!(cmd6.op, Op::CopyBuffer);
        assert_eq!(*cmd6.payload::<CopyBuffer>(), copy);

        let cmd7 = iter.next().unwrap();
        assert_eq!(cmd7.op, Op::ImageBarrier);
        assert_eq!(*cmd7.payload::<ImageBarrier>(), img_barrier);

        let cmd8 = iter.next().unwrap();
        assert_eq!(cmd8.op, Op::BufferBarrier);
        assert_eq!(*cmd8.payload::<BufferBarrier>(), buf_barrier);

        let cmd9 = iter.next().unwrap();
        assert_eq!(cmd9.op, Op::DebugMarkerBegin);
        assert_eq!(*cmd9.payload::<DebugMarkerBegin>(), marker_begin);

        let cmd10 = iter.next().unwrap();
        assert_eq!(cmd10.op, Op::DebugMarkerEnd);
        assert_eq!(*cmd10.payload::<DebugMarkerEnd>(), marker_end);

        assert!(iter.next().is_none());
    }
}

