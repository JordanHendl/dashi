use bytemuck::{Pod, Zeroable};
use core::convert::TryInto;

use crate::{
    BindGroup, BindTable, Buffer, ClearValue, ComputePipeline, DynamicBuffer, Fence, Filter,
    GraphicsPipeline, Image, ImageView, Rect2D, SubmitInfo2, Viewport,
};

use super::{
    state::{Layout, LayoutTransition, StateTracker, SubresourceRange},
    types::{Handle, UsageBits},
};

//===----------------------------------------------------------------------===//
// Command definitions
//===----------------------------------------------------------------------===//

#[repr(u16)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Op {
    BeginDrawing = 0,
    EndDrawing = 1,
    BindGraphicsPipeline = 2,
    Draw = 3,
    DrawIndexed = 4,
    DrawIndirect = 5,
    Dispatch = 6,
    DispatchIndirect = 7,
    CopyBuffer = 8,
    CopyBufferToImage = 9,
    CopyImageToBuffer = 10,
    CopyImage = 11,
    BlitImage = 12,
    ImageBarrier = 13,
    BufferBarrier = 14,
    DebugMarkerBegin = 15,
    DebugMarkerEnd = 16,
    TransitionImage = 17,
}

fn align_up(v: usize, a: usize) -> usize {
    debug_assert!(a.is_power_of_two());
    (v + (a - 1)) & !(a - 1)
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
#[derive(Clone, Copy, Debug)]
pub struct BeginDrawing {
    pub viewport: Viewport,
    pub pipeline: Handle<GraphicsPipeline>,
    pub colors: [Option<ImageView>; 4],
    pub depth: Option<ImageView>,
    pub color_clears: [Option<ClearValue>; 4],
    pub depth_clear: Option<ClearValue>,
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct EndDrawing {
    padding: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct BindGraphicsPipeline {
    pub pipeline: Handle<GraphicsPipeline>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Draw {
    /// Vertex buffer handle.
    pub vertices: Handle<Buffer>,
    /// Resources to bind before drawing.
    pub bind_groups: [Option<Handle<BindGroup>>; 4],
    pub bind_tables: [Option<Handle<BindTable>>; 4],
    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
    /// Number of instances to draw.
    pub instance_count: u32,
    /// Number of vertices to draw.
    pub count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DrawIndexed {
    /// Vertex buffer handle.
    pub vertices: Handle<Buffer>,
    pub indices: Handle<Buffer>,
    /// Resources to bind before drawing.
    pub bind_groups: [Option<Handle<BindGroup>>; 4],
    pub bind_tables: [Option<Handle<BindTable>>; 4],
    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
    /// Number of instances to draw.
    pub instance_count: u32,
    pub first_instance: u32,
    /// Number of indices to draw.
    pub index_count: u32,
}

impl Default for DrawIndexed {
    fn default() -> Self {
        Self {
            vertices: Default::default(),
            indices: Default::default(),
            index_count: Default::default(),
            instance_count: 1,
            first_instance: 0,
            bind_groups: Default::default(),
            bind_tables: Default::default(),
            dynamic_buffers: Default::default(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dispatch {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub pipeline: Handle<ComputePipeline>,
    pub bind_groups: [Option<Handle<BindGroup>>; 4],
    pub bind_tables: [Option<Handle<BindTable>>; 4],
    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq, Default)]
pub struct CopyBuffer {
    pub src: Handle<Buffer>,
    pub dst: Handle<Buffer>,
    pub src_offset: u32,
    pub dst_offset: u32,
    pub amount: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq, Default)]
pub struct CopyBufferImage {
    pub src: Handle<Buffer>,
    pub dst: Handle<Image>,
    pub range: SubresourceRange,
    pub src_offset: u32,
    pub amount: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq, Default)]
pub struct CopyImageBuffer {
    pub src: Handle<Image>,
    pub dst: Handle<Buffer>,
    pub range: SubresourceRange,
    pub dst_offset: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct CopyImage {
    pub src: Handle<Image>,
    pub dst: Handle<Image>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct BlitPadding {
    p: [u8; 52],
}

impl Default for BlitPadding {
    fn default() -> Self {
        return unsafe { std::mem::zeroed() };
    }
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct BlitImage {
    pub src: Handle<Image>,
    pub dst: Handle<Image>,
    pub src_range: SubresourceRange,
    pub dst_range: SubresourceRange,
    pub filter: Filter,
    /// Region in the source image.
    pub src_region: Rect2D,
    /// Region in the destination image.
    pub dst_region: Rect2D,
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
    pub usage: UsageBits,
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

/// A single structure that both records commands and stores them in per-frame arenas.
///
/// Payloads up to 32 bytes are stored inline with the command header while larger
/// payloads are written into a side buffer and referenced by index.  This keeps the
/// hot path free of heap allocations and allows the encoder to be reused across
/// frames without reallocating.
pub struct CommandEncoder {
    data: Vec<u8>,
    side: Vec<u8>,
    state: StateTracker,
    curr_colors: [Option<ImageView>; 4],
    curr_depth: Option<ImageView>,
}

impl CommandEncoder {
    /// Create a new empty encoder. Some initial capacity is reserved to avoid
    /// frequent reallocations when recording many commands.
    pub fn new() -> Self {
        Self {
            data: Vec::with_capacity(1024),
            side: Vec::with_capacity(256),
            state: StateTracker::new(),
            curr_colors: [None; 4],
            curr_depth: None,
        }
    }

    /// Clear all recorded commands while retaining allocated arenas.
    pub fn reset(&mut self) {
        self.data.clear();
        self.side.clear();
        self.state.reset();
        self.curr_colors = [None; 4];
        self.curr_depth = None;
    }

    /// Push a command payload into the internal byte stream. The method is marked
    /// `inline(always)` so that callers can be optimized tightly.
    #[inline(always)]
    fn push<T: std::fmt::Debug>(&mut self, op: Op, payload: &T) {
        const INLINE_ALIGN: usize = 8;
        const SIDE_ALIGN: usize = 8;
        const INLINE_MAX: usize = 32;
        const SIDE_FLAG: u16 = 0x8000;

        let size = size_of::<T>();
        let p = unsafe { core::slice::from_raw_parts(payload as *const T as *const u8, size) };

        if size <= INLINE_MAX && align_of::<T>() <= INLINE_ALIGN {
            // Write header (op, size)
            self.data.extend_from_slice(&(op as u16).to_ne_bytes());
            self.data.extend_from_slice(&(size as u16).to_ne_bytes());

            // Make sure we won't reallocate while padding+writing payload
            // (realloc would change the base address after we compute it).
            self.data.reserve(INLINE_ALIGN - 1 + size);

            // Compute padding based on the *absolute* address (base + current len)
            let base = self.data.as_ptr() as usize;
            let here = base + self.data.len();
            let start = align_up(here, INLINE_ALIGN);
            let pad = start - here;

            if pad != 0 {
                self.data.resize(self.data.len() + pad, 0);
            }

            // Write payload
            self.data.extend_from_slice(p);
        } else {
            // Side path: header with SIDE_FLAG and u32 offset to side blob.
            let header_size = SIDE_FLAG;
            self.data.extend_from_slice(&(op as u16).to_ne_bytes());
            self.data.extend_from_slice(&header_size.to_ne_bytes());

            // Offset (where [len][pad][payload] starts in side)
            let offset = self.side.len() as u32;
            self.data.extend_from_slice(&offset.to_ne_bytes());

            // Side blob: [u32 len], pad to SIDE_ALIGN, then payload
            let len_u32: u32 = size as u32;
            self.side.extend_from_slice(&len_u32.to_ne_bytes());

            // Ensure capacity so base won't change during padding+payload write
            self.side.reserve(SIDE_ALIGN - 1 + size);

            let base = self.side.as_ptr() as usize;
            let here = base + self.side.len();
            let start = align_up(here, SIDE_ALIGN);
            let pad = start - here;

            if pad != 0 {
                self.side.resize(self.side.len() + pad, 0);
            }

            self.side.extend_from_slice(p);
        }
    }

    /// Begin a render pass with the provided attachments.
    pub fn begin_drawing(&mut self, desc: &BeginDrawing) {
        self.curr_colors = desc.colors;
        self.curr_depth = desc.depth;
        for view in desc.colors.iter().flatten() {
            let range = SubresourceRange::new(view.mip_level, 1, view.layer, 1);
            if let Some(bar) =
                self.state
                    .request_image_state(view.img, range, UsageBits::RT_WRITE, Layout::ColorAttachment)
            {
                self.push(Op::ImageBarrier, &bar);
            }
        }
        if let Some(view) = desc.depth {
            let range = SubresourceRange::new(view.mip_level, 1, view.layer, 1);
            if let Some(bar) =
                self.state
                    .request_image_state(view.img, range, UsageBits::DEPTH_WRITE, Layout::DepthStencilAttachment)
            {
                self.push(Op::ImageBarrier, &bar);
            }
        }
        self.push(Op::BeginDrawing, desc);
    }

    /// End the current render pass.
    pub fn end_drawing(&mut self) {
        let colors = self.curr_colors;
        for view in colors.into_iter().flatten() {
            let range = SubresourceRange::new(view.mip_level, 1, view.layer, 1);
            if let Some(bar) = self
                .state
                .request_image_state(view.img, range, UsageBits::SAMPLED, Layout::ShaderReadOnly)
            {
                self.push(Op::ImageBarrier, &bar);
            }
        }
        if let Some(view) = self.curr_depth {
            let range = SubresourceRange::new(view.mip_level, 1, view.layer, 1);
            if let Some(bar) = self
                .state
                .request_image_state(view.img, range, UsageBits::DEPTH_READ, Layout::DepthStencilReadOnly)
            {
                self.push(Op::ImageBarrier, &bar);
            }
        }
        self.push(Op::EndDrawing, &EndDrawing::default());
    }

    /// Bind a graphics or compute pipeline.
    pub fn bind_graphics_pipeline(&mut self, pipe: Handle<GraphicsPipeline>) {
        let payload = BindGraphicsPipeline { pipeline: pipe };
        self.push(Op::BindGraphicsPipeline, &payload);
    }

    /// Issue a draw call.
    pub fn draw(&mut self, cmd: &Draw) {
        self.push(Op::Draw, cmd);
    }

    pub fn draw_indexed(&mut self, cmd: &DrawIndexed) {
        self.push(Op::DrawIndexed, cmd);
    }

    /// Issue a dispatch call.
    pub fn dispatch(&mut self, cmd: &Dispatch) {
        self.push(Op::Dispatch, cmd);
    }

    /// Copy data between buffers. Barriers for source and destination are emitted automatically.
    pub fn copy_buffer(&mut self, cmd: &CopyBuffer) {
        if let Some(bar) = self
            .state
            .request_buffer_state(cmd.src, UsageBits::COPY_SRC)
        {
            self.push(Op::BufferBarrier, &bar);
        }
        if let Some(bar) = self
            .state
            .request_buffer_state(cmd.dst, UsageBits::COPY_DST)
        {
            self.push(Op::BufferBarrier, &bar);
        }
        self.push(Op::CopyBuffer, cmd);
    }

    /// Copy data between buffers. Barriers for source and destination are emitted automatically.
    pub fn copy_buffer_to_image(&mut self, cmd: &CopyBufferImage) {
        if let Some(bar) = self
            .state
            .request_buffer_state(cmd.src, UsageBits::COPY_SRC)
        {
            self.push(Op::BufferBarrier, &bar);
        }
        if let Some(bar) = self.state.request_image_state(
            cmd.dst,
            cmd.range,
            UsageBits::COPY_DST,
            Layout::TransferDst,
        ) {
            self.push(Op::ImageBarrier, &bar);
        }
        self.push(Op::CopyBufferToImage, cmd);
    }

    /// Copy data between buffers. Barriers for source and destination are emitted automatically.
    pub fn copy_image_to_buffer(&mut self, cmd: &CopyImageBuffer) {
        if let Some(bar) = self.state.request_image_state(
            cmd.src,
            cmd.range,
            UsageBits::COPY_SRC,
            Layout::TransferSrc,
        ) {
            self.push(Op::BufferBarrier, &bar);
        }
        if let Some(bar) = self
            .state
            .request_buffer_state(cmd.dst, UsageBits::COPY_DST)
        {
            self.push(Op::ImageBarrier, &bar);
        }
        self.push(Op::CopyImageToBuffer, cmd);
    }

    /// Copy data between images, emitting required barriers.
    pub fn copy_image(&mut self, src: Handle<Image>, dst: Handle<Image>, range: SubresourceRange) {
        if let Some(bar) =
            self.state
                .request_image_state(src, range, UsageBits::COPY_SRC, Layout::TransferSrc)
        {
            self.push(Op::ImageBarrier, &bar);
        }
        if let Some(bar) =
            self.state
                .request_image_state(dst, range, UsageBits::COPY_DST, Layout::TransferDst)
        {
            self.push(Op::ImageBarrier, &bar);
        }
        let payload = CopyImage { src, dst };
        self.push(Op::CopyImage, &payload);
    }

    /// Copy data between images, emitting required barriers.
    pub fn blit_image(&mut self, cmd: &BlitImage) {
        if let Some(bar) = self.state.request_image_state(
            cmd.src,
            cmd.src_range,
            UsageBits::COPY_SRC,
            Layout::TransferSrc,
        ) {
            self.push(Op::ImageBarrier, &bar);
        }
        if let Some(bar) = self.state.request_image_state(
            cmd.dst,
            cmd.dst_range,
            UsageBits::COPY_DST,
            Layout::TransferDst,
        ) {
            self.push(Op::ImageBarrier, &bar);
        }
        self.push(Op::BlitImage, cmd);
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
    pub fn append<S: CommandSink>(&self, sink: &mut S) -> usize {
        let mut cnt = 0;
        for cmd in self.iter() {
            cnt += 1;
            match cmd.op {
                Op::BeginDrawing => sink.begin_drawing(cmd.payload()),
                Op::EndDrawing => sink.end_drawing(cmd.payload()),
                Op::BindGraphicsPipeline => sink.bind_graphics_pipeline(cmd.payload()),
                Op::Draw => sink.draw(cmd.payload()),
                Op::Dispatch => sink.dispatch(cmd.payload()),
                Op::CopyBuffer => sink.copy_buffer(cmd.payload()),
                Op::CopyBufferToImage => sink.copy_buffer_to_image(cmd.payload()),
                Op::CopyImage => sink.copy_image(cmd.payload()),
                Op::ImageBarrier => sink.image_barrier(cmd.payload()),
                Op::BufferBarrier => sink.buffer_barrier(cmd.payload()),
                Op::DebugMarkerBegin => sink.debug_marker_begin(cmd.payload()),
                Op::DebugMarkerEnd => sink.debug_marker_end(cmd.payload()),
                Op::CopyImageToBuffer => sink.copy_image_to_buffer(cmd.payload()),
                Op::BlitImage => sink.blit_image(cmd.payload()),
                Op::DrawIndexed => sink.draw_indexed(cmd.payload()),
                Op::DrawIndirect => todo!(),
                Op::DispatchIndirect => todo!(),
                Op::TransitionImage => todo!(),
            }
        }
        cnt
    }

    /// Submit the recorded commands to a backend context implementing [`CommandSink`].
    pub fn submit<S: CommandSink>(
        &self,
        sink: &mut S,
        submit: &SubmitInfo2,
    ) -> Option<Handle<Fence>> {
        if self.append(sink) != 0 {
            return Some(sink.submit(submit));
        }

        None
    }

    /// Iterate over recorded commands.
    pub fn iter(&self) -> CommandIter {
        CommandIter {
            data: &self.data,
            side: &self.side,
        }
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
    pub fn payload<T>(&self) -> &T {

        assert_eq!(
            self.bytes.len(),
            size_of::<T>(),
            "payload<T>: wrong byte length"
        );
        let addr = self.bytes.as_ptr() as usize;
        assert_eq!(
            addr % align_of::<T>(),
            0,
            "payload<T>: misaligned buffer for T"
        );
        unsafe { &*(self.bytes.as_ptr() as *const T) }
    }
}

pub struct CommandIter<'a> {
    data: &'a [u8],
    side: &'a [u8],
}

impl<'a> Iterator for CommandIter<'a> {
    type Item = Command<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        use core::mem::size_of;
        const INLINE_ALIGN: usize = 8;
        const SIDE_ALIGN: usize = 8;
        const INLINE_MAX: usize = 32;
        const SIDE_FLAG: u16 = 0x8000;

        // Need 4 bytes for header (u16 op, u16 size)
        if self.data.len() < size_of::<u16>() * 2 {
            return None;
        }

        // Parse header
        let (head, rest0) = self.data.split_at(4);
        let op = u16::from_ne_bytes([head[0], head[1]]);
        let size = u16::from_ne_bytes([head[2], head[3]]);

        if (size & SIDE_FLAG) != 0 {
            // SIDE record: expect u32 offset next
            if rest0.len() < size_of::<u32>() {
                return None;
            }
            let (off_bytes, remaining) = rest0.split_at(4);
            self.data = remaining;

            let offset = u32::from_ne_bytes(off_bytes.try_into().ok()?) as usize;
            if self.side.len() < offset + 4 {
                return None;
            }

            // New-format side record: [u32 len][pad->SIDE_ALIGN][payload...]
            let len = u32::from_ne_bytes(self.side[offset..offset + 4].try_into().ok()?) as usize;

            let side_base = self.side.as_ptr() as usize;
            let after_len = side_base + offset + 4;
            let payloadaddr = align_up(after_len, SIDE_ALIGN);
            let payload_off = payloadaddr - side_base;
            let end = payload_off.checked_add(len)?;

            if end > self.side.len() {
                return None;
            }

            let payload = &self.side[payload_off..end];
            Some(Command {
                op: Op::from_u16(op).unwrap(),
                bytes: payload,
            })
        } else {
            // INLINE record: size = payload bytes, payload starts at INLINE_ALIGN
            let payload_len = size as usize;

            // Inline payload location is based on the absolute address of 'rest0'
            let base_addr = rest0.as_ptr() as usize;
            let payload_addr = align_up(base_addr, INLINE_ALIGN);
            let pad = payload_addr - base_addr;

            if rest0.len() < pad + payload_len {
                return None;
            }

            let (pad_plus_payload, remaining) = rest0.split_at(pad + payload_len);
            self.data = remaining;

            let payload = &pad_plus_payload[pad..];
            Some(Command {
                op: Op::from_u16(op).unwrap(),
                bytes: payload,
            })
        }
    }
}

impl Op {
    fn from_u16(v: u16) -> Option<Self> {
        match v {
            x if x == Op::BeginDrawing as u16 => Some(Op::BeginDrawing),
            x if x == Op::EndDrawing as u16 => Some(Op::EndDrawing),
            x if x == Op::BindGraphicsPipeline as u16 => Some(Op::BindGraphicsPipeline),
            x if x == Op::Draw as u16 => Some(Op::Draw),
            x if x == Op::DrawIndexed as u16 => Some(Op::DrawIndexed),
            x if x == Op::DrawIndirect as u16 => Some(Op::DrawIndirect),
            x if x == Op::Dispatch as u16 => Some(Op::Dispatch),
            x if x == Op::CopyBuffer as u16 => Some(Op::CopyBuffer),
            x if x == Op::CopyBufferToImage as u16 => Some(Op::CopyBufferToImage),
            x if x == Op::CopyImageToBuffer as u16 => Some(Op::CopyImageToBuffer),
            x if x == Op::CopyImage as u16 => Some(Op::CopyImage),
            x if x == Op::BlitImage as u16 => Some(Op::BlitImage),
            x if x == Op::ImageBarrier as u16 => Some(Op::ImageBarrier),
            x if x == Op::BufferBarrier as u16 => Some(Op::BufferBarrier),
            x if x == Op::DebugMarkerBegin as u16 => Some(Op::DebugMarkerBegin),
            x if x == Op::DebugMarkerEnd as u16 => Some(Op::DebugMarkerEnd),
            _ => None,
        }
    }
}

pub trait CommandSink {
    fn begin_drawing(&mut self, pass: &BeginDrawing);
    fn end_drawing(&mut self, pass: &EndDrawing);
    fn bind_graphics_pipeline(&mut self, cmd: &BindGraphicsPipeline);
    fn blit_image(&mut self, cmd: &BlitImage);
    fn draw(&mut self, cmd: &Draw);
    fn draw_indexed(&mut self, cmd: &DrawIndexed);
    fn dispatch(&mut self, cmd: &Dispatch);
    fn copy_buffer(&mut self, cmd: &CopyBuffer);
    fn copy_buffer_to_image(&mut self, cmd: &CopyBufferImage);
    fn copy_image_to_buffer(&mut self, cmd: &CopyImageBuffer);
    fn copy_image(&mut self, cmd: &CopyImage);
    #[deprecated(note = "renamed to copy_image")]
    /// Deprecated: renamed to [`copy_image`].
    fn copy_texture(&mut self, cmd: &CopyImage) {
        self.copy_image(cmd)
    }
    fn image_barrier(&mut self, cmd: &LayoutTransition);
    fn buffer_barrier(&mut self, cmd: &BufferBarrier);
    fn submit(&mut self, cmd: &SubmitInfo2) -> Handle<Fence>;
    fn debug_marker_begin(&mut self, cmd: &DebugMarkerBegin);
    fn debug_marker_end(&mut self, cmd: &DebugMarkerEnd);
}

//===----------------------------------------------------------------------===//
// Tests
//===----------------------------------------------------------------------===//

#[cfg(test)]
mod tests {}
