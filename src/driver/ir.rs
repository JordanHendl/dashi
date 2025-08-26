use bytemuck::{bytes_of, from_bytes, Pod, Zeroable};

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
    CopyTexture = 7,
    TextureBarrier = 8,
    BufferBarrier = 9,
    DebugMarkerBegin = 10,
    DebugMarkerEnd = 11,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct BeginRenderPass {
    pub color_attachments: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct EndRenderPass {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct BindPipeline {
    pub pipeline_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct BindTable {
    pub table_id: u32,
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
    pub src: u32,
    pub dst: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct CopyTexture {
    pub src: u32,
    pub dst: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct TextureBarrier {
    pub texture_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct BufferBarrier {
    pub buffer_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct DebugMarkerBegin {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq)]
pub struct DebugMarkerEnd {}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CmdHeader {
    op: u16,
    size: u16,
}

pub struct CommandStream {
    data: Vec<u8>,
}

impl CommandStream {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push<T: Pod>(&mut self, op: Op, payload: &T) {
        let header = CmdHeader { op: op as u16, size: core::mem::size_of::<T>() as u16 };
        self.data.extend_from_slice(bytes_of(&header));
        self.data.extend_from_slice(bytes_of(payload));
    }

    pub fn iter(&self) -> CommandIter {
        CommandIter { data: &self.data }
    }
}

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
            x if x == Op::CopyTexture as u16 => Some(Op::CopyTexture),
            x if x == Op::TextureBarrier as u16 => Some(Op::TextureBarrier),
            x if x == Op::BufferBarrier as u16 => Some(Op::BufferBarrier),
            x if x == Op::DebugMarkerBegin as u16 => Some(Op::DebugMarkerBegin),
            x if x == Op::DebugMarkerEnd as u16 => Some(Op::DebugMarkerEnd),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let mut stream = CommandStream::new();
        let begin = BeginRenderPass { color_attachments: 1 };
        let end = EndRenderPass {};
        let bind = BindPipeline { pipeline_id: 2 };
        let draw = Draw { vertex_count: 3, instance_count: 1 };
        let dispatch = Dispatch { x: 1, y: 2, z: 3 };
        let copy = CopyBuffer { src: 4, dst: 5 };
        let tex_barrier = TextureBarrier { texture_id: 7 };
        let buf_barrier = BufferBarrier { buffer_id: 9 };
        let marker_begin = DebugMarkerBegin {};
        let marker_end = DebugMarkerEnd {};

        stream.push(Op::BeginRenderPass, &begin);
        stream.push(Op::EndRenderPass, &end);
        stream.push(Op::BindPipeline, &bind);
        stream.push(Op::Draw, &draw);
        stream.push(Op::Dispatch, &dispatch);
        stream.push(Op::CopyBuffer, &copy);
        stream.push(Op::TextureBarrier, &tex_barrier);
        stream.push(Op::BufferBarrier, &buf_barrier);
        stream.push(Op::DebugMarkerBegin, &marker_begin);
        stream.push(Op::DebugMarkerEnd, &marker_end);

        let mut iter = stream.iter();

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
        assert_eq!(cmd7.op, Op::TextureBarrier);
        assert_eq!(*cmd7.payload::<TextureBarrier>(), tex_barrier);

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
