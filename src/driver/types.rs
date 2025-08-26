use bitflags::bitflags;

#[repr(transparent)]
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PipelineHandle(pub u32);

#[repr(transparent)]
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BufferHandle(pub u32);

#[repr(transparent)]
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TextureHandle(pub u32);

#[repr(transparent)]
#[derive(Default, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct BindTableHandle(pub u32);

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Format {
    Unknown,
    R8Uint,
    R8Sint,
    RGBA8,
    RGBA8Unorm,
    RGBA32Float,
    BGRA8Unorm,
    D24S8,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum IndexType {
    U16,
    U32,
}

bitflags! {
    #[derive(Default)]
    pub struct UsageBits: u32 {
        const SAMPLED     = 0x1;
        const RT_WRITE    = 0x2;
        const UAV_READ    = 0x4;
        const UAV_WRITE   = 0x8;
        const COPY_SRC    = 0x10;
        const COPY_DST    = 0x20;
        const PRESENT     = 0x40;
        const DEPTH_READ  = 0x80;
        const DEPTH_WRITE = 0x100;
    }
}

