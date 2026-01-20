use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};
pub use resource_pool::Handle;

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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ResourceUse {
    CopySrc,
    CopyDst,
    Sampled,
    StorageRead,
    StorageWrite,
    ColorAttachment,
    DepthAttachment,
    DepthRead,
    Present,
    VertexRead,
    IndexRead,
    IndirectRead,
    UniformRead,
    HostRead,
    HostWrite,
    ComputeShader,
}

bitflags! {
    #[repr(C)]
    #[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
        const VERTEX_READ = 0x200;
        const INDEX_READ  = 0x400;
        const UNIFORM_READ = 0x800;
        const STORAGE_READ = 0x1000;
        const STORAGE_WRITE = 0x2000;
        const HOST_READ = 0x4000;
        const HOST_WRITE = 0x8000;
        const COMPUTE_SHADER = 0x9000;
        const INDIRECT_READ = 0x10000;
    }
}

unsafe impl Pod for UsageBits {}
unsafe impl Zeroable for UsageBits {}
