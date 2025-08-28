use ash::vk;
use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};

bitflags! {
    #[repr(transparent)]
    #[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Access: u64 {
        const NONE = 0;
        const INDIRECT_COMMAND_READ = vk::AccessFlags2::INDIRECT_COMMAND_READ.as_raw();
        const INDEX_READ = vk::AccessFlags2::INDEX_READ.as_raw();
        const VERTEX_ATTRIBUTE_READ = vk::AccessFlags2::VERTEX_ATTRIBUTE_READ.as_raw();
        const UNIFORM_READ = vk::AccessFlags2::UNIFORM_READ.as_raw();
        const INPUT_ATTACHMENT_READ = vk::AccessFlags2::INPUT_ATTACHMENT_READ.as_raw();
        const SHADER_READ = vk::AccessFlags2::SHADER_READ.as_raw();
        const SHADER_WRITE = vk::AccessFlags2::SHADER_WRITE.as_raw();
        const COLOR_ATTACHMENT_READ = vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw();
        const COLOR_ATTACHMENT_WRITE = vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw();
        const DEPTH_STENCIL_ATTACHMENT_READ = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw();
        const DEPTH_STENCIL_ATTACHMENT_WRITE = vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw();
        const TRANSFER_READ = vk::AccessFlags2::TRANSFER_READ.as_raw();
        const TRANSFER_WRITE = vk::AccessFlags2::TRANSFER_WRITE.as_raw();
        const HOST_READ = vk::AccessFlags2::HOST_READ.as_raw();
        const HOST_WRITE = vk::AccessFlags2::HOST_WRITE.as_raw();
    }
}
unsafe impl Zeroable for Access {}
unsafe impl Pod for Access {}

bitflags! {
    #[repr(transparent)]
    #[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Stage: u64 {
        const TOP_OF_PIPE = vk::PipelineStageFlags2::TOP_OF_PIPE.as_raw();
        const DRAW_INDIRECT = vk::PipelineStageFlags2::DRAW_INDIRECT.as_raw();
        const VERTEX_INPUT = vk::PipelineStageFlags2::VERTEX_INPUT.as_raw();
        const VERTEX_SHADER = vk::PipelineStageFlags2::VERTEX_SHADER.as_raw();
        const FRAGMENT_SHADER = vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw();
        const COMPUTE_SHADER = vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw();
        const EARLY_FRAGMENT_TESTS = vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw();
        const LATE_FRAGMENT_TESTS = vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw();
        const COLOR_ATTACHMENT_OUTPUT = vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw();
        const TRANSFER = vk::PipelineStageFlags2::TRANSFER.as_raw();
        const BOTTOM_OF_PIPE = vk::PipelineStageFlags2::BOTTOM_OF_PIPE.as_raw();
    }
}
unsafe impl Zeroable for Stage {}
unsafe impl Pod for Stage {}

#[repr(transparent)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Pod, Zeroable, Default)]
pub struct ImageLayout(pub i32);

impl From<ImageLayout> for vk::ImageLayout {
    fn from(layout: ImageLayout) -> Self {
        vk::ImageLayout::from_raw(layout.0)
    }
}

impl From<vk::ImageLayout> for ImageLayout {
    fn from(layout: vk::ImageLayout) -> Self {
        Self(layout.as_raw())
    }
}

impl From<Access> for vk::AccessFlags2 {
    fn from(acc: Access) -> Self {
        vk::AccessFlags2::from_raw(acc.bits())
    }
}

impl From<Stage> for vk::PipelineStageFlags2 {
    fn from(stage: Stage) -> Self {
        vk::PipelineStageFlags2::from_raw(stage.bits())
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default, Pod, Zeroable)]
pub struct ResState {
    pub access: Access,
    pub stages: Stage,
    pub layout: ImageLayout,
    pub _pad: u32,
}
