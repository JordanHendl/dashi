use std::collections::HashMap;

use super::{
    command::{BufferBarrier, TextureBarrier},
    types::{Buffer, Texture, UsageBits, Handle},
};

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct SubresourceRange {
    pub base_mip: u32,
    pub level_count: u32,
    pub base_layer: u32,
    pub layer_count: u32,
}

impl SubresourceRange {
    pub fn new(base_mip: u32, level_count: u32, base_layer: u32, layer_count: u32) -> Self {
        Self { base_mip, level_count, base_layer, layer_count }
    }
}

#[derive(Default)]
pub struct StateTracker {
    textures: HashMap<(Handle<Texture>, SubresourceRange), UsageBits>,
    buffers: HashMap<Handle<Buffer>, UsageBits>,
}

impl StateTracker {
    pub fn new() -> Self {
        Self { textures: HashMap::new(), buffers: HashMap::new() }
    }

    pub fn request_texture_state(
        &mut self,
        texture: Handle<Texture>,
        range: SubresourceRange,
        usage: UsageBits,
    ) -> Option<TextureBarrier> {
        let key = (texture, range);
        let current = self.textures.get(&key).copied().unwrap_or_default();
        if current != usage {
            self.textures.insert(key, usage);
            Some(TextureBarrier { texture, range })
        } else {
            None
        }
    }

    pub fn request_buffer_state(
        &mut self,
        buffer: Handle<Buffer>,
        usage: UsageBits,
    ) -> Option<BufferBarrier> {
        let current = self.buffers.get(&buffer).copied().unwrap_or_default();
        if current != usage {
            self.buffers.insert(buffer, usage);
            Some(BufferBarrier { buffer })
        } else {
            None
        }
    }
}

#[cfg(feature = "vulkan")]
pub mod vulkan {
    use super::*;
    use ash::vk;

    pub const USAGE_TO_LAYOUT: &[(UsageBits, vk::ImageLayout)] = &[
        (UsageBits::SAMPLED, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        (UsageBits::RT_WRITE, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
        (UsageBits::UAV_READ, vk::ImageLayout::GENERAL),
        (UsageBits::UAV_WRITE, vk::ImageLayout::GENERAL),
        (UsageBits::COPY_SRC, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        (UsageBits::COPY_DST, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
        (UsageBits::PRESENT, vk::ImageLayout::PRESENT_SRC_KHR),
        (UsageBits::DEPTH_READ, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL),
        (UsageBits::DEPTH_WRITE, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
    ];

    pub const USAGE_TO_STAGE: &[(UsageBits, vk::PipelineStageFlags)] = &[
        (UsageBits::SAMPLED, vk::PipelineStageFlags::FRAGMENT_SHADER),
        (UsageBits::RT_WRITE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT),
        (UsageBits::UAV_READ, vk::PipelineStageFlags::COMPUTE_SHADER),
        (UsageBits::UAV_WRITE, vk::PipelineStageFlags::COMPUTE_SHADER),
        (UsageBits::COPY_SRC, vk::PipelineStageFlags::TRANSFER),
        (UsageBits::COPY_DST, vk::PipelineStageFlags::TRANSFER),
        (UsageBits::PRESENT, vk::PipelineStageFlags::BOTTOM_OF_PIPE),
        (UsageBits::DEPTH_READ, vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS),
        (UsageBits::DEPTH_WRITE, vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS),
    ];

    pub const USAGE_TO_ACCESS: &[(UsageBits, vk::AccessFlags)] = &[
        (UsageBits::SAMPLED, vk::AccessFlags::SHADER_READ),
        (UsageBits::RT_WRITE, vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        (UsageBits::UAV_READ, vk::AccessFlags::SHADER_READ),
        (UsageBits::UAV_WRITE, vk::AccessFlags::SHADER_WRITE),
        (UsageBits::COPY_SRC, vk::AccessFlags::TRANSFER_READ),
        (UsageBits::COPY_DST, vk::AccessFlags::TRANSFER_WRITE),
        (UsageBits::PRESENT, vk::AccessFlags::empty()),
        (UsageBits::DEPTH_READ, vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ),
        (UsageBits::DEPTH_WRITE, vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_state_changes() {
        let mut tracker = StateTracker::new();
        let tex = Handle::<Texture>::new(1, 0);
        let range = SubresourceRange::new(0, 1, 0, 1);
        assert!(tracker.request_texture_state(tex, range, UsageBits::SAMPLED).is_some());
        assert!(tracker.request_texture_state(tex, range, UsageBits::SAMPLED).is_none());
        assert!(tracker.request_texture_state(tex, range, UsageBits::RT_WRITE).is_some());
    }

    #[test]
    fn buffer_state_changes() {
        let mut tracker = StateTracker::new();
        let buf = Handle::<Buffer>::new(1, 0);
        assert!(tracker.request_buffer_state(buf, UsageBits::COPY_SRC).is_some());
        assert!(tracker.request_buffer_state(buf, UsageBits::COPY_SRC).is_none());
        assert!(tracker.request_buffer_state(buf, UsageBits::COPY_DST).is_some());
    }
}
