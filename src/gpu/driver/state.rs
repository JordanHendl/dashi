use std::collections::HashMap;

use crate::{Buffer, Image, QueueType};

use super::{
    command::BufferBarrier,
    types::{Handle, UsageBits},
};

use bytemuck::{Pod, Zeroable};

// --- New: backend-agnostic image layout + transition info ---
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Layout {
    Undefined,
    General,
    ShaderReadOnly,
    ColorAttachment,
    DepthStencilAttachment,
    DepthStencilReadOnly,
    TransferSrc,
    TransferDst,
    Present,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LayoutTransition {
    pub image: Handle<Image>,
    pub range: SubresourceRange,
    pub old_usage: UsageBits,
    pub new_usage: UsageBits,
    pub old_layout: Layout,
    pub new_layout: Layout,
    pub old_queue: QueueType,
    pub new_queue: QueueType,
}

#[inline]
fn pick_layout_for_usage(usage: UsageBits) -> Layout {
    // Priority order: writes/attachments/present beat copies, which beat UAV/general, which beat sampled.
    // Adjust to taste for your engine’s rules.
    if usage.contains(UsageBits::PRESENT) {
        return Layout::Present;
    }
    if usage.contains(UsageBits::RT_WRITE) {
        return Layout::ColorAttachment;
    }
    if usage.contains(UsageBits::DEPTH_WRITE) {
        return Layout::DepthStencilAttachment;
    }
    if usage.contains(UsageBits::DEPTH_READ) {
        return Layout::DepthStencilReadOnly;
    }
    if usage.contains(UsageBits::COPY_DST) {
        return Layout::TransferDst;
    }
    if usage.contains(UsageBits::COPY_SRC) {
        return Layout::TransferSrc;
    }
    if usage.intersects(UsageBits::UAV_READ | UsageBits::UAV_WRITE) {
        return Layout::General;
    }
    if usage.contains(UsageBits::SAMPLED) {
        return Layout::ShaderReadOnly;
    }
    Layout::General
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct SubresourceRange {
    pub base_mip: u32,
    pub level_count: u32,
    pub base_layer: u32,
    pub layer_count: u32,
}

impl Default for SubresourceRange {
    fn default() -> Self {
        Self {
            base_mip: Default::default(),
            level_count: 1,
            base_layer: Default::default(),
            layer_count: 1,
        }
    }
}

impl SubresourceRange {
    pub fn new(base_mip: u32, level_count: u32, base_layer: u32, layer_count: u32) -> Self {
        Self {
            base_mip,
            level_count,
            base_layer,
            layer_count,
        }
    }
}

#[derive(Default)]
pub struct StateTracker {
    images: HashMap<(Handle<Image>, SubresourceRange), UsageBits>,
    buffers: HashMap<Handle<Buffer>, UsageBits>,
    image_layouts: HashMap<(Handle<Image>, SubresourceRange), Layout>,
    image_queues: HashMap<(Handle<Image>, SubresourceRange), QueueType>,
    buffer_queues: HashMap<Handle<Buffer>, QueueType>,
}

impl StateTracker {
    pub fn new() -> Self {
        Self {
            images: HashMap::new(),
            image_layouts: HashMap::new(),
            buffers: HashMap::new(),
            image_queues: HashMap::new(),
            buffer_queues: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        self.images.clear();
        self.buffers.clear();
        self.image_layouts.clear();
        self.image_queues.clear();
        self.buffer_queues.clear();
    }

    /// Returns (usage barrier if usage changed, layout transition if usage or layout changed).
    /// If only usage changed, `LayoutTransition` uses old_layout == new_layout.
    pub fn request_image_state(
        &mut self,
        image: Handle<Image>,
        range: SubresourceRange,
        mut new_usage: UsageBits,
        mut new_layout: Layout,
        queue: QueueType,
    ) -> Option<LayoutTransition> {
        let key = (image, range);

        let old_usage = self.images.get(&key).copied().unwrap_or_default();
        let usage_changed = old_usage != new_usage;

        let old_layout = self
            .image_layouts
            .get(&key)
            .copied()
            .unwrap_or(Layout::Undefined);
        let layout_changed = old_layout != new_layout;

        let old_queue = self.image_queues.get(&key).copied().unwrap_or(queue);
        let queue_changed = old_queue != queue;

        // Persist tracker state.
        if usage_changed {
            self.images.insert(key, new_usage);
        } else {
            new_usage = old_usage;
        }
        if layout_changed {
            self.image_layouts.insert(key, new_layout);
        } else {
            new_layout = old_layout;
            if old_layout == Layout::Undefined {
                self.image_layouts.insert(key, new_layout);
            }
        }
        if queue_changed {
            self.image_queues.insert(key, queue);
        } else {
            self.image_queues.entry(key).or_insert(queue);
        }

        if usage_changed || layout_changed || queue_changed {
            Some(LayoutTransition {
                image,
                range,
                old_usage,
                new_usage,
                old_layout,
                new_layout,
                old_queue,
                new_queue: queue,
            })
        } else {
            None
        }
    }

    pub fn request_buffer_state(
        &mut self,
        buffer: Handle<Buffer>,
        mut usage: UsageBits,
        queue: QueueType,
    ) -> Option<BufferBarrier> {
        let current = self.buffers.get(&buffer).copied().unwrap_or_default();
        let usage_changed = current != usage;
        let old_queue = self.buffer_queues.get(&buffer).copied().unwrap_or(queue);
        let queue_changed = old_queue != queue;

        if usage_changed {
            self.buffers.insert(buffer, usage);
        } else {
            usage = current;
        }

        if queue_changed {
            self.buffer_queues.insert(buffer, queue);
        } else {
            self.buffer_queues.entry(buffer).or_insert(queue);
        }

        if usage_changed || queue_changed {
            Some(BufferBarrier {
                buffer,
                usage,
                old_queue,
                new_queue: queue,
            })
        } else {
            None
        }
    }

    /// New: tell the tracker the current layout (e.g., after external/initial transitions).
    pub fn force_image_layout(
        &mut self,
        image: Handle<Image>,
        range: SubresourceRange,
        layout: Layout,
    ) {
        let key = (image, range);
        self.image_layouts.insert(key, layout);
    }

    /// New: query the last-known layout (Layout::Undefined if unknown).
    pub fn current_image_layout(&self, image: Handle<Image>, range: SubresourceRange) -> Layout {
        self.image_layouts
            .get(&(image, range))
            .copied()
            .unwrap_or(Layout::Undefined)
    }
}

#[cfg(feature = "vulkan")]
pub mod vulkan {
    use super::*;
    use ash::vk;

    pub const USAGE_TO_LAYOUT: &[(UsageBits, vk::ImageLayout)] = &[
        (
            UsageBits::SAMPLED,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ),
        (
            UsageBits::RT_WRITE,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ),
        (UsageBits::UAV_READ, vk::ImageLayout::GENERAL),
        (UsageBits::UAV_WRITE, vk::ImageLayout::GENERAL),
        (UsageBits::COPY_SRC, vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
        (UsageBits::COPY_DST, vk::ImageLayout::TRANSFER_DST_OPTIMAL),
        (UsageBits::PRESENT, vk::ImageLayout::PRESENT_SRC_KHR),
        (
            UsageBits::DEPTH_READ,
            vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        ),
        (
            UsageBits::DEPTH_WRITE,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ),
    ];

    pub const USAGE_TO_STAGE: &[(UsageBits, vk::PipelineStageFlags)] = &[
        (UsageBits::SAMPLED, vk::PipelineStageFlags::FRAGMENT_SHADER),
        (
            UsageBits::RT_WRITE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
        (UsageBits::UAV_READ, vk::PipelineStageFlags::COMPUTE_SHADER),
        (UsageBits::UAV_WRITE, vk::PipelineStageFlags::COMPUTE_SHADER),
        (UsageBits::COPY_SRC, vk::PipelineStageFlags::TRANSFER),
        (UsageBits::COPY_DST, vk::PipelineStageFlags::TRANSFER),
        (UsageBits::PRESENT, vk::PipelineStageFlags::BOTTOM_OF_PIPE),
        (
            UsageBits::DEPTH_READ,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        ),
        (
            UsageBits::DEPTH_WRITE,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        ),
    ];

    pub const USAGE_TO_ACCESS: &[(UsageBits, vk::AccessFlags)] = &[
        (UsageBits::SAMPLED, vk::AccessFlags::SHADER_READ),
        (UsageBits::RT_WRITE, vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        (UsageBits::UAV_READ, vk::AccessFlags::SHADER_READ),
        (UsageBits::UAV_WRITE, vk::AccessFlags::SHADER_WRITE),
        (UsageBits::COPY_SRC, vk::AccessFlags::TRANSFER_READ),
        (UsageBits::COPY_DST, vk::AccessFlags::TRANSFER_WRITE),
        (UsageBits::PRESENT, vk::AccessFlags::empty()),
        (
            UsageBits::DEPTH_READ,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
        ),
        (
            UsageBits::DEPTH_WRITE,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        ),
    ];
}

#[cfg(test)]
mod tests {
//
//    #[test]
//    fn image_state_changes() {
//        let mut tracker = StateTracker::new();
//        let tex = Handle::<Image>::new(1, 0);
//        let range = SubresourceRange::new(0, 1, 0, 1);
//        assert!(tracker
//            .request_image_state(tex, range, UsageBits::SAMPLED)
//            .is_some());
//        assert!(tracker
//            .request_image_state(tex, range, UsageBits::SAMPLED)
//            .is_none());
//        assert!(tracker
//            .request_image_state(tex, range, UsageBits::RT_WRITE)
//            .is_some());
//    }
//
//    #[test]
//    fn buffer_state_changes() {
//        let mut tracker = StateTracker::new();
//        let buf = Handle::<Buffer>::new(1, 0);
//        assert!(tracker
//            .request_buffer_state(buf, UsageBits::COPY_SRC)
//            .is_some());
//        assert!(tracker
//            .request_buffer_state(buf, UsageBits::COPY_SRC)
//            .is_none());
//        assert!(tracker
//            .request_buffer_state(buf, UsageBits::COPY_DST)
//            .is_some());
//    }
}
