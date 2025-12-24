use std::collections::HashMap;

use crate::structs::SubresourceRange;
use crate::{Buffer, Image, QueueType};

use super::types::{Handle, ResourceUse, UsageBits};

#[cfg(feature = "vulkan")]
pub type PipelineStage = ash::vk::PipelineStageFlags;
#[cfg(not(feature = "vulkan"))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct PipelineStage;

#[cfg(feature = "vulkan")]
pub type Access = ash::vk::AccessFlags;
#[cfg(not(feature = "vulkan"))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub struct Access;

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

#[inline]
pub fn resource_use_to_image_usage_layout(resource_use: ResourceUse) -> (UsageBits, Layout) {
    match resource_use {
        ResourceUse::CopySrc => (UsageBits::COPY_SRC, Layout::TransferSrc),
        ResourceUse::CopyDst => (UsageBits::COPY_DST, Layout::TransferDst),
        ResourceUse::Sampled => (UsageBits::SAMPLED, Layout::ShaderReadOnly),
        ResourceUse::StorageRead => (UsageBits::STORAGE_READ, Layout::General),
        ResourceUse::StorageWrite => (UsageBits::STORAGE_WRITE, Layout::General),
        ResourceUse::ColorAttachment => (UsageBits::RT_WRITE, Layout::ColorAttachment),
        ResourceUse::DepthAttachment => (UsageBits::DEPTH_WRITE, Layout::DepthStencilAttachment),
        ResourceUse::DepthRead => (UsageBits::DEPTH_READ, Layout::DepthStencilReadOnly),
        ResourceUse::Present => (UsageBits::PRESENT, Layout::Present),
        ResourceUse::VertexRead => (UsageBits::VERTEX_READ, Layout::General),
        ResourceUse::IndexRead => (UsageBits::INDEX_READ, Layout::General),
        ResourceUse::UniformRead => (UsageBits::UNIFORM_READ, Layout::General),
        ResourceUse::HostRead => (UsageBits::HOST_READ, Layout::General),
        ResourceUse::HostWrite => (UsageBits::HOST_WRITE, Layout::General),
        ResourceUse::ComputeShader => (UsageBits::COMPUTE_SHADER, Layout::General),
    }
}

#[inline]
pub fn resource_use_to_buffer_usage(resource_use: ResourceUse) -> UsageBits {
    match resource_use {
        ResourceUse::CopySrc => UsageBits::COPY_SRC,
        ResourceUse::CopyDst => UsageBits::COPY_DST,
        ResourceUse::Sampled => UsageBits::SAMPLED,
        ResourceUse::StorageRead => UsageBits::STORAGE_READ,
        ResourceUse::StorageWrite => UsageBits::STORAGE_WRITE,
        ResourceUse::ColorAttachment => UsageBits::RT_WRITE,
        ResourceUse::DepthAttachment => UsageBits::DEPTH_WRITE,
        ResourceUse::DepthRead => UsageBits::DEPTH_READ,
        ResourceUse::Present => UsageBits::PRESENT,
        ResourceUse::VertexRead => UsageBits::VERTEX_READ,
        ResourceUse::IndexRead => UsageBits::INDEX_READ,
        ResourceUse::UniformRead => UsageBits::UNIFORM_READ,
        ResourceUse::HostRead => UsageBits::HOST_READ,
        ResourceUse::HostWrite => UsageBits::HOST_WRITE,
        ResourceUse::ComputeShader => UsageBits::COMPUTE_SHADER,
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct LayoutTransition {
    pub image: Handle<Image>,
    pub range: SubresourceRange,
    pub old_usage: UsageBits,
    pub new_usage: UsageBits,
    pub old_stage: PipelineStage,
    pub new_stage: PipelineStage,
    pub old_access: Access,
    pub new_access: Access,
    pub old_layout: Layout,
    pub new_layout: Layout,
    pub old_queue: QueueType,
    pub new_queue: QueueType,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferBarrier {
    pub buffer: Handle<Buffer>,
    pub old_usage: UsageBits,
    pub new_usage: UsageBits,
    pub old_stage: PipelineStage,
    pub new_stage: PipelineStage,
    pub old_access: Access,
    pub new_access: Access,
    pub old_queue: QueueType,
    pub new_queue: QueueType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BufferState {
    usage: UsageBits,
    queue: QueueType,
    stage: PipelineStage,
    access: Access,
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
    if usage.intersects(
        UsageBits::UAV_READ
            | UsageBits::UAV_WRITE
            | UsageBits::STORAGE_READ
            | UsageBits::STORAGE_WRITE,
    ) {
        return Layout::General;
    }
    if usage.contains(UsageBits::SAMPLED) {
        return Layout::ShaderReadOnly;
    }
    Layout::General
}

#[cfg(feature = "vulkan")]
#[inline]
pub fn usage_to_stage(usage: UsageBits) -> PipelineStage {
    let mut flags = PipelineStage::empty();
    for (u, s) in vulkan::USAGE_TO_STAGE {
        if usage.contains(*u) {
            flags |= *s;
        }
    }
    if flags.is_empty() {
        flags = PipelineStage::TOP_OF_PIPE;
    }
    flags
}

#[cfg(not(feature = "vulkan"))]
#[inline]
pub fn usage_to_stage(_usage: UsageBits) -> PipelineStage {
    PipelineStage
}

#[cfg(feature = "vulkan")]
#[inline]
pub fn usage_to_access(usage: UsageBits) -> Access {
    let mut flags = Access::empty();
    for (u, a) in vulkan::USAGE_TO_ACCESS {
        if usage.contains(*u) {
            flags |= *a;
        }
    }
    flags
}

#[cfg(not(feature = "vulkan"))]
#[inline]
pub fn usage_to_access(_usage: UsageBits) -> Access {
    Access
}

#[derive(Default)]
pub struct StateTracker {
    images: HashMap<(Handle<Image>, SubresourceRange), UsageBits>,
    buffers: HashMap<Handle<Buffer>, BufferState>,
    image_layouts: HashMap<(Handle<Image>, SubresourceRange), Layout>,
    image_queues: HashMap<(Handle<Image>, SubresourceRange), QueueType>,
    image_stages: HashMap<(Handle<Image>, SubresourceRange), PipelineStage>,
    image_accesses: HashMap<(Handle<Image>, SubresourceRange), Access>,
}

impl StateTracker {
    pub fn new() -> Self {
        Self {
            images: HashMap::new(),
            image_layouts: HashMap::new(),
            buffers: HashMap::new(),
            image_queues: HashMap::new(),
            image_stages: HashMap::new(),
            image_accesses: HashMap::new(),
        }
    }

    pub fn reset(&mut self) {
        self.images.clear();
        self.buffers.clear();
        self.image_layouts.clear();
        self.image_queues.clear();
        self.image_stages.clear();
        self.image_accesses.clear();
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

        let old_stage = self
            .image_stages
            .get(&key)
            .copied()
            .unwrap_or_else(|| usage_to_stage(old_usage));
        let old_access = self
            .image_accesses
            .get(&key)
            .copied()
            .unwrap_or_else(|| usage_to_access(old_usage));

        let mut new_stage = usage_to_stage(new_usage);
        let mut new_access = usage_to_access(new_usage);

        // Persist tracker state.
        if usage_changed {
            self.images.insert(key, new_usage);
        } else {
            new_usage = old_usage;
            new_stage = old_stage;
            new_access = old_access;
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
        self.image_stages.insert(key, new_stage);
        self.image_accesses.insert(key, new_access);

        if usage_changed || layout_changed || queue_changed {
            Some(LayoutTransition {
                image,
                range,
                old_usage,
                new_usage,
                old_stage,
                new_stage,
                old_access,
                new_access,
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
        usage: UsageBits,
        queue: QueueType,
    ) -> Option<BufferBarrier> {
        let previous = self.buffers.get(&buffer).copied().unwrap_or(BufferState {
            usage: UsageBits::empty(),
            queue,
            stage: usage_to_stage(UsageBits::empty()),
            access: usage_to_access(UsageBits::empty()),
        });
        let usage_changed = previous.usage != usage;
        let queue_changed = previous.queue != queue;

        let stage = usage_to_stage(usage);
        let access = usage_to_access(usage);

        self.buffers.insert(
            buffer,
            BufferState {
                usage,
                queue,
                stage,
                access,
            },
        );

        if usage_changed || queue_changed {
            Some(BufferBarrier {
                buffer,
                old_usage: previous.usage,
                new_usage: usage,
                old_stage: previous.stage,
                new_stage: stage,
                old_access: previous.access,
                new_access: access,
                old_queue: previous.queue,
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
        (UsageBits::STORAGE_READ, vk::ImageLayout::GENERAL),
        (UsageBits::STORAGE_WRITE, vk::ImageLayout::GENERAL),
        (UsageBits::HOST_READ, vk::ImageLayout::GENERAL),
        (UsageBits::HOST_WRITE, vk::ImageLayout::GENERAL),
        (UsageBits::COMPUTE_SHADER, vk::ImageLayout::GENERAL),
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
        (UsageBits::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER),
        // When transitioning to PRESENT we still need to synchronize with the
        // producing stage (e.g., TRANSFER). Avoid BOTTOM_OF_PIPE, which is
        // incompatible with non-empty access masks.
        (UsageBits::PRESENT, vk::PipelineStageFlags::ALL_COMMANDS),
        (
            UsageBits::DEPTH_READ,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        ),
        (
            UsageBits::DEPTH_WRITE,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        ),
        (UsageBits::VERTEX_READ, vk::PipelineStageFlags::VERTEX_INPUT),
        (UsageBits::INDEX_READ, vk::PipelineStageFlags::VERTEX_INPUT),
        (
            UsageBits::UNIFORM_READ,
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
        (
            UsageBits::STORAGE_READ,
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
        (
            UsageBits::STORAGE_WRITE,
            vk::PipelineStageFlags::ALL_COMMANDS,
        ),
        (UsageBits::HOST_READ, vk::PipelineStageFlags::HOST),
        (UsageBits::HOST_WRITE, vk::PipelineStageFlags::HOST),
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
        (
            UsageBits::VERTEX_READ,
            vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
        ),
        (UsageBits::INDEX_READ, vk::AccessFlags::INDEX_READ),
        (UsageBits::UNIFORM_READ, vk::AccessFlags::UNIFORM_READ),
        (UsageBits::STORAGE_READ, vk::AccessFlags::SHADER_READ),
        (UsageBits::STORAGE_WRITE, vk::AccessFlags::SHADER_WRITE),
        (UsageBits::HOST_READ, vk::AccessFlags::HOST_READ),
        (UsageBits::HOST_WRITE, vk::AccessFlags::HOST_WRITE),
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
