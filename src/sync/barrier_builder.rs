use ash::vk;
use smallvec::SmallVec;

use crate::Handle;
use crate::sync::state::ResState;
use crate::gpu::vulkan::{Image, Buffer};

pub trait ResourceLookup {
    fn image_raw(&self, image: Handle<Image>) -> vk::Image;
    fn buffer_raw(&self, buffer: Handle<Buffer>) -> vk::Buffer;
}

pub struct BarrierBuilder<'a, R: ResourceLookup> {
    lookup: &'a R,
    images: SmallVec<[vk::ImageMemoryBarrier2; 4]>,
    buffers: SmallVec<[vk::BufferMemoryBarrier2; 4]>,
}

impl<'a, R: ResourceLookup> BarrierBuilder<'a, R> {
    pub fn new(lookup: &'a R) -> Self {
        Self {
            lookup,
            images: SmallVec::new(),
            buffers: SmallVec::new(),
        }
    }

    pub fn image(&mut self, image: Handle<Image>, src: ResState, dst: ResState) {
        if src == dst { return; }
        let raw = self.lookup.image_raw(image);
        let barrier = vk::ImageMemoryBarrier2 {
            src_stage_mask: src.stages.into(),
            src_access_mask: src.access.into(),
            dst_stage_mask: dst.stages.into(),
            dst_access_mask: dst.access.into(),
            old_layout: src.layout.into(),
            new_layout: dst.layout.into(),
            image: raw,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            },
            ..Default::default()
        };
        self.images.push(barrier);
    }

    pub fn buffer(&mut self, buffer: Handle<Buffer>, src: ResState, dst: ResState) {
        if src == dst { return; }
        let raw = self.lookup.buffer_raw(buffer);
        let barrier = vk::BufferMemoryBarrier2 {
            src_stage_mask: src.stages.into(),
            src_access_mask: src.access.into(),
            dst_stage_mask: dst.stages.into(),
            dst_access_mask: dst.access.into(),
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: raw,
            offset: 0,
            size: vk::WHOLE_SIZE,
            ..Default::default()
        };
        self.buffers.push(barrier);
    }

    pub unsafe fn emit(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        if self.images.is_empty() && self.buffers.is_empty() {
            return;
        }
        let deps = vk::DependencyInfo::builder()
            .image_memory_barriers(&self.images)
            .buffer_memory_barriers(&self.buffers)
            .build();
        device.cmd_pipeline_barrier2(cmd, &deps);
        self.images.clear();
        self.buffers.clear();
    }
}
