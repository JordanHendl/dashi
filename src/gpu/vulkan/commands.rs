//! Refactored command helpers for Dashi CommandQueue

use ash::vk;

use super::{convert_rect2d_to_vulkan, RenderPass, SampleCount, SubpassSampleInfo};
use crate::gpu::driver::command::CommandSink;
use crate::gpu::driver::state::vulkan::{USAGE_TO_ACCESS, USAGE_TO_STAGE};
use crate::gpu::driver::state::{BufferBarrier, Layout, LayoutTransition};
use crate::utils::Handle;
use crate::{
    Buffer, ClearValue, CommandQueue, ComputePipeline, Context, Fence, GPUError, GraphicsPipeline,
    Image, ImageView, QueueType, Result, Semaphore, SubmitInfo2, UsageBits,
};

// --- New: helpers to map engine Layout/UsageBits to Vulkan ---
#[inline]
pub fn layout_to_vk(layout: Layout) -> vk::ImageLayout {
    match layout {
        Layout::Undefined => vk::ImageLayout::UNDEFINED,
        Layout::General => vk::ImageLayout::GENERAL,
        Layout::ShaderReadOnly => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        Layout::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        Layout::DepthStencilAttachment => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        Layout::DepthStencilReadOnly => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
        Layout::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        Layout::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        Layout::Present => vk::ImageLayout::PRESENT_SRC_KHR,
    }
}

#[inline]
pub fn usage_to_stages(usage: UsageBits) -> vk::PipelineStageFlags {
    let mut flags = vk::PipelineStageFlags::empty();
    for (u, s) in USAGE_TO_STAGE {
        if usage.contains(*u) {
            flags |= *s;
        }
    }
    // Fallback: something benign
    if flags.is_empty() {
        flags = vk::PipelineStageFlags::TOP_OF_PIPE;
    }
    flags
}

#[inline]
pub fn usage_to_access(usage: UsageBits) -> vk::AccessFlags {
    let mut flags = vk::AccessFlags::empty();
    for (u, a) in USAGE_TO_ACCESS {
        if usage.contains(*u) {
            flags |= *a;
        }
    }
    flags
}

#[inline]
fn queue_family_index(ctx: &Context, ty: QueueType) -> u32 {
    match ty {
        QueueType::Graphics => ctx.gfx_queue.family,
        QueueType::Compute => ctx.compute_queue.as_ref().unwrap_or(&ctx.gfx_queue).family,
        QueueType::Transfer => {
            ctx.transfer_queue
                .as_ref()
                .or(ctx.compute_queue.as_ref())
                .unwrap_or(&ctx.gfx_queue)
                .family
        }
    }
}
fn clear_value_to_vk(cv: &ClearValue) -> vk::ClearValue {
    match cv {
        ClearValue::Color(cols) => vk::ClearValue {
            color: vk::ClearColorValue { float32: *cols },
        },
        ClearValue::IntColor(cols) => vk::ClearValue {
            color: vk::ClearColorValue { int32: *cols },
        },
        ClearValue::UintColor(cols) => vk::ClearValue {
            color: vk::ClearColorValue { uint32: *cols },
        },
        ClearValue::DepthStencil { depth, stencil } => vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: *depth,
                stencil: *stencil,
            },
        },
    }
}

#[cfg(debug_assertions)]
fn validate_render_pass_compatibility(
    pipeline: &GraphicsPipeline,
    render_pass: &RenderPass,
    subpass_index: usize,
) -> Result<(), GPUError> {
    let subpass_formats = render_pass
        .subpass_formats
        .get(subpass_index)
        .ok_or_else(|| GPUError::InvalidSubpass {
            subpass: subpass_index as u32,
            available: render_pass.subpass_formats.len() as u32,
        })?;

    let subpass_samples = render_pass
        .subpass_samples
        .get(subpass_index)
        .ok_or_else(|| GPUError::InvalidSubpass {
            subpass: subpass_index as u32,
            available: render_pass.subpass_samples.len() as u32,
        })?;

    if pipeline.subpass as usize != subpass_index {
        return Err(GPUError::InvalidSubpass {
            subpass: subpass_index as u32,
            available: render_pass.subpass_samples.len() as u32,
        });
    }

    if pipeline.subpass_formats.color_formats.len() != subpass_formats.color_formats.len() {
        return Err(GPUError::LibraryError());
    }

    for (attachment_idx, (expected, actual)) in pipeline
        .subpass_formats
        .color_formats
        .iter()
        .zip(subpass_formats.color_formats.iter())
        .enumerate()
    {
        if expected != actual {
            return Err(GPUError::MismatchedAttachmentFormat {
                context: format!(
                    "render pass subpass {} color attachment {}",
                    subpass_index, attachment_idx
                ),
                expected: *expected,
                actual: *actual,
            });
        }
    }

    for (attachment_idx, (expected, actual)) in pipeline
        .subpass_samples
        .color_samples
        .iter()
        .zip(subpass_samples.color_samples.iter())
        .enumerate()
    {
        if expected != actual {
            return Err(GPUError::MismatchedSampleCount {
                context: format!(
                    "render pass subpass {} color attachment {}",
                    subpass_index, attachment_idx
                ),
                expected: *expected,
                actual: *actual,
            });
        }
    }

    match (
        pipeline.subpass_formats.depth_format,
        subpass_formats.depth_format,
    ) {
        (Some(expected), Some(actual)) if expected != actual => {
            return Err(GPUError::MismatchedAttachmentFormat {
                context: format!("render pass subpass {} depth attachment", subpass_index),
                expected,
                actual,
            });
        }
        (None, Some(_)) | (Some(_), None) => {
            return Err(GPUError::LibraryError());
        }
        _ => {}
    }

    match (
        pipeline.subpass_samples.depth_sample,
        subpass_samples.depth_sample,
    ) {
        (Some(expected), Some(actual)) if expected != actual => {
            return Err(GPUError::MismatchedSampleCount {
                context: format!("render pass subpass {} depth attachment", subpass_index),
                expected,
                actual,
            });
        }
        (None, Some(_)) | (Some(_), None) => {
            return Err(GPUError::LibraryError());
        }
        _ => {}
    }

    Ok(())
}

impl CommandQueue {
    /// Reset the command buffer and begin recording again.
    pub fn reset(&mut self) -> Result<()> {
        unsafe {
            (*self.pool).reset(self)?;

            (*self.ctx).device.begin_command_buffer(
                self.cmd_buf,
                &vk::CommandBufferBeginInfo::builder().build(),
            )?;
        }

        self.dirty = true;
        self.curr_rp = None;
        self.curr_subpass = None;
        self.curr_pipeline = None;
        Ok(())
    }

    /// Begin recording a secondary command queue using the same pool.
    fn begin_secondary(&mut self, debug_name: &str) -> Result<CommandQueue> {
        unsafe { (*self.pool).begin(self.ctx, debug_name, true) }
    }
    fn submit(&mut self, info: &SubmitInfo2) -> Result<Handle<Fence>, GPUError> {
        if self.dirty {
            unsafe { (*self.ctx).device.end_command_buffer(self.cmd_buf)? };
            self.dirty = false;
        }

        unsafe {
            let wait_sems: Vec<Handle<Semaphore>> = info
                .wait_sems
                .iter()
                .copied()
                .filter(|h| h.valid())
                .collect();

            let signal_sems: Vec<Handle<Semaphore>> = info
                .signal_sems
                .iter()
                .copied()
                .filter(|h| h.valid())
                .collect();

            let raw_wait_sems: Vec<vk::Semaphore> = wait_sems
                .into_iter()
                .map(|a| (*self.ctx).semaphores.get_ref(a.clone()).unwrap().raw)
                .collect();

            let raw_signal_sems: Vec<vk::Semaphore> = signal_sems
                .into_iter()
                .map(|a| (*self.ctx).semaphores.get_ref(a.clone()).unwrap().raw)
                .collect();

            let stage_masks = vec![vk::PipelineStageFlags::ALL_COMMANDS; raw_wait_sems.len()];

            let queue = (*self.ctx).queue(self.queue_type);
            (*self.ctx).device.queue_submit(
                queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[self.cmd_buf])
                    .signal_semaphores(&raw_signal_sems)
                    .wait_dst_stage_mask(&stage_masks)
                    .wait_semaphores(&raw_wait_sems)
                    .build()],
                (*self.ctx).fences.get_ref(self.fence).unwrap().raw.clone(),
            )?;

            return Ok(self.fence.clone());
        }
    }

    fn bind_compute_pipeline(&mut self, pipeline: Handle<ComputePipeline>) -> Result<()> {
        todo!()
        //        if self.curr_rp.is_none() {
        //            return Err(GPUError::LibraryError());
        //        }
        //        if self.curr_pipeline == Some(pipeline) {
        //            return Ok(());
        //        }
        //        self.curr_pipeline = Some(pipeline);
        //        unsafe {
        //            if let Some(pipeline) = info.compute {
        //                let comp = self
        //                    .ctx_ref()
        //                    .compute_pipelines
        //                    .get_ref(pipeline)
        //                    .ok_or(GPUError::SlotError())?;
        //                let layout = self
        //                    .ctx_ref()
        //                    .compute_pipeline_layouts
        //                    .get_ref(comp.layout)
        //                    .ok_or(GPUError::SlotError())?
        //                    .layout;
        //                self.ctx_ref().device.cmd_bind_pipeline(
        //                    self.cmd_buf,
        //                    vk::PipelineBindPoint::COMPUTE,
        //                    comp.raw,
        //                );
        //                let offsets: Vec<u32> = info.bindings.dynamic_offsets().collect();
        //                for (bt, bg) in info.bindings.iter() {
        //                    self.bind_descriptor_set(
        //                        vk::PipelineBindPoint::COMPUTE,
        //                        layout,
        //                        bt,
        //                        bg,
        //                        &offsets,
        //                    );
        //                }
        //            }
        //        }
        //        Ok(())
    }

    /// Bind a graphics pipeline for subsequent draw calls.
    fn bind_graphics_pipeline(&mut self, pipeline: Handle<GraphicsPipeline>) -> Result<()> {
        let curr_rp = self.curr_rp.ok_or(GPUError::LibraryError())?;
        if self.curr_pipeline == Some(pipeline) {
            return Ok(());
        }
        let gfx = self
            .ctx_ref()
            .gfx_pipelines
            .get_ref(pipeline)
            .ok_or(GPUError::SlotError())?;

        #[cfg(debug_assertions)]
        {
            let rp = self
                .ctx_ref()
                .render_passes
                .get_ref(curr_rp)
                .ok_or(GPUError::SlotError())?;
            let active_subpass = self.curr_subpass.unwrap_or(gfx.subpass as usize);

            validate_render_pass_compatibility(gfx, rp, active_subpass)?;
        }
        self.curr_pipeline = Some(pipeline);
        unsafe {
            self.ctx_ref().device.cmd_bind_pipeline(
                self.cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                gfx.raw,
            );
        }
        Ok(())
    }

    fn apply_graphics_pipeline_state_update(
        &mut self,
        update: &crate::gpu::driver::command::GraphicsPipelineStateUpdate,
    ) {
        let Some(pipeline) = self.curr_pipeline else {
            return;
        };
        let ctx = self.ctx_ref();
        let Some(gfx) = ctx.gfx_pipelines.get_ref(pipeline) else {
            return;
        };
        let Some(layout) = ctx.gfx_pipeline_layouts.get_ref(gfx.layout) else {
            return;
        };

        if let Some(viewport) = update.viewport {
            unsafe {
                if layout.dynamic_states.contains(&vk::DynamicState::VIEWPORT) {
                    let vk_viewport = vk::Viewport {
                        x: viewport.area.x,
                        y: viewport.area.y,
                        width: viewport.area.w,
                        height: viewport.area.h,
                        min_depth: viewport.min_depth,
                        max_depth: viewport.max_depth,
                    };
                    ctx.device.cmd_set_viewport(self.cmd_buf, 0, &[vk_viewport]);
                }

                if layout.dynamic_states.contains(&vk::DynamicState::SCISSOR) {
                    let vk_scissor = convert_rect2d_to_vulkan(viewport.scissor);
                    ctx.device.cmd_set_scissor(self.cmd_buf, 0, &[vk_scissor]);
                }
            }
        }
    }

    fn update_last_access(&mut self, stage: vk::PipelineStageFlags, access: vk::AccessFlags) {
        self.last_op_stage = stage;
        self.last_op_access = access;
    }

    fn ensure_image_state(
        &mut self,
        image: Handle<Image>,
        range: crate::structs::SubresourceRange,
        usage: UsageBits,
        layout: Layout,
    ) {
        if let Some(transition) = {
            let ctx = self.ctx_ref();
            ctx.resource_states
                .request_image_state(image, range, usage, layout, self.queue_type)
        } {
            self.apply_image_barrier(&transition);
        }
    }

    fn ensure_buffer_state(&mut self, buffer: Handle<Buffer>, usage: UsageBits) {
        if let Some(barrier) = {
            let ctx = self.ctx_ref();
            ctx.resource_states
                .request_buffer_state(buffer, usage, self.queue_type)
        } {
            self.apply_buffer_barrier(&barrier);
        }
    }

    fn apply_image_barrier(&mut self, cmd: &LayoutTransition) {
        let ctx = self.ctx_ref();
        let img_data = ctx
            .images
            .get_ref(cmd.image)
            .ok_or(GPUError::SlotError())
            .unwrap();

        let aspect = if cmd
            .new_usage
            .intersects(UsageBits::DEPTH_READ | UsageBits::DEPTH_WRITE)
        {
            if img_data.format == crate::gpu::Format::D24S8 {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            } else {
                vk::ImageAspectFlags::DEPTH
            }
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let sub = vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: cmd.range.base_mip,
            level_count: cmd.range.level_count,
            base_array_layer: cmd.range.base_layer,
            layer_count: cmd.range.layer_count,
        };

        let dst_stage = usage_to_stages(cmd.new_usage);
        let src_access = usage_to_access(cmd.old_usage);
        let dst_access = usage_to_access(cmd.new_usage);

        let (src_family, dst_family) = if cmd.old_queue != cmd.new_queue {
            (
                queue_family_index(&ctx, cmd.old_queue),
                queue_family_index(&ctx, cmd.new_queue),
            )
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };

        let barrier = vk::ImageMemoryBarrier::builder()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .old_layout(layout_to_vk(cmd.old_layout))
            .new_layout(layout_to_vk(cmd.new_layout))
            .src_queue_family_index(src_family)
            .dst_queue_family_index(dst_family)
            .image(img_data.img)
            .subresource_range(sub)
            .build();

        unsafe {
            ctx.device.cmd_pipeline_barrier(
                self.cmd_buf,
                self.last_op_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };
        self.update_last_access(dst_stage, dst_access);
        let img = ctx.images.get_mut_ref(cmd.image).unwrap();
        for level in 0..cmd.range.level_count {
            let mip = (cmd.range.base_mip + level) as usize;
            if let Some(l) = img.layouts.get_mut(mip) {
                *l = layout_to_vk(cmd.new_layout);
            }
        }
    }

    fn apply_buffer_barrier(&mut self, cmd: &BufferBarrier) {
        let ctx = self.ctx_ref();
        let buffer = ctx
            .buffers
            .get_ref(cmd.buffer)
            .ok_or(GPUError::SlotError())
            .unwrap();

        let src_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        let src_access = vk::AccessFlags::empty();

        let dst_stage = usage_to_stages(cmd.usage);
        let dst_access = usage_to_access(cmd.usage);

        let (src_family, dst_family) = if cmd.old_queue != cmd.new_queue {
            (
                queue_family_index(&ctx, cmd.old_queue),
                queue_family_index(&ctx, cmd.new_queue),
            )
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };

        let barrier = vk::BufferMemoryBarrier::builder()
            .buffer(buffer.buf)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .size(vk::WHOLE_SIZE)
            .src_queue_family_index(src_family)
            .dst_queue_family_index(dst_family)
            .build();

        unsafe {
            ctx.device.cmd_pipeline_barrier(
                self.cmd_buf,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            )
        };
        self.update_last_access(dst_stage, dst_access);
    }
    pub(crate) fn ctx_ref(&self) -> &'static mut Context {
        unsafe { &mut *self.ctx }
    }

    fn validate_subpass_samples(
        &mut self,
        subpass_samples: &[SubpassSampleInfo],
        subpass_index: usize,
        color_attachments: &[Option<crate::ImageView>],
        depth_attachment: Option<crate::ImageView>,
        pipeline_samples: Option<SampleCount>,
    ) -> Result<(), GPUError> {
        let subpass_info =
            subpass_samples
                .get(subpass_index)
                .ok_or_else(|| GPUError::InvalidSubpass {
                    subpass: subpass_index as u32,
                    available: subpass_samples.len() as u32,
                })?;

        for (attachment_idx, attachment) in color_attachments.iter().enumerate() {
            if let Some(view) = attachment {
                let image = self
                    .ctx_ref()
                    .images
                    .get_ref(view.img)
                    .ok_or(GPUError::SlotError())?;

                if let Some(expected) = subpass_info.color_samples.get(attachment_idx) {
                    if image.samples != *expected {
                        return Err(GPUError::MismatchedSampleCount {
                            context: format!(
                                "image for render pass subpass {} color attachment {}",
                                subpass_index, attachment_idx
                            ),
                            expected: *expected,
                            actual: image.samples,
                        });
                    }
                }

                if let Some(pipeline_samples) = pipeline_samples {
                    if image.samples != pipeline_samples {
                        return Err(GPUError::MismatchedSampleCount {
                            context: format!(
                                "graphics pipeline vs image for color attachment {}",
                                attachment_idx
                            ),
                            expected: pipeline_samples,
                            actual: image.samples,
                        });
                    }
                }
            }
        }

        if let Some(view) = depth_attachment {
            let image = self
                .ctx_ref()
                .images
                .get_ref(view.img)
                .ok_or(GPUError::SlotError())?;

            if let Some(expected) = subpass_info.depth_sample {
                if image.samples != expected {
                    return Err(GPUError::MismatchedSampleCount {
                        context: format!(
                            "image for render pass subpass {} depth attachment",
                            subpass_index
                        ),
                        expected,
                        actual: image.samples,
                    });
                }

                if let Some(pipeline_samples) = pipeline_samples {
                    if image.samples != pipeline_samples {
                        return Err(GPUError::MismatchedSampleCount {
                            context: "graphics pipeline vs image for depth attachment".to_string(),
                            expected: pipeline_samples,
                            actual: image.samples,
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

impl CommandSink for CommandQueue {
    fn begin_render_pass(&mut self, cmd: &crate::gpu::driver::command::BeginRenderPass) {
        for view in cmd.color_attachments.iter().flatten() {
            self.ensure_image_state(
                view.img,
                view.range,
                UsageBits::RT_WRITE,
                Layout::ColorAttachment,
            );
        }
        if let Some(depth) = cmd.depth_attachment {
            self.ensure_image_state(
                depth.img,
                depth.range,
                UsageBits::DEPTH_WRITE,
                Layout::DepthStencilAttachment,
            );
        }
        // end previous pass
        if self.curr_rp.is_some() {
            unsafe {
                self.ctx_ref().device.cmd_end_render_pass(self.cmd_buf);
            }
            // finalize CPU layouts for previous attachments
            let attachments_prev = std::mem::take(&mut self.curr_attachments);
            for (view, layout) in attachments_prev {
                let ctx = self.ctx_ref();
                let v = ctx.image_views.get_ref(view).unwrap();
                let img = ctx.images.get_mut_ref(v.img).unwrap();
                let base = v.range.base_mip_level as usize;
                let count = v.range.level_count as usize;
                for i in base..base + count {
                    if let Some(l) = img.layouts.get_mut(i) {
                        *l = layout;
                    }
                }
            }
        }

        self.curr_rp = Some(cmd.render_pass);
        self.curr_subpass = Some(0);

        let rp_obj = self
            .ctx_ref()
            .render_passes
            .get_ref(cmd.render_pass)
            .ok_or(GPUError::SlotError())
            .unwrap();
        let rp_raw = rp_obj.raw;
        let fb = rp_obj.fb;

        #[cfg(debug_assertions)]
        {
            let subpass_samples = rp_obj.subpass_samples.clone();
            if let Err(err) = self.validate_subpass_samples(
                &subpass_samples,
                0,
                &cmd.color_attachments,
                cmd.depth_attachment,
                None,
            ) {
                panic!("{}", err);
            }
        }

        let mut attachments_vk: Vec<vk::ImageView> = Vec::new();
        self.curr_attachments.clear();

        for view in cmd.color_attachments.iter().flatten() {
            let handle = self.ctx_ref().get_or_create_image_view(view).unwrap();
            let vk_view = {
                let v = self.ctx_ref().image_views.get_ref(handle).unwrap();
                v.view
            };
            attachments_vk.push(vk_view);
            self.curr_attachments
                .push((handle, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL));
        }

        if let Some(depth) = cmd.depth_attachment {
            let handle = self.ctx_ref().get_or_create_image_view(&depth).unwrap();
            let vk_view = {
                let v = self.ctx_ref().image_views.get_ref(handle).unwrap();
                v.view
            };
            attachments_vk.push(vk_view);
            self.curr_attachments
                .push((handle, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL));
        }

        for (view_handle, layout) in &self.curr_attachments {
            let ctx = self.ctx_ref();
            let v = ctx.image_views.get_ref(*view_handle).unwrap();
            let img = ctx.images.get_mut_ref(v.img).unwrap();
            let base = v.range.base_mip_level as usize;
            let count = v.range.level_count as usize;
            for i in base..base + count {
                if let Some(l) = img.layouts.get_mut(i) {
                    *l = *layout;
                }
            }
        }

        let mut attachment_info =
            vk::RenderPassAttachmentBeginInfo::builder().attachments(&attachments_vk);

        let clears: Vec<vk::ClearValue> = cmd
            .clear_values
            .iter()
            .flatten()
            .map(clear_value_to_vk)
            .collect();

        unsafe {
            self.ctx_ref().device.cmd_begin_render_pass(
                self.cmd_buf,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(rp_raw)
                    .framebuffer(fb)
                    .render_area(convert_rect2d_to_vulkan(cmd.viewport.scissor))
                    .clear_values(&clears)
                    .push_next(&mut attachment_info)
                    .build(),
                vk::SubpassContents::INLINE,
            );

            self.update_last_access(vk::PipelineStageFlags::NONE, vk::AccessFlags::empty());
        }
    }

    fn begin_drawing(&mut self, cmd: &crate::gpu::driver::command::BeginDrawing) {
        for view in cmd.color_attachments.iter().flatten() {
            self.ensure_image_state(
                view.img,
                view.range,
                UsageBits::RT_WRITE,
                Layout::ColorAttachment,
            );
        }
        if let Some(depth) = cmd.depth_attachment {
            self.ensure_image_state(
                depth.img,
                depth.range,
                UsageBits::DEPTH_WRITE,
                Layout::DepthStencilAttachment,
            );
        }
        let (pipeline_rp, pipeline_subpass, layout_handle) = {
            let gfx = self
                .ctx_ref()
                .gfx_pipelines
                .get_ref(cmd.pipeline)
                .ok_or(GPUError::SlotError())
                .unwrap();
            (gfx.render_pass, gfx.subpass as usize, gfx.layout)
        };

        if self.curr_rp.map_or(false, |rp| rp == pipeline_rp) {
            return;
        }

        // end previous pass
        if self.curr_rp.is_some() {
            unsafe {
                self.ctx_ref().device.cmd_end_render_pass(self.cmd_buf);
            }
            // finalize CPU layouts for previous attachments
            let attachments_prev = std::mem::take(&mut self.curr_attachments);
            for (view, layout) in attachments_prev {
                let ctx = self.ctx_ref();
                let v = ctx.image_views.get_ref(view).unwrap();
                let img = ctx.images.get_mut_ref(v.img).unwrap();
                let base = v.range.base_mip_level as usize;
                let count = v.range.level_count as usize;
                for i in base..base + count {
                    if let Some(l) = img.layouts.get_mut(i) {
                        *l = layout;
                    }
                }
            }
        }

        self.curr_rp = Some(pipeline_rp);
        self.curr_subpass = Some(0);

        let rp_obj = self
            .ctx_ref()
            .render_passes
            .get_ref(pipeline_rp)
            .ok_or(GPUError::SlotError())
            .unwrap();
        let rp_raw = rp_obj.raw;
        let fb = rp_obj.fb;

        #[cfg(debug_assertions)]
        {
            let pipeline_samples = {
                let layout = self
                    .ctx_ref()
                    .gfx_pipeline_layouts
                    .get_ref(layout_handle)
                    .ok_or(GPUError::SlotError())
                    .unwrap();
                layout.sample_count
            };

            let subpass_samples = rp_obj.subpass_samples.clone();

            if let Err(err) = self.validate_subpass_samples(
                &subpass_samples,
                pipeline_subpass,
                &cmd.color_attachments,
                cmd.depth_attachment,
                Some(pipeline_samples),
            ) {
                panic!("{}", err);
            }
        }

        let mut attachments_vk: Vec<vk::ImageView> = Vec::new();
        self.curr_attachments.clear();

        for view in cmd.color_attachments.iter().flatten() {
            let handle = self.ctx_ref().get_or_create_image_view(view).unwrap();
            let vk_view = {
                let v = self.ctx_ref().image_views.get_ref(handle).unwrap();
                v.view
            };
            attachments_vk.push(vk_view);
            self.curr_attachments
                .push((handle, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL));
        }

        if let Some(depth) = cmd.depth_attachment {
            let handle = self.ctx_ref().get_or_create_image_view(&depth).unwrap();
            let vk_view = {
                let v = self.ctx_ref().image_views.get_ref(handle).unwrap();
                v.view
            };
            attachments_vk.push(vk_view);
            self.curr_attachments
                .push((handle, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL));
        }

        for (view_handle, layout) in &self.curr_attachments {
            let ctx = self.ctx_ref();
            let v = ctx.image_views.get_ref(*view_handle).unwrap();
            let img = ctx.images.get_mut_ref(v.img).unwrap();
            let base = v.range.base_mip_level as usize;
            let count = v.range.level_count as usize;
            for i in base..base + count {
                if let Some(l) = img.layouts.get_mut(i) {
                    *l = *layout;
                }
            }
        }

        let mut attachment_info =
            vk::RenderPassAttachmentBeginInfo::builder().attachments(&attachments_vk);

        let mut clears: Vec<vk::ClearValue> = cmd
            .clear_values
            .iter()
            .flatten()
            .map(clear_value_to_vk)
            .collect();

        if let Some(depth_clear) = cmd.depth_clear.as_ref() {
            clears.push(clear_value_to_vk(depth_clear));
        }

        unsafe {
            self.ctx_ref().device.cmd_begin_render_pass(
                self.cmd_buf,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(rp_raw)
                    .framebuffer(fb)
                    .render_area(convert_rect2d_to_vulkan(cmd.viewport.scissor))
                    .clear_values(&clears)
                    .push_next(&mut attachment_info)
                    .build(),
                vk::SubpassContents::INLINE,
            );

            let _ = self.bind_graphics_pipeline(cmd.pipeline);
            self.update_last_access(vk::PipelineStageFlags::NONE, vk::AccessFlags::empty());
        }
    }

    fn end_drawing(&mut self, _pass: &crate::gpu::driver::command::EndDrawing) {
        unsafe { (*self.ctx).device.cmd_end_render_pass(self.cmd_buf) };
        self.last_op_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        self.last_op_access = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;

        let attachments = std::mem::take(&mut self.curr_attachments);
        for (view, layout) in attachments {
            let ctx = self.ctx_ref();
            let v = ctx.image_views.get_ref(view).unwrap();
            let img = ctx.images.get_mut_ref(v.img).unwrap();
            let base = v.range.base_mip_level as usize;
            let count = v.range.level_count as usize;
            for i in base..base + count {
                if let Some(l) = img.layouts.get_mut(i) {
                    *l = layout;
                }
            }
        }

        self.curr_rp = None;
        self.curr_subpass = None;
        self.curr_pipeline = None;
    }

    fn next_subpass(&mut self, _cmd: &crate::gpu::driver::command::NextSubpass) {
        let curr = self.curr_subpass.unwrap_or(0);
        let rp = self
            .curr_rp
            .ok_or_else(|| GPUError::InvalidSubpass {
                subpass: curr as u32,
                available: 0,
            })
            .unwrap();

        let rp_obj = self
            .ctx_ref()
            .render_passes
            .get_ref(rp)
            .ok_or(GPUError::SlotError())
            .unwrap();

        let next = curr + 1;
        let available = rp_obj.subpass_samples.len();
        if next >= available {
            panic!(
                "{}",
                GPUError::InvalidSubpass {
                    subpass: next as u32,
                    available: available as u32,
                }
            );
        }

        unsafe {
            self.ctx_ref()
                .device
                .cmd_next_subpass(self.cmd_buf, vk::SubpassContents::INLINE);
        }

        self.curr_subpass = Some(next);
        self.curr_pipeline = None;
        self.last_op_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        self.last_op_access = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
    }

    fn bind_graphics_pipeline(&mut self, cmd: &crate::gpu::driver::command::BindGraphicsPipeline) {
        let _ = self.bind_graphics_pipeline(cmd.pipeline);
    }

    fn update_graphics_pipeline_state(
        &mut self,
        cmd: &crate::gpu::driver::command::GraphicsPipelineStateUpdate,
    ) {
        self.apply_graphics_pipeline_state_update(cmd);
    }

    fn blit_image(&mut self, cmd: &crate::gpu::driver::command::BlitImage) {
        self.ensure_image_state(
            cmd.src,
            cmd.src_range,
            UsageBits::COPY_SRC,
            Layout::TransferSrc,
        );
        self.ensure_image_state(
            cmd.dst,
            cmd.dst_range,
            UsageBits::COPY_DST,
            Layout::TransferDst,
        );
        unsafe {
            let src_data = self
                .ctx_ref()
                .images
                .get_ref(cmd.src)
                .ok_or(GPUError::SlotError())
                .unwrap();
            let dst_data = self
                .ctx_ref()
                .images
                .get_ref(cmd.dst)
                .ok_or(GPUError::SlotError())
                .unwrap();
            let src_dim = crate::gpu::mip_dimensions(src_data.dim, cmd.src_range.base_mip);
            let dst_dim = crate::gpu::mip_dimensions(dst_data.dim, cmd.dst_range.base_mip);
            let regions = [vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: cmd.src_range.base_mip,
                    base_array_layer: cmd.src_range.base_layer,
                    layer_count: cmd.src_range.layer_count,
                },
                src_offsets: [
                    vk::Offset3D {
                        x: cmd.src_region.x as i32,
                        y: cmd.src_region.y as i32,
                        z: 0,
                    },
                    vk::Offset3D {
                        x: (cmd.src_region.w.max(src_data.dim[0])) as i32,
                        y: (cmd.src_region.h.max(src_data.dim[1])) as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: cmd.dst_range.base_mip,
                    base_array_layer: cmd.dst_range.base_layer,
                    layer_count: cmd.dst_range.layer_count,
                },
                dst_offsets: [
                    vk::Offset3D {
                        x: cmd.dst_region.x as i32,
                        y: cmd.dst_region.y as i32,
                        z: 0,
                    },
                    vk::Offset3D {
                        x: (cmd.dst_region.w.max(dst_data.dim[0])) as i32,
                        y: (cmd.dst_region.h.max(dst_data.dim[1])) as i32,
                        z: 1,
                    },
                ],
            }];
            let src_mip = cmd.src_range.base_mip as usize;
            let dst_mip = cmd.dst_range.base_mip as usize;

            self.ctx_ref().device.cmd_blit_image(
                self.cmd_buf,
                src_data.img,
                src_data.layouts[src_mip],
                dst_data.img,
                dst_data.layouts[dst_mip],
                &regions,
                cmd.filter.into(),
            );

            self.update_last_access(
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            );
        }
    }

    fn draw(&mut self, cmd: &crate::gpu::driver::command::Draw) {
        let v = self.ctx_ref().buffers.get_ref(cmd.vertices).unwrap();
        unsafe {
            self.ctx_ref().device.cmd_bind_vertex_buffers(
                self.cmd_buf,
                0,
                &[v.buf],
                &[v.offset as u64],
            );
        }
        let offsets: Vec<u32> = cmd
            .dynamic_buffers
            .iter()
            .filter_map(|&d| d.map(|b| b.alloc.offset))
            .collect();

        if let Some(pipe) = self.curr_pipeline {
            let p = self.ctx_ref().gfx_pipelines.get_ref(pipe).unwrap();
            let l = self
                .ctx_ref()
                .gfx_pipeline_layouts
                .get_ref(p.layout)
                .unwrap();

            unsafe {
                for table in &cmd.bind_tables {
                    if let Some(bt) = *table {
                        let bt_data = self.ctx_ref().bind_tables.get_ref(bt).unwrap();
                        self.ctx_ref().device.cmd_bind_descriptor_sets(
                            self.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            l.layout,
                            bt_data.set_id,
                            &[bt_data.set],
                            &[],
                        );
                    }
                }

                for group in cmd.bind_groups {
                    if let Some(bg) = group {
                        let bg_data = self.ctx_ref().bind_groups.get_ref(bg).unwrap();
                        self.ctx_ref().device.cmd_bind_descriptor_sets(
                            self.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            l.layout,
                            bg_data.set_id,
                            &[bg_data.set],
                            &offsets,
                        );
                    }
                }
            }
        }

        unsafe {
            self.ctx_ref()
                .device
                .cmd_draw(self.cmd_buf, cmd.count, cmd.instance_count, 0, 0);
            self.update_last_access(
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            );
        }
    }

    fn draw_indexed(&mut self, cmd: &crate::gpu::driver::command::DrawIndexed) {
        let v = self.ctx_ref().buffers.get_ref(cmd.vertices).unwrap();
        let i = self.ctx_ref().buffers.get_ref(cmd.indices).unwrap();
        unsafe {
            self.ctx_ref().device.cmd_bind_vertex_buffers(
                self.cmd_buf,
                0,
                &[v.buf],
                &[v.offset as u64],
            );

            self.ctx_ref().device.cmd_bind_index_buffer(
                self.cmd_buf,
                i.buf,
                i.offset as u64,
                vk::IndexType::UINT32,
            );
        }
        let offsets: Vec<u32> = cmd
            .dynamic_buffers
            .iter()
            .filter_map(|&d| d.map(|b| b.alloc.offset))
            .collect();

        if let Some(pipe) = self.curr_pipeline {
            let p = self.ctx_ref().gfx_pipelines.get_ref(pipe).unwrap();
            let l = self
                .ctx_ref()
                .gfx_pipeline_layouts
                .get_ref(p.layout)
                .unwrap();

            unsafe {
                for table in &cmd.bind_tables {
                    if let Some(bt) = *table {
                        let bt_data = self.ctx_ref().bind_tables.get_ref(bt).unwrap();
                        self.ctx_ref().device.cmd_bind_descriptor_sets(
                            self.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            l.layout,
                            bt_data.set_id,
                            &[bt_data.set],
                            &[],
                        );
                    }
                }

                for group in cmd.bind_groups {
                    if let Some(bg) = group {
                        let bg_data = self.ctx_ref().bind_groups.get_ref(bg).unwrap();
                        self.ctx_ref().device.cmd_bind_descriptor_sets(
                            self.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            l.layout,
                            bg_data.set_id,
                            &[bg_data.set],
                            &offsets,
                        );
                    }
                }
            }
        }

        unsafe {
            self.ctx_ref().device.cmd_draw_indexed(
                self.cmd_buf,
                cmd.index_count as u32,
                cmd.instance_count,
                0,
                0,
                cmd.first_instance,
            );
            self.update_last_access(
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            );
        }
    }

    fn dispatch(&mut self, cmd: &crate::gpu::driver::command::Dispatch) {
        unsafe {
            self.ctx_ref()
                .device
                .cmd_dispatch(self.cmd_buf, cmd.x, cmd.y, cmd.z);
            self.update_last_access(
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
            );
        }
    }

    fn copy_buffer(&mut self, cmd: &crate::gpu::driver::command::CopyBuffer) {
        self.ensure_buffer_state(cmd.src, UsageBits::COPY_SRC);
        self.ensure_buffer_state(cmd.dst, UsageBits::COPY_DST);
        unsafe {
            let src = self.ctx_ref().buffers.get_ref(cmd.src).unwrap();
            let dst = self.ctx_ref().buffers.get_ref(cmd.dst).unwrap();
            self.ctx_ref().device.cmd_copy_buffer(
                self.cmd_buf,
                src.buf,
                dst.buf,
                &[vk::BufferCopy {
                    src_offset: cmd.src_offset as u64,
                    dst_offset: cmd.dst_offset as u64,
                    size: if cmd.amount == 0 {
                        src.size.min(dst.size) as u64
                    } else {
                        cmd.amount as u64
                    },
                }],
            );
            self.update_last_access(
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_WRITE,
            );
        }
    }

    fn copy_buffer_to_image(&mut self, cmd: &crate::gpu::driver::command::CopyBufferImage) {
        self.ensure_buffer_state(cmd.src, UsageBits::COPY_SRC);
        self.ensure_image_state(cmd.dst, cmd.range, UsageBits::COPY_DST, Layout::TransferDst);
        unsafe {
            let img_data = self
                .ctx_ref()
                .images
                .get_ref(cmd.dst)
                .ok_or(GPUError::SlotError())
                .unwrap();
            let mip = cmd.range.base_mip as usize;
            let dims = crate::gpu::mip_dimensions(img_data.dim, cmd.range.base_mip);
            self.ctx_ref().device.cmd_copy_buffer_to_image(
                self.cmd_buf,
                self.ctx_ref()
                    .buffers
                    .get_ref(cmd.src)
                    .ok_or(GPUError::SlotError())
                    .unwrap()
                    .buf,
                img_data.img,
                img_data.layouts[mip],
                &[vk::BufferImageCopy {
                    buffer_offset: cmd.src_offset as u64,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: img_data.sub_layers.aspect_mask,
                        mip_level: cmd.range.base_mip,
                        base_array_layer: cmd.range.base_layer,
                        layer_count: cmd.range.layer_count,
                    },
                    image_extent: vk::Extent3D {
                        width: dims[0],
                        height: dims[1],
                        depth: dims[2],
                    },
                    ..Default::default()
                }],
            );

            self.update_last_access(
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_WRITE,
            );
        }
    }

    fn copy_image_to_buffer(&mut self, cmd: &crate::gpu::driver::command::CopyImageBuffer) {
        self.ensure_image_state(cmd.src, cmd.range, UsageBits::COPY_SRC, Layout::TransferSrc);
        self.ensure_buffer_state(cmd.dst, UsageBits::COPY_DST);
        unsafe {
            let img_data = self
                .ctx_ref()
                .images
                .get_ref(cmd.src)
                .ok_or(GPUError::SlotError())
                .unwrap();
            let mip = cmd.range.base_mip as usize;

            let dims = crate::gpu::mip_dimensions(img_data.dim, cmd.range.base_mip);
            self.ctx_ref().device.cmd_copy_image_to_buffer(
                self.cmd_buf,
                img_data.img,
                img_data.layouts[mip],
                self.ctx_ref()
                    .buffers
                    .get_ref(cmd.dst)
                    .ok_or(GPUError::SlotError())
                    .unwrap()
                    .buf,
                &[vk::BufferImageCopy {
                    buffer_offset: cmd.dst_offset as u64,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: img_data.sub_layers.aspect_mask,
                        mip_level: cmd.range.base_mip,
                        base_array_layer: cmd.range.base_layer,
                        layer_count: cmd.range.layer_count,
                    },
                    image_extent: vk::Extent3D {
                        width: dims[0],
                        height: dims[1],
                        depth: dims[2],
                    },
                    ..Default::default()
                }],
            );

            self.update_last_access(
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_WRITE,
            );
        }
    }

    fn copy_image(&mut self, _cmd: &crate::gpu::driver::command::CopyImage) {
        todo!()
    }

    fn transition_image(&mut self, cmd: &crate::gpu::driver::command::TransitionImage) {
        self.ensure_image_state(cmd.image, cmd.range, cmd.usage, cmd.layout);
    }

    fn submit(&mut self, cmd: &crate::SubmitInfo2) -> Handle<Fence> {
        self.submit(cmd).unwrap()
    }

    fn debug_marker_begin(&mut self, _cmd: &crate::gpu::driver::command::DebugMarkerBegin) {
        // Debug markers are not directly supported in Vulkan, so we need to use a memory barrier
        unsafe {
            self.ctx_ref().device.cmd_pipeline_barrier(
                self.cmd_buf,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[],
            )
        };
    }

    fn debug_marker_end(&mut self, _cmd: &crate::gpu::driver::command::DebugMarkerEnd) {
        unsafe {
            self.ctx_ref().device.cmd_pipeline_barrier(
                self.cmd_buf,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[],
            )
        };
    }
}
