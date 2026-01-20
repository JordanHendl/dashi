//! Refactored command helpers for Dashi CommandQueue

use ash::vk;

use super::VulkanContext;
use super::{convert_rect2d_to_vulkan, RenderPass, SampleCount, SubpassSampleInfo};
use crate::gpu::driver::command::{CommandSink, Scope, SyncPoint};
use crate::gpu::driver::state::{usage_to_access, BufferBarrier, Layout, LayoutTransition};
use crate::utils::Handle;
use crate::{
    BindTable, Buffer, ClearValue, CommandQueue, ComputePipeline, Fence, GPUError,
    GraphicsPipeline, Image, QueueType, Rect2D, Result, Semaphore, SubmitInfo2, UsageBits,
};

// --- New: helpers to map engine Layout/UsageBits to Vulkan ---
#[inline]
pub(super) fn layout_to_vk(layout: Layout) -> vk::ImageLayout {
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
fn queue_family_index(ctx: &VulkanContext, ty: QueueType) -> u32 {
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

fn image_aspect_for_format(format: crate::gpu::Format) -> vk::ImageAspectFlags {
    if format == crate::gpu::Format::D24S8 {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
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

fn sync_scope_access(scope: Scope) -> vk::AccessFlags {
    match scope {
        Scope::AllCommonReads => usage_to_access(
            UsageBits::SAMPLED
                | UsageBits::STORAGE_READ
                | UsageBits::UNIFORM_READ
                | UsageBits::VERTEX_READ
                | UsageBits::INDEX_READ
                | UsageBits::INDIRECT_READ,
        ),
        Scope::All => vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
    }
}

fn sync_point_stages(point: SyncPoint) -> (vk::PipelineStageFlags, vk::PipelineStageFlags) {
    match point {
        SyncPoint::ComputeToGraphics => (
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::ALL_GRAPHICS,
        ),
        SyncPoint::GraphicsToCompute => (
            vk::PipelineStageFlags::ALL_GRAPHICS,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        SyncPoint::GraphicsToGraphics => (
            vk::PipelineStageFlags::ALL_GRAPHICS,
            vk::PipelineStageFlags::ALL_GRAPHICS,
        ),
        SyncPoint::TransferToGraphics => (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::ALL_GRAPHICS,
        ),
        SyncPoint::TransferToCompute => (
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        ),
        SyncPoint::ComputeToTransfer => (
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
        ),
        SyncPoint::GraphicsToTransfer => (
            vk::PipelineStageFlags::ALL_GRAPHICS,
            vk::PipelineStageFlags::TRANSFER,
        ),
    }
}

fn sync_point_src_access(point: SyncPoint) -> vk::AccessFlags {
    match point {
        SyncPoint::ComputeToGraphics | SyncPoint::ComputeToTransfer => vk::AccessFlags::SHADER_WRITE,
        SyncPoint::GraphicsToCompute
        | SyncPoint::GraphicsToGraphics
        | SyncPoint::GraphicsToTransfer => {
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags::SHADER_WRITE
        }
        SyncPoint::TransferToGraphics | SyncPoint::TransferToCompute => {
            vk::AccessFlags::TRANSFER_WRITE
        }
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
        return Err(GPUError::LibraryError(format!(
            "Pipeline subpass formats do not match actual subpass formats ({} -> {})",
            pipeline.subpass_formats.color_formats.len(),
            subpass_formats.color_formats.len()
        )));
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
            return Err(GPUError::LibraryError(format!(
                "Pipeline/subpass depth format mismatch."
            )));
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
            return Err(GPUError::LibraryError(format!(
                "Pipeline/subpass depth sample mismatch."
            )));
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
        unsafe { (*self.pool).begin_raw(self.ctx, debug_name, true) }
    }

    #[cfg(debug_assertions)]
    fn ensure_active_render_pass(&self) -> Result<()> {
        if self.curr_rp.is_none() {
            return Err(GPUError::LibraryError(
                "Drawing requires an active render pass.".to_string(),
            ));
        }
        Ok(())
    }

    #[cfg(not(debug_assertions))]
    fn ensure_active_render_pass(&self) -> Result<()> {
        Ok(())
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
                .map(|a| {
                    (*self.ctx)
                        .semaphores
                        .get_ref(a.clone())
                        .ok_or(GPUError::SlotError())
                        .map(|sem| sem.raw)
                })
                .collect::<Result<_, _>>()?;

            let raw_signal_sems: Vec<vk::Semaphore> = signal_sems
                .into_iter()
                .map(|a| {
                    (*self.ctx)
                        .semaphores
                        .get_ref(a.clone())
                        .ok_or(GPUError::SlotError())
                        .map(|sem| sem.raw)
                })
                .collect::<Result<_, _>>()?;

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
                (*self.ctx)
                    .fences
                    .get_ref(self.fence)
                    .ok_or(GPUError::SlotError())?
                    .raw
                    .clone(),
            )?;

            return Ok(self.fence.clone());
        }
    }

    fn bind_compute_pipeline(&mut self, pipeline: Handle<ComputePipeline>) -> Result<()> {
        unsafe {
            let comp = self
                .ctx_ref()
                .compute_pipelines
                .get_ref(pipeline)
                .ok_or(GPUError::SlotError())?;
            self.ctx_ref().device.cmd_bind_pipeline(
                self.cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                comp.raw,
            );
        }
        Ok(())
    }

    /// Bind a graphics pipeline for subsequent draw calls.
    fn bind_graphics_pipeline(&mut self, pipeline: Handle<GraphicsPipeline>) -> Result<()> {
        let curr_rp = self.curr_rp.ok_or(GPUError::LibraryError(format!(
            "Trying to bind graphics pipeline without active bound render pass."
        )))?;
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
    ) -> Result<()> {
        if let Some(transition) = {
            let ctx = self.ctx_ref();
            ctx.resource_states
                .request_image_state(image, range, usage, layout, self.queue_type)
        } {
            self.apply_image_barrier(&transition)?;
        }
        Ok(())
    }

    fn ensure_buffer_state(&mut self, buffer: Handle<Buffer>, usage: UsageBits) -> Result<()> {
        self.ensure_buffer_state_on_queue(buffer, usage, self.queue_type)
    }

    fn ensure_buffer_state_on_queue(
        &mut self,
        buffer: Handle<Buffer>,
        usage: UsageBits,
        queue: QueueType,
    ) -> Result<()> {
        if let Some(barrier) = {
            let ctx = self.ctx_ref();
            ctx.resource_states
                .request_buffer_state(buffer, usage, queue)
        } {
            self.apply_buffer_barrier(&barrier)?;
        }
        Ok(())
    }

    fn ensure_bind_table_state(
        &mut self,
        table: Handle<BindTable>,
        queue: QueueType,
    ) -> Result<()> {
        if let Some(bt) = self.ctx_ref().bind_tables.get_ref(table) {
            for (buffer, usage) in &bt.buffer_states {
                self.ensure_buffer_state_on_queue(*buffer, *usage, queue)?;
            }
        }
        Ok(())
    }

    fn ensure_binding_states(
        &mut self,
        bind_tables: &[Option<Handle<BindTable>>; 4],
    ) -> Result<()> {
        let queue = self.queue_type;
        for table in bind_tables.iter().flatten() {
            self.ensure_bind_table_state(*table, queue)?;
        }
        Ok(())
    }

    fn apply_image_barrier(&mut self, cmd: &LayoutTransition) -> Result<()> {
        let ctx = self.ctx_ref();
        let img_data = ctx.images.get_ref(cmd.image).ok_or(GPUError::SlotError())?;
        let img_info = ctx.image_info(cmd.image);

        let aspect = if cmd
            .new_usage
            .intersects(UsageBits::DEPTH_READ | UsageBits::DEPTH_WRITE)
        {
            if img_info.format == crate::gpu::Format::D24S8 {
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

        let src_stage = cmd.old_stage;
        let dst_stage = cmd.new_stage;
        let src_access = cmd.old_access;
        let dst_access = cmd.new_access;

        let (src_family, dst_family) = if cmd.old_queue != cmd.new_queue {
            (
                queue_family_index(&ctx, cmd.old_queue),
                queue_family_index(&ctx, cmd.new_queue),
            )
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };

        #[cfg(debug_assertions)]
        if cmd.old_queue != cmd.new_queue {
            if src_family == dst_family {
                eprintln!(
                    "Queue ownership transfer requested for image {:?} from {:?} to {:?}, but both map to queue family {}.",
                    cmd.image, cmd.old_queue, cmd.new_queue, src_family
                );
            } else {
                eprintln!(
                    "Queue ownership transfer for image {:?} from {:?} (family {}) to {:?} (family {}).",
                    cmd.image, cmd.old_queue, src_family, cmd.new_queue, dst_family
                );
            }
        }

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
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };
        self.update_last_access(dst_stage, dst_access);
        let img = ctx
            .images
            .get_mut_ref(cmd.image)
            .ok_or(GPUError::SlotError())?;
        for level in 0..cmd.range.level_count {
            let mip = (cmd.range.base_mip + level) as usize;
            if let Some(l) = img.layouts.get_mut(mip) {
                *l = layout_to_vk(cmd.new_layout);
            }
        }
        Ok(())
    }

    fn apply_buffer_barrier(&mut self, cmd: &BufferBarrier) -> Result<()> {
        let ctx = self.ctx_ref();
        let buffer = ctx
            .buffers
            .get_ref(cmd.buffer)
            .ok_or(GPUError::SlotError())?;

        let src_stage = cmd.old_stage;
        let src_access = cmd.old_access;

        let dst_stage = cmd.new_stage;
        let dst_access = cmd.new_access;

        let (src_family, dst_family) = if cmd.old_queue != cmd.new_queue {
            (
                queue_family_index(&ctx, cmd.old_queue),
                queue_family_index(&ctx, cmd.new_queue),
            )
        } else {
            (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
        };

        #[cfg(debug_assertions)]
        if cmd.old_queue != cmd.new_queue {
            if src_family == dst_family {
                eprintln!(
                    "Queue ownership transfer requested for buffer {:?} from {:?} to {:?}, but both map to queue family {}.",
                    cmd.buffer, cmd.old_queue, cmd.new_queue, src_family
                );
            } else {
                eprintln!(
                    "Queue ownership transfer for buffer {:?} from {:?} (family {}) to {:?} (family {}).",
                    cmd.buffer, cmd.old_queue, src_family, cmd.new_queue, dst_family
                );
            }
        }

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
        Ok(())
    }
    pub(crate) fn ctx_ref(&self) -> &'static mut VulkanContext {
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
                let image_info = self.ctx_ref().image_info(view.img);

                if let Some(expected) = subpass_info.color_samples.get(attachment_idx) {
                    if image_info.samples != *expected {
                        return Err(GPUError::MismatchedSampleCount {
                            context: format!(
                                "image for render pass subpass {} color attachment {}",
                                subpass_index, attachment_idx
                            ),
                            expected: *expected,
                            actual: image_info.samples,
                        });
                    }
                }

                if let Some(pipeline_samples) = pipeline_samples {
                    if image_info.samples != pipeline_samples {
                        return Err(GPUError::MismatchedSampleCount {
                            context: format!(
                                "graphics pipeline vs image for color attachment {}",
                                attachment_idx
                            ),
                            expected: pipeline_samples,
                            actual: image_info.samples,
                        });
                    }
                }
            }
        }

        if let Some(view) = depth_attachment {
            let image_info = self.ctx_ref().image_info(view.img);

            if let Some(expected) = subpass_info.depth_sample {
                if image_info.samples != expected {
                    return Err(GPUError::MismatchedSampleCount {
                        context: format!(
                            "image for render pass subpass {} depth attachment",
                            subpass_index
                        ),
                        expected,
                        actual: image_info.samples,
                    });
                }

                if let Some(pipeline_samples) = pipeline_samples {
                    if image_info.samples != pipeline_samples {
                        return Err(GPUError::MismatchedSampleCount {
                            context: "graphics pipeline vs image for depth attachment".to_string(),
                            expected: pipeline_samples,
                            actual: image_info.samples,
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

impl CommandSink for CommandQueue {
    fn begin_render_pass(
        &mut self,
        cmd: &crate::gpu::driver::command::BeginRenderPass,
    ) -> Result<()> {
        for view in cmd.color_attachments.iter().flatten() {
            self.ensure_image_state(
                view.img,
                view.range,
                UsageBits::RT_WRITE,
                Layout::ColorAttachment,
            )?;
        }
        if let Some(depth) = cmd.depth_attachment {
            self.ensure_image_state(
                depth.img,
                depth.range,
                UsageBits::DEPTH_WRITE,
                Layout::DepthStencilAttachment,
            )?;
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
                let v = ctx.image_views.get_ref(view).ok_or(GPUError::SlotError())?;
                let img = ctx.images.get_mut_ref(v.img).ok_or(GPUError::SlotError())?;
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

        let (
            rp_raw,
            mut fb,
            attachment_formats,
            rp_subpass_samples,
            rp_initial_layouts,
            mut rp_width,
            mut rp_height,
        ) = {
            let rp_obj = self
                .ctx_ref()
                .render_passes
                .get_ref(cmd.render_pass)
                .ok_or(GPUError::SlotError())?;
            (
                rp_obj.raw,
                rp_obj.fb,
                rp_obj.attachment_formats.clone(),
                rp_obj.subpass_samples.clone(),
                rp_obj.attachment_initial_layouts.clone(),
                rp_obj.width,
                rp_obj.height,
            )
        };
        let mut target_width = cmd.viewport.scissor.w.max(1);
        let mut target_height = cmd.viewport.scissor.h.max(1);

        #[cfg(debug_assertions)]
        {
            if let Err(err) = self.validate_subpass_samples(
                &rp_subpass_samples,
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
            let handle = self.ctx_ref().get_or_create_image_view(view)?;
            let vk_view = {
                let v = self
                    .ctx_ref()
                    .image_views
                    .get_ref(handle)
                    .ok_or(GPUError::SlotError())?;
                v.view
            };
            attachments_vk.push(vk_view);
            self.curr_attachments
                .push((handle, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL));
        }

        if let Some(depth) = cmd.depth_attachment {
            let handle = self.ctx_ref().get_or_create_image_view(&depth)?;
            let vk_view = {
                let v = self
                    .ctx_ref()
                    .image_views
                    .get_ref(handle)
                    .ok_or(GPUError::SlotError())?;
                v.view
            };
            attachments_vk.push(vk_view);
            self.curr_attachments
                .push((handle, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL));
        }

        let mut attachment_min_width = u32::MAX;
        let mut attachment_min_height = u32::MAX;
        {
            let ctx = self.ctx_ref();
            for (view_handle, _) in &self.curr_attachments {
                let v = ctx
                    .image_views
                    .get_ref(*view_handle)
                    .ok_or(GPUError::SlotError())?;
                let info = ctx.image_info(v.img);
                attachment_min_width = attachment_min_width.min(info.dim[0]);
                attachment_min_height = attachment_min_height.min(info.dim[1]);
            }
        }

        if attachment_min_width != u32::MAX {
            target_width = target_width.min(attachment_min_width).max(1);
            target_height = target_height.min(attachment_min_height).max(1);
        }

        if target_width != rp_width || target_height != rp_height {
            let ctx = self.ctx_ref();
            let new_fb = ctx.create_imageless_framebuffer(
                rp_raw,
                &attachment_formats,
                target_width,
                target_height,
            )?;
            {
                let rp_mut = ctx
                    .render_passes
                    .get_mut_ref(cmd.render_pass)
                    .ok_or(GPUError::SlotError())?;
                unsafe {
                    ctx.device.destroy_framebuffer(rp_mut.fb, None);
                }
                rp_mut.fb = new_fb;
                rp_mut.width = target_width;
                rp_mut.height = target_height;
            }
            fb = new_fb;
            rp_width = target_width;
            rp_height = target_height;
        }

        for (index, (view_handle, _)) in self.curr_attachments.iter().enumerate() {
            let initial_layout = rp_initial_layouts
                .get(index)
                .copied()
                .unwrap_or(vk::ImageLayout::UNDEFINED);
            let ctx = self.ctx_ref();
            let v = ctx
                .image_views
                .get_ref(*view_handle)
                .ok_or(GPUError::SlotError())?;
            let img = ctx.images.get_mut_ref(v.img).ok_or(GPUError::SlotError())?;
            let base = v.range.base_mip_level as usize;
            let count = v.range.level_count as usize;
            for i in base..base + count {
                if let Some(l) = img.layouts.get_mut(i) {
                    *l = initial_layout;
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

        if let Some(dc) = cmd.depth_clear {
            clears.push(clear_value_to_vk(&dc));
        }

        unsafe {
            self.ctx_ref().device.cmd_begin_render_pass(
                self.cmd_buf,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(rp_raw)
                    .framebuffer(fb)
                    .render_area(convert_rect2d_to_vulkan(Rect2D {
                        x: 0,
                        y: 0,
                        w: rp_width,
                        h: rp_height,
                    }))
                    .clear_values(&clears)
                    .push_next(&mut attachment_info)
                    .build(),
                vk::SubpassContents::INLINE,
            );

            self.update_last_access(vk::PipelineStageFlags::NONE, vk::AccessFlags::empty());
        }

        Ok(())
    }

    fn begin_drawing(&mut self, cmd: &crate::gpu::driver::command::BeginDrawing) -> Result<()> {
        for view in cmd.color_attachments.iter().flatten() {
            self.ensure_image_state(
                view.img,
                view.range,
                UsageBits::RT_WRITE,
                Layout::ColorAttachment,
            )?;
        }
        if let Some(depth) = cmd.depth_attachment {
            self.ensure_image_state(
                depth.img,
                depth.range,
                UsageBits::DEPTH_WRITE,
                Layout::DepthStencilAttachment,
            )?;
        }
        let (pipeline_rp_layout, pipeline_subpass, layout_handle) = {
            let gfx = self
                .ctx_ref()
                .gfx_pipelines
                .get_ref(cmd.pipeline)
                .ok_or(GPUError::SlotError())?;
            (gfx.render_pass, gfx.subpass as usize, gfx.layout)
        };

        if self.curr_rp.map_or(false, |rp| rp == cmd.render_pass) {
            return Ok(());
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
                let v = ctx.image_views.get_ref(view).ok_or(GPUError::SlotError())?;
                let img = ctx.images.get_mut_ref(v.img).ok_or(GPUError::SlotError())?;
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

        let (
            rp_raw,
            mut fb,
            rp_subpass_samples,
            rp_subpass_formats,
            attachment_formats,
            rp_initial_layouts,
            mut rp_width,
            mut rp_height,
        ) = {
            let rp = self
                .ctx_ref()
                .render_passes
                .get_ref(cmd.render_pass)
                .ok_or(GPUError::SlotError())?;
            (
                rp.raw,
                rp.fb,
                rp.subpass_samples.clone(),
                rp.subpass_formats.clone(),
                rp.attachment_formats.clone(),
                rp.attachment_initial_layouts.clone(),
                rp.width,
                rp.height,
            )
        };
        let mut target_width = cmd.viewport.scissor.w.max(1);
        let mut target_height = cmd.viewport.scissor.h.max(1);

        #[cfg(debug_assertions)]
        {
            let pipeline_samples = {
                let layout = self
                    .ctx_ref()
                    .gfx_pipeline_layouts
                    .get_ref(layout_handle)
                    .ok_or(GPUError::SlotError())?;
                layout.sample_count
            };

            if let Err(err) = self.validate_subpass_samples(
                &rp_subpass_samples,
                pipeline_subpass,
                &cmd.color_attachments,
                cmd.depth_attachment,
                Some(pipeline_samples),
            ) {
                panic!("{}", err);
            }

            let pipeline_rp = self
                .ctx_ref()
                .render_passes
                .get_ref(pipeline_rp_layout)
                .ok_or(GPUError::SlotError())?;

            let pipeline_subpass_samples = pipeline_rp
                .subpass_samples
                .get(pipeline_subpass)
                .expect("pipeline subpass index should be valid");
            let pipeline_subpass_formats = pipeline_rp
                .subpass_formats
                .get(pipeline_subpass)
                .expect("pipeline subpass index should be valid");

            let rp_subpass_samples = rp_subpass_samples
                .get(pipeline_subpass)
                .expect("render pass subpass index should be valid");
            let rp_subpass_formats = rp_subpass_formats
                .get(pipeline_subpass)
                .expect("render pass subpass index should be valid");

            assert_eq!(
                rp_subpass_samples, pipeline_subpass_samples,
                "input render pass subpass samples do not match pipeline layout",
            );

            assert_eq!(
                rp_subpass_formats, pipeline_subpass_formats,
                "input render pass subpass formats do not match pipeline layout",
            );
        }

        let mut attachments_vk: Vec<vk::ImageView> = Vec::new();
        self.curr_attachments.clear();

        for view in cmd.color_attachments.iter().flatten() {
            let handle = self.ctx_ref().get_or_create_image_view(view)?;
            let vk_view = {
                let v = self
                    .ctx_ref()
                    .image_views
                    .get_ref(handle)
                    .ok_or(GPUError::SlotError())?;
                v.view
            };
            attachments_vk.push(vk_view);
            self.curr_attachments
                .push((handle, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL));
        }

        if let Some(depth) = cmd.depth_attachment {
            let handle = self.ctx_ref().get_or_create_image_view(&depth)?;
            let vk_view = {
                let v = self
                    .ctx_ref()
                    .image_views
                    .get_ref(handle)
                    .ok_or(GPUError::SlotError())?;
                v.view
            };
            attachments_vk.push(vk_view);
            self.curr_attachments
                .push((handle, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL));
        }

        let mut attachment_min_width = u32::MAX;
        let mut attachment_min_height = u32::MAX;
        {
            let ctx = self.ctx_ref();
            for (view_handle, _) in &self.curr_attachments {
                let v = ctx
                    .image_views
                    .get_ref(*view_handle)
                    .ok_or(GPUError::SlotError())?;
                let info = ctx.image_info(v.img);
                attachment_min_width = attachment_min_width.min(info.dim[0]);
                attachment_min_height = attachment_min_height.min(info.dim[1]);
            }
        }

        if attachment_min_width != u32::MAX {
            target_width = target_width.min(attachment_min_width).max(1);
            target_height = target_height.min(attachment_min_height).max(1);
        }

        if target_width != rp_width || target_height != rp_height {
            let ctx = self.ctx_ref();
            let new_fb = ctx.create_imageless_framebuffer(
                rp_raw,
                &attachment_formats,
                target_width,
                target_height,
            )?;
            {
                let rp_mut = ctx
                    .render_passes
                    .get_mut_ref(cmd.render_pass)
                    .ok_or(GPUError::SlotError())?;
                unsafe {
                    ctx.device.destroy_framebuffer(rp_mut.fb, None);
                }
                rp_mut.fb = new_fb;
                rp_mut.width = target_width;
                rp_mut.height = target_height;
            }
            fb = new_fb;
            rp_width = target_width;
            rp_height = target_height;
        }

        for (index, (view_handle, _)) in self.curr_attachments.iter().enumerate() {
            let initial_layout = rp_initial_layouts
                .get(index)
                .copied()
                .unwrap_or(vk::ImageLayout::UNDEFINED);
            let ctx = self.ctx_ref();
            let v = ctx
                .image_views
                .get_ref(*view_handle)
                .ok_or(GPUError::SlotError())?;
            let img = ctx.images.get_mut_ref(v.img).ok_or(GPUError::SlotError())?;
            let base = v.range.base_mip_level as usize;
            let count = v.range.level_count as usize;
            for i in base..base + count {
                if let Some(l) = img.layouts.get_mut(i) {
                    *l = initial_layout;
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
                    .render_area(convert_rect2d_to_vulkan(Rect2D {
                        x: 0,
                        y: 0,
                        w: rp_width,
                        h: rp_height,
                    }))
                    .clear_values(&clears)
                    .push_next(&mut attachment_info)
                    .build(),
                vk::SubpassContents::INLINE,
            );

            self.bind_graphics_pipeline(cmd.pipeline)?;
            self.update_last_access(vk::PipelineStageFlags::NONE, vk::AccessFlags::empty());
        }

        Ok(())
    }

    fn end_drawing(&mut self, _pass: &crate::gpu::driver::command::EndDrawing) -> Result<()> {
        unsafe { (*self.ctx).device.cmd_end_render_pass(self.cmd_buf) };
        self.last_op_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        self.last_op_access = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;

        let attachments = std::mem::take(&mut self.curr_attachments);
        for (view, layout) in attachments {
            let ctx = self.ctx_ref();
            let v = ctx.image_views.get_ref(view).ok_or(GPUError::SlotError())?;
            let img = ctx.images.get_mut_ref(v.img).ok_or(GPUError::SlotError())?;
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

        Ok(())
    }

    fn next_subpass(&mut self, _cmd: &crate::gpu::driver::command::NextSubpass) -> Result<()> {
        let curr = self.curr_subpass.unwrap_or(0);
        let rp = self.curr_rp.ok_or_else(|| GPUError::InvalidSubpass {
            subpass: curr as u32,
            available: 0,
        })?;

        let rp_obj = self
            .ctx_ref()
            .render_passes
            .get_ref(rp)
            .ok_or(GPUError::SlotError())?;

        let next = curr + 1;
        let available = rp_obj.subpass_samples.len();
        if next >= available {
            return Err(GPUError::InvalidSubpass {
                subpass: next as u32,
                available: available as u32,
            });
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

        Ok(())
    }

    fn bind_graphics_pipeline(
        &mut self,
        cmd: &crate::gpu::driver::command::BindGraphicsPipeline,
    ) -> Result<()> {
        self.bind_graphics_pipeline(cmd.pipeline)
    }

    fn update_graphics_pipeline_state(
        &mut self,
        cmd: &crate::gpu::driver::command::GraphicsPipelineStateUpdate,
    ) -> Result<()> {
        self.apply_graphics_pipeline_state_update(cmd);
        Ok(())
    }

    fn blit_image(&mut self, cmd: &crate::gpu::driver::command::BlitImage) -> Result<()> {
        self.ensure_image_state(
            cmd.src,
            cmd.src_range,
            UsageBits::COPY_SRC,
            Layout::TransferSrc,
        )?;
        self.ensure_image_state(
            cmd.dst,
            cmd.dst_range,
            UsageBits::COPY_DST,
            Layout::TransferDst,
        )?;
        unsafe {
            let src_data = self
                .ctx_ref()
                .images
                .get_ref(cmd.src)
                .ok_or(GPUError::SlotError())?;
            let dst_data = self
                .ctx_ref()
                .images
                .get_ref(cmd.dst)
                .ok_or(GPUError::SlotError())?;
            let src_info = self.ctx_ref().image_info(cmd.src);
            let dst_info = self.ctx_ref().image_info(cmd.dst);
            let src_dims = super::mip_dimensions(src_info.dim, cmd.src_range.base_mip);
            let dst_dims = super::mip_dimensions(dst_info.dim, cmd.dst_range.base_mip);
            let src_width = if cmd.src_region.w == 0 {
                src_dims[0]
            } else {
                cmd.src_region.w
            };
            let src_height = if cmd.src_region.h == 0 {
                src_dims[1]
            } else {
                cmd.src_region.h
            };
            let dst_width = if cmd.dst_region.w == 0 {
                dst_dims[0]
            } else {
                cmd.dst_region.w
            };
            let dst_height = if cmd.dst_region.h == 0 {
                dst_dims[1]
            } else {
                cmd.dst_region.h
            };
            let regions = [vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: image_aspect_for_format(src_info.format),
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
                        x: cmd.src_region.x.saturating_add(src_width) as i32,
                        y: cmd.src_region.y.saturating_add(src_height) as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: image_aspect_for_format(dst_info.format),
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
                        x: cmd.dst_region.x.saturating_add(dst_width) as i32,
                        y: cmd.dst_region.y.saturating_add(dst_height) as i32,
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
        Ok(())
    }

    fn resolve_image(&mut self, cmd: &crate::gpu::driver::command::MSImageResolve) -> Result<()> {
        self.ensure_image_state(
            cmd.src,
            cmd.src_range,
            UsageBits::COPY_SRC,
            Layout::TransferSrc,
        )?;
        self.ensure_image_state(
            cmd.dst,
            cmd.dst_range,
            UsageBits::COPY_DST,
            Layout::TransferDst,
        )?;

        unsafe {
            let src_data = self
                .ctx_ref()
                .images
                .get_ref(cmd.src)
                .ok_or(GPUError::SlotError())?;
            let dst_data = self
                .ctx_ref()
                .images
                .get_ref(cmd.dst)
                .ok_or(GPUError::SlotError())?;
            let src_info = self.ctx_ref().image_info(cmd.src);
            let dst_info = self.ctx_ref().image_info(cmd.dst);

            let width = match (cmd.src_region.w, cmd.dst_region.w) {
                (0, 0) => src_info.dim[0].min(dst_info.dim[0]),
                (s, 0) => s,
                (0, d) => d,
                (s, d) => s.min(d),
            }
            .max(1);
            let height = match (cmd.src_region.h, cmd.dst_region.h) {
                (0, 0) => src_info.dim[1].min(dst_info.dim[1]),
                (s, 0) => s,
                (0, d) => d,
                (s, d) => s.min(d),
            }
            .max(1);

            let src_mip = cmd.src_range.base_mip as usize;
            let dst_mip = cmd.dst_range.base_mip as usize;

            if src_info.samples == SampleCount::S1 {
                let regions = [vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: image_aspect_for_format(src_info.format),
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
                            x: cmd.src_region.x.saturating_add(width) as i32,
                            y: cmd.src_region.y.saturating_add(height) as i32,
                            z: 1,
                        },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: image_aspect_for_format(dst_info.format),
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
                            x: cmd.dst_region.x.saturating_add(width) as i32,
                            y: cmd.dst_region.y.saturating_add(height) as i32,
                            z: 1,
                        },
                    ],
                }];

                self.ctx_ref().device.cmd_blit_image(
                    self.cmd_buf,
                    src_data.img,
                    src_data.layouts[src_mip],
                    dst_data.img,
                    dst_data.layouts[dst_mip],
                    &regions,
                    vk::Filter::NEAREST,
                );
            } else {
                let regions = [vk::ImageResolve {
                    src_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: cmd.src_range.base_mip,
                        base_array_layer: cmd.src_range.base_layer,
                        layer_count: cmd.src_range.layer_count,
                    },
                    src_offset: vk::Offset3D {
                        x: cmd.src_region.x as i32,
                        y: cmd.src_region.y as i32,
                        z: 0,
                    },
                    dst_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: cmd.dst_range.base_mip,
                        base_array_layer: cmd.dst_range.base_layer,
                        layer_count: cmd.dst_range.layer_count,
                    },
                    dst_offset: vk::Offset3D {
                        x: cmd.dst_region.x as i32,
                        y: cmd.dst_region.y as i32,
                        z: 0,
                    },
                    extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                }];

                self.ctx_ref().device.cmd_resolve_image(
                    self.cmd_buf,
                    src_data.img,
                    src_data.layouts[src_mip],
                    dst_data.img,
                    dst_data.layouts[dst_mip],
                    &regions,
                );
            }

            self.update_last_access(
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            );
        }
        Ok(())
    }

    fn draw(&mut self, cmd: &crate::gpu::driver::command::Draw) -> Result<()> {
        self.ensure_active_render_pass()?;
        // self.ensure_buffer_state(cmd.vertices, UsageBits::VERTEX_READ);
        // self.ensure_binding_states(&cmd.bind_tables);

        if cmd.vertices.valid() {
            let v = self
                .ctx_ref()
                .buffers
                .get_ref(cmd.vertices)
                .ok_or(GPUError::SlotError())?;
            unsafe {
                self.ctx_ref().device.cmd_bind_vertex_buffers(
                    self.cmd_buf,
                    0,
                    &[v.buf],
                    &[v.offset as u64],
                );
            }
        }
        if let Some(pipe) = self.curr_pipeline {
            let p = self
                .ctx_ref()
                .gfx_pipelines
                .get_ref(pipe)
                .ok_or(GPUError::SlotError())?;
            let l = self
                .ctx_ref()
                .gfx_pipeline_layouts
                .get_ref(p.layout)
                .ok_or(GPUError::SlotError())?;

            unsafe {
                for (index, table) in cmd.bind_tables.iter().enumerate() {
                    if let Some(bt) = *table {
                        let bt_data = self
                            .ctx_ref()
                            .bind_tables
                            .get_ref(bt)
                            .ok_or(GPUError::SlotError())?;
                        let offsets = cmd
                            .dynamic_buffers
                            .get(index)
                            .and_then(|&d| d.map(|b| b.alloc.offset))
                            .into_iter()
                            .collect::<Vec<_>>();
                        self.ctx_ref().device.cmd_bind_descriptor_sets(
                            self.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            l.layout,
                            bt_data.set_id,
                            &[bt_data.set],
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
        Ok(())
    }

    fn draw_indexed(&mut self, cmd: &crate::gpu::driver::command::DrawIndexed) -> Result<()> {
        self.ensure_active_render_pass()?;
        // self.ensure_buffer_state(cmd.vertices, UsageBits::VERTEX_READ);
        // self.ensure_buffer_state(cmd.indices, UsageBits::INDEX_READ);
        // self.ensure_binding_states(&cmd.bind_tables);
        //

        if cmd.vertices.valid() && cmd.indices.valid() {
            let v = self
                .ctx_ref()
                .buffers
                .get_ref(cmd.vertices)
                .ok_or(GPUError::SlotError())?;
            let i = self
                .ctx_ref()
                .buffers
                .get_ref(cmd.indices)
                .ok_or(GPUError::SlotError())?;
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
        }
        if let Some(pipe) = self.curr_pipeline {
            let p = self
                .ctx_ref()
                .gfx_pipelines
                .get_ref(pipe)
                .ok_or(GPUError::SlotError())?;
            let l = self
                .ctx_ref()
                .gfx_pipeline_layouts
                .get_ref(p.layout)
                .ok_or(GPUError::SlotError())?;

            unsafe {
                for (index, table) in cmd.bind_tables.iter().enumerate() {
                    if let Some(bt) = *table {
                        let bt_data = self
                            .ctx_ref()
                            .bind_tables
                            .get_ref(bt)
                            .ok_or(GPUError::SlotError())?;
                        let offsets = cmd
                            .dynamic_buffers
                            .get(index)
                            .and_then(|&d| d.map(|b| b.alloc.offset))
                            .into_iter()
                            .collect::<Vec<_>>();
                        self.ctx_ref().device.cmd_bind_descriptor_sets(
                            self.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            l.layout,
                            bt_data.set_id,
                            &[bt_data.set],
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
        Ok(())
    }

    fn draw_indirect(&mut self, cmd: &crate::gpu::driver::command::DrawIndirect) -> Result<()> {
        self.ensure_active_render_pass()?;
        if cmd.vertices.valid() {
            let v = self
                .ctx_ref()
                .buffers
                .get_ref(cmd.vertices)
                .ok_or(GPUError::SlotError())?;
            unsafe {
                self.ctx_ref().device.cmd_bind_vertex_buffers(
                    self.cmd_buf,
                    0,
                    &[v.buf],
                    &[v.offset as u64],
                );
            }
        }

        if let Some(pipe) = self.curr_pipeline {
            let p = self
                .ctx_ref()
                .gfx_pipelines
                .get_ref(pipe)
                .ok_or(GPUError::SlotError())?;
            let l = self
                .ctx_ref()
                .gfx_pipeline_layouts
                .get_ref(p.layout)
                .ok_or(GPUError::SlotError())?;

            unsafe {
                for (index, table) in cmd.bind_tables.iter().enumerate() {
                    if let Some(bt) = *table {
                        let bt_data = self
                            .ctx_ref()
                            .bind_tables
                            .get_ref(bt)
                            .ok_or(GPUError::SlotError())?;
                        let offsets = cmd
                            .dynamic_buffers
                            .get(index)
                            .and_then(|&d| d.map(|b| b.alloc.offset))
                            .into_iter()
                            .collect::<Vec<_>>();
                        self.ctx_ref().device.cmd_bind_descriptor_sets(
                            self.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            l.layout,
                            bt_data.set_id,
                            &[bt_data.set],
                            &offsets,
                        );
                    }
                }
            }
        }

        let indirect = self
            .ctx_ref()
            .buffers
            .get_ref(cmd.indirect)
            .ok_or(GPUError::SlotError())?;

        unsafe {
            self.ctx_ref().device.cmd_draw_indirect(
                self.cmd_buf,
                indirect.buf,
                indirect.offset as u64 + cmd.offset as u64,
                cmd.draw_count,
                cmd.stride,
            );
            self.update_last_access(
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            );
        }

        Ok(())
    }

    fn draw_indexed_indirect(
        &mut self,
        cmd: &crate::gpu::driver::command::DrawIndexedIndirect,
    ) -> Result<()> {
        self.ensure_active_render_pass()?;
        if cmd.vertices.valid() {
            let v = self
                .ctx_ref()
                .buffers
                .get_ref(cmd.vertices)
                .ok_or(GPUError::SlotError())?;
            unsafe {
                self.ctx_ref().device.cmd_bind_vertex_buffers(
                    self.cmd_buf,
                    0,
                    &[v.buf],
                    &[v.offset as u64],
                );
            }
        }

        if cmd.indices.valid() {
            let i = self
                .ctx_ref()
                .buffers
                .get_ref(cmd.indices)
                .ok_or(GPUError::SlotError())?;

            unsafe {
                self.ctx_ref().device.cmd_bind_index_buffer(
                    self.cmd_buf,
                    i.buf,
                    i.offset as u64,
                    vk::IndexType::UINT32,
                );
            }
        }

        if let Some(pipe) = self.curr_pipeline {
            let p = self
                .ctx_ref()
                .gfx_pipelines
                .get_ref(pipe)
                .ok_or(GPUError::SlotError())?;
            let l = self
                .ctx_ref()
                .gfx_pipeline_layouts
                .get_ref(p.layout)
                .ok_or(GPUError::SlotError())?;

            unsafe {
                for (index, table) in cmd.bind_tables.iter().enumerate() {
                    if let Some(bt) = *table {
                        let bt_data = self
                            .ctx_ref()
                            .bind_tables
                            .get_ref(bt)
                            .ok_or(GPUError::SlotError())?;
                        let offsets = cmd
                            .dynamic_buffers
                            .get(index)
                            .and_then(|&d| d.map(|b| b.alloc.offset))
                            .into_iter()
                            .collect::<Vec<_>>();
                        self.ctx_ref().device.cmd_bind_descriptor_sets(
                            self.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            l.layout,
                            bt_data.set_id,
                            &[bt_data.set],
                            &offsets,
                        );
                    }
                }
            }
        }

        let indirect = self
            .ctx_ref()
            .buffers
            .get_ref(cmd.indirect)
            .ok_or(GPUError::SlotError())?;

        unsafe {
            self.ctx_ref().device.cmd_draw_indexed_indirect(
                self.cmd_buf,
                indirect.buf,
                indirect.offset as u64 + cmd.offset as u64,
                cmd.draw_count,
                cmd.stride,
            );
            self.update_last_access(
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            );
        }

        Ok(())
    }

    fn dispatch(&mut self, cmd: &crate::gpu::driver::command::Dispatch) -> Result<()> {
        self.ensure_binding_states(&cmd.bind_tables)?;
        unsafe {
            self.bind_compute_pipeline(cmd.pipeline)?;
            let layout_handle = self
                .ctx_ref()
                .compute_pipelines
                .get_ref(cmd.pipeline)
                .ok_or(GPUError::SlotError())?
                .layout;

            let layout = self
                .ctx_ref()
                .compute_pipeline_layouts
                .get_ref(layout_handle)
                .ok_or(GPUError::SlotError())?;

            unsafe {
                for (index, table) in cmd.bind_tables.iter().enumerate() {
                    if let Some(bt) = *table {
                        let bt_data = self
                            .ctx_ref()
                            .bind_tables
                            .get_ref(bt)
                            .ok_or(GPUError::SlotError())?;
                        let offsets = cmd
                            .dynamic_buffers
                            .get(index)
                            .and_then(|&d| d.map(|b| b.alloc.offset))
                            .into_iter()
                            .collect::<Vec<_>>();
                        self.ctx_ref().device.cmd_bind_descriptor_sets(
                            self.cmd_buf,
                            vk::PipelineBindPoint::COMPUTE,
                            layout.layout,
                            bt_data.set_id,
                            &[bt_data.set],
                            &offsets,
                        );
                    }
                }

                self.ctx_ref()
                    .device
                    .cmd_dispatch(self.cmd_buf, cmd.x, cmd.y, cmd.z);
                self.update_last_access(
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::SHADER_WRITE,
                );
            }
        }
        Ok(())
    }

    fn copy_buffer(&mut self, cmd: &crate::gpu::driver::command::CopyBuffer) -> Result<()> {
        self.ensure_buffer_state(cmd.src, UsageBits::COPY_SRC)?;
        self.ensure_buffer_state(cmd.dst, UsageBits::COPY_DST)?;
        unsafe {
            let src = self
                .ctx_ref()
                .buffers
                .get_ref(cmd.src)
                .ok_or(GPUError::SlotError())?;
            let dst = self
                .ctx_ref()
                .buffers
                .get_ref(cmd.dst)
                .ok_or(GPUError::SlotError())?;
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
        Ok(())
    }

    fn copy_buffer_to_image(
        &mut self,
        cmd: &crate::gpu::driver::command::CopyBufferImage,
    ) -> Result<()> {
        self.ensure_buffer_state(cmd.src, UsageBits::COPY_SRC)?;
        self.ensure_image_state(cmd.dst, cmd.range, UsageBits::COPY_DST, Layout::TransferDst)?;
        unsafe {
            let img_data = self
                .ctx_ref()
                .images
                .get_ref(cmd.dst)
                .ok_or(GPUError::SlotError())?;
            let img_info = self.ctx_ref().image_info(cmd.dst);
            let mip = cmd.range.base_mip as usize;
            let dims = crate::gpu::mip_dimensions(img_info.dim, cmd.range.base_mip);
            self.ctx_ref().device.cmd_copy_buffer_to_image(
                self.cmd_buf,
                self.ctx_ref()
                    .buffers
                    .get_ref(cmd.src)
                    .ok_or(GPUError::SlotError())?
                    .buf,
                img_data.img,
                img_data.layouts[mip],
                &[vk::BufferImageCopy {
                    buffer_offset: cmd.src_offset as u64,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: image_aspect_for_format(img_info.format),
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
        Ok(())
    }

    fn copy_image_to_buffer(
        &mut self,
        cmd: &crate::gpu::driver::command::CopyImageBuffer,
    ) -> Result<()> {
        self.ensure_image_state(cmd.src, cmd.range, UsageBits::COPY_SRC, Layout::TransferSrc)?;
        self.ensure_buffer_state(cmd.dst, UsageBits::COPY_DST)?;
        unsafe {
            let img_data = self
                .ctx_ref()
                .images
                .get_ref(cmd.src)
                .ok_or(GPUError::SlotError())?;
            let img_info = self.ctx_ref().image_info(cmd.src);
            let mip = cmd.range.base_mip as usize;

            let dims = crate::gpu::mip_dimensions(img_info.dim, cmd.range.base_mip);
            self.ctx_ref().device.cmd_copy_image_to_buffer(
                self.cmd_buf,
                img_data.img,
                img_data.layouts[mip],
                self.ctx_ref()
                    .buffers
                    .get_ref(cmd.dst)
                    .ok_or(GPUError::SlotError())?
                    .buf,
                &[vk::BufferImageCopy {
                    buffer_offset: cmd.dst_offset as u64,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: image_aspect_for_format(img_info.format),
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
        Ok(())
    }

    fn copy_image(&mut self, _cmd: &crate::gpu::driver::command::CopyImage) -> Result<()> {
        todo!()
    }

    fn transition_image(
        &mut self,
        cmd: &crate::gpu::driver::command::TransitionImage,
    ) -> Result<()> {
        self.ensure_image_state(cmd.image, cmd.range, cmd.usage, cmd.layout)?;
        Ok(())
    }

    fn prepare_buffer(&mut self, cmd: &crate::gpu::driver::command::PrepareBuffer) -> Result<()> {
        self.ensure_buffer_state_on_queue(cmd.buffer, cmd.usage, cmd.queue)?;
        Ok(())
    }

    fn sync_point(&mut self, cmd: &crate::gpu::driver::command::SyncPointCmd) -> Result<()> {
        let point = SyncPoint::from_u8(cmd.point)
            .ok_or(GPUError::Unimplemented("invalid sync point"))?;
        let scope =
            Scope::from_u8(cmd.scope).ok_or(GPUError::Unimplemented("invalid sync scope"))?;
        let (src_stage, dst_stage) = sync_point_stages(point);
        let src_access = sync_point_src_access(point);
        let dst_access = sync_scope_access(scope);
        let barrier = vk::MemoryBarrier::builder()
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .build();

        unsafe {
            self.ctx_ref().device.cmd_pipeline_barrier(
                self.cmd_buf,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            )
        };
        self.update_last_access(dst_stage, dst_access);
        Ok(())
    }

    fn submit(&mut self, cmd: &crate::SubmitInfo2) -> Result<Handle<Fence>> {
        self.submit(cmd)
    }

    fn debug_marker_begin(
        &mut self,
        _cmd: &crate::gpu::driver::command::DebugMarkerBegin,
    ) -> Result<()> {
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
        Ok(())
    }

    fn debug_marker_end(
        &mut self,
        _cmd: &crate::gpu::driver::command::DebugMarkerEnd,
    ) -> Result<()> {
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
        Ok(())
    }

    fn gpu_timer_begin(&mut self, cmd: &crate::gpu::driver::command::GpuTimerBegin) -> Result<()> {
        let ctx = self.ctx;
        if !ctx.is_null() {
            unsafe { (*ctx).gpu_timer_begin(self, cmd.frame as usize) };
        }
        Ok(())
    }

    fn gpu_timer_end(&mut self, cmd: &crate::gpu::driver::command::GpuTimerEnd) -> Result<()> {
        let ctx = self.ctx;
        if !ctx.is_null() {
            unsafe { (*ctx).gpu_timer_end(self, cmd.frame as usize) };
        }
        Ok(())
    }
}
