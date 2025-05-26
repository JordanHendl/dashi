//! Refactored command helpers for Dashi CommandList

use ash::vk;
use std::hash::{DefaultHasher, Hash, Hasher};

use super::{convert_barrier_point_vk, convert_rect2d_to_vulkan, ImageView, RenderPass};
use crate::utils::Handle;
use crate::{
    Attachment, BarrierPoint, BindGroup, Buffer, CommandList, ComputePipeline, Context,
    DynamicBuffer, Filter, GPUError, GraphicsPipeline, IndexedBindGroup, IndexedIndirectCommand,
    IndirectCommand, Rect2D, SubpassContainer, Viewport,
};
// Converts an Attachment's clear value into a Vulkan clear value
impl Attachment {
    fn to_vk_clear_value(&self) -> vk::ClearValue {
        match self.clear {
            crate::ClearValue::Color(cols) => vk::ClearValue {
                color: vk::ClearColorValue { float32: cols },
            },
            crate::ClearValue::IntColor(cols) => vk::ClearValue {
                color: vk::ClearColorValue { int32: cols },
            },
            crate::ClearValue::UintColor(cols) => vk::ClearValue {
                color: vk::ClearColorValue { uint32: cols },
            },
            crate::ClearValue::DepthStencil { depth, stencil } => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
            },
        }
    }
}
#[derive(Clone)]
pub struct BufferCopy {
    pub src: Handle<Buffer>,
    pub dst: Handle<Buffer>,
    pub src_offset: usize,
    pub dst_offset: usize,
    pub size: usize,
}

impl Default for BufferCopy {
    fn default() -> Self {
        Self {
            src: Default::default(),
            dst: Default::default(),
            src_offset: 0,
            dst_offset: 0,
            size: 0,
        }
    }
}

#[derive(Clone, Default)]
pub struct ImageBufferCopy {
    pub src: Handle<ImageView>,
    pub dst: Handle<Buffer>,
    pub dst_offset: usize,
}

#[derive(Clone, Default)]
pub struct BufferImageCopy {
    pub src: Handle<Buffer>,
    pub dst: Handle<ImageView>,
    pub src_offset: usize,
}

#[derive(Clone)]
pub struct ImageBlit {
    pub src: Handle<ImageView>,
    pub dst: Handle<ImageView>,
    pub filter: Filter,
    pub src_region: Rect2D,
    pub dst_region: Rect2D,
}

impl Default for ImageBlit {
    fn default() -> Self {
        Self {
            src: Default::default(),
            dst: Default::default(),
            filter: Filter::Nearest,
            src_region: Default::default(),
            dst_region: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct Dispatch {
    pub compute: Handle<ComputePipeline>,
    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
    pub bind_groups: [Option<Handle<BindGroup>>; 4],
    pub workgroup_size: [u32; 3],
}

impl Default for Dispatch {
    fn default() -> Self {
        Self {
            compute: Default::default(),
            dynamic_buffers: Default::default(),
            bind_groups: Default::default(),
            workgroup_size: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct DrawBegin<'a> {
    pub viewport: Viewport,
    pub pipeline: Handle<GraphicsPipeline>,
    pub attachments: &'a [Attachment],
}

#[derive(Clone, Default)]
pub struct Draw {
    pub vertices: Handle<Buffer>,
    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
    pub bind_groups: [Option<Handle<BindGroup>>; 4],
    pub instance_count: u32,
    pub count: u32,
}

#[derive(Clone)]
pub struct DrawIndexedDynamic {
    pub vertices: DynamicBuffer,
    pub indices: DynamicBuffer,
    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
    pub bind_groups: [Option<Handle<BindGroup>>; 4],
    pub index_count: u32,
    pub instance_count: u32,
    pub first_instance: u32,
}

impl Default for DrawIndexedDynamic {
    fn default() -> Self {
        Self {
            vertices: Default::default(),
            indices: Default::default(),
            index_count: Default::default(),
            instance_count: 1,
            first_instance: 0,
            bind_groups: [None, None, None, None],
            dynamic_buffers: [None, None, None, None],
        }
    }
}

#[derive(Clone)]
pub struct BindIndexedBG {
    pub bind_groups: [Option<Handle<IndexedBindGroup>>; 4],
}

#[derive(Clone, Default)]
pub struct RenderPassBegin<'a> {
    pub render_pass: Handle<RenderPass>,
    pub viewport: Viewport,
    pub attachments: &'a [Attachment],
}

#[derive(Clone)]
pub struct DrawIndexed {
    pub vertices: Handle<Buffer>,
    pub indices: Handle<Buffer>,
    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
    pub bind_groups: [Option<Handle<BindGroup>>; 4],
    pub index_count: u32,
    pub instance_count: u32,
    pub first_instance: u32,
}

impl Default for DrawIndexed {
    fn default() -> Self {
        Self {
            vertices: Default::default(),
            indices: Default::default(),
            index_count: Default::default(),
            instance_count: 1,
            first_instance: 0,
            bind_groups: [None, None, None, None],
            dynamic_buffers: [None, None, None, None],
        }
    }
}

#[derive(Clone)]
pub struct DrawIndexedIndirect {
    pub draw_params: Handle<Buffer>,
    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
    pub bind_groups: [Option<Handle<BindGroup>>; 4],
    pub offset: u32,
    pub draw_count: u32,
    pub stride: u32,
}

impl Default for DrawIndexedIndirect {
    fn default() -> Self {
        Self {
            draw_params: Default::default(),
            bind_groups: [None, None, None, None],
            dynamic_buffers: [None, None, None, None],
            offset: 0,
            draw_count: 0,
            stride: std::mem::size_of::<IndexedIndirectCommand>() as u32,
        }
    }
}

#[derive(Clone)]
pub struct DrawIndirect {
    pub draw_params: Handle<Buffer>,
    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
    pub bind_groups: [Option<Handle<BindGroup>>; 4],
    pub offset: u32,
    pub draw_count: u32,
    pub stride: u32,
}

impl Default for DrawIndirect {
    fn default() -> Self {
        Self {
            draw_params: Default::default(),
            bind_groups: [None, None, None, None],
            dynamic_buffers: [None, None, None, None],
            offset: 0,
            draw_count: 0,
            stride: std::mem::size_of::<IndirectCommand>() as u32,
        }
    }
}

#[derive(Clone, Copy)]
pub struct BindPipeline {
    pub gfx: Option<Handle<GraphicsPipeline>>,
    pub compute: Option<Handle<ComputePipeline>>,
    pub bg: Handle<BindGroup>,
}

#[derive(Clone, Copy)]
pub struct ImageBarrier {
    pub view: Handle<ImageView>,
    pub src: BarrierPoint,
    pub dst: BarrierPoint,
}

#[derive(Clone)]
pub enum Command {
    BufferCopy(BufferCopy),
    BufferImageCopy(BufferImageCopy),
    ImageBufferCopy(ImageBufferCopy),
    Draw(Draw),
    DrawIndexed(DrawIndexed),
    DrawIndexedDynamic(DrawIndexedDynamic),
    DrawIndexedIndirect(DrawIndexedIndirect),
    DrawIndirect(DrawIndirect),
    Blit(ImageBlit),
    ImageBarrier(ImageBarrier),
    Dispatch(Dispatch),
}

impl CommandList {
    /// Append a generic command to the list and mark dirty.
    pub fn append(&mut self, cmd: Command) {
        if self.ctx.is_null() {
            return;
        }
        match cmd {
            Command::BufferCopy(c) => self.copy_buffer(c),
            Command::BufferImageCopy(c) => self.copy_buffer_to_image(c),
            Command::ImageBufferCopy(c) => self.copy_image_to_buffer(c),
            Command::Blit(c) => self.blit_image(c),
            Command::ImageBarrier(b) => self.image_barrier(b),
            Command::Dispatch(d) => self.dispatch_compute(d),
            Command::Draw(d) => self.draw(d),
            Command::DrawIndexed(d) => self.draw_indexed(d),
            Command::DrawIndexedDynamic(d) => self.draw_indexed_dynamic(d),
            Command::DrawIndexedIndirect(d) => self.draw_indexed_indirect(d),
            Command::DrawIndirect(d) => self.draw_indirect(d),
        }
        self.dirty = true;
    }

    pub fn copy_buffer(&mut self, info: BufferCopy) {
        unsafe {
            let src = self.ctx_ref().buffers.get_ref(info.src).unwrap();
            let dst = self.ctx_ref().buffers.get_ref(info.dst).unwrap();
            self.ctx_ref().device.cmd_copy_buffer(
                self.cmd_buf,
                src.buf,
                dst.buf,
                &[vk::BufferCopy {
                    src_offset: info.src_offset as u64,
                    dst_offset: info.dst_offset as u64,
                    size: if info.size == 0 {
                        src.size.min(dst.size) as u64
                    } else {
                        info.size as u64
                    },
                }],
            );
            self.update_last_access(
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_WRITE,
            );
        }
    }

    pub fn copy_buffer_to_image(&mut self, rec: BufferImageCopy) {
        unsafe {
            let view_data = self.ctx_ref().image_views.get_ref(rec.dst).unwrap();
            let img_data = self.ctx_ref().images.get_ref(view_data.img).unwrap();
            let old_layout = img_data.layout;
            self.transition_image(
                rec.dst,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            );

            self.ctx_ref().device.cmd_copy_buffer_to_image(
                self.cmd_buf,
                self.ctx_ref().buffers.get_ref(rec.src).unwrap().buf,
                img_data.img,
                img_data.layout,
                &[vk::BufferImageCopy {
                    buffer_offset: rec.src_offset as u64,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: img_data.sub_layers.aspect_mask,
                        mip_level: view_data.range.base_mip_level,
                        base_array_layer: view_data.range.base_array_layer,
                        layer_count: view_data.range.layer_count,
                    },
                    image_extent: img_data.extent,
                    ..Default::default()
                }],
            );

            self.transition_image_layout(
                rec.dst,
                old_layout,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
            );
            self.update_last_access(
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_WRITE,
            );
        }
    }

    pub fn copy_image_to_buffer(&mut self, rec: ImageBufferCopy) {
        unsafe {
            let view_data = self.ctx_ref().image_views.get_ref(rec.src).unwrap();
            let img_data = self.ctx_ref().images.get_ref(view_data.img).unwrap();
            let old_layout = img_data.layout;
            self.transition_image(
                rec.src,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_READ,
            );

            self.ctx_ref().device.cmd_copy_image_to_buffer(
                self.cmd_buf,
                img_data.img,
                img_data.layout,
                self.ctx_ref().buffers.get_ref(rec.dst).unwrap().buf,
                &[vk::BufferImageCopy {
                    buffer_offset: rec.dst_offset as u64,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: img_data.sub_layers.aspect_mask,
                        mip_level: view_data.range.base_mip_level,
                        base_array_layer: view_data.range.base_array_layer,
                        layer_count: view_data.range.layer_count,
                    },
                    image_extent: img_data.extent,
                    ..Default::default()
                }],
            );

            self.transition_image_layout(
                rec.src,
                old_layout,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_READ,
            );
            self.update_last_access(
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::MEMORY_WRITE,
            );
        }
    }

    pub fn blit_image(&mut self, cmd: ImageBlit) {
        unsafe {
            let src_data = self
                .ctx_ref()
                .images
                .get_ref(self.ctx_ref().image_views.get_ref(cmd.src).unwrap().img)
                .unwrap();
            let dst_data = self
                .ctx_ref()
                .images
                .get_ref(self.ctx_ref().image_views.get_ref(cmd.dst).unwrap().img)
                .unwrap();
            let regions = [vk::ImageBlit {
                src_subresource: src_data.sub_layers,
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
                dst_subresource: dst_data.sub_layers,
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
            let src_layout = src_data.layout;
            let dst_layout = dst_data.layout;

            self.transition_image_layout(
                cmd.src,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TRANSFER,
                self.last_op_access,
            );
            self.transition_image_layout(
                cmd.dst,
                vk::ImageLayout::GENERAL,
                vk::PipelineStageFlags::TRANSFER,
                self.last_op_access,
            );
            self.ctx_ref().device.cmd_blit_image(
                self.cmd_buf,
                src_data.img,
                src_data.layout,
                dst_data.img,
                dst_data.layout,
                &regions,
                cmd.filter.into(),
            );

            self.transition_image_layout(
                cmd.src,
                src_layout,
                vk::PipelineStageFlags::TRANSFER,
                self.last_op_access,
            );
            self.transition_image_layout(
                cmd.dst,
                dst_layout,
                vk::PipelineStageFlags::TRANSFER,
                self.last_op_access,
            );
            self.update_last_access(
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::TRANSFER_WRITE,
            );
        }
    }

    pub fn image_barrier(&mut self, barrier: ImageBarrier) {
        unsafe {
            let view_data = self.ctx_ref().image_views.get_ref(barrier.view).unwrap();
            let img_data = self.ctx_ref().images.get_ref(view_data.img).unwrap();
            self.ctx_ref().device.cmd_pipeline_barrier(
                self.cmd_buf,
                self.last_op_stage,
                convert_barrier_point_vk(barrier.dst),
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .old_layout(img_data.layout)
                    .new_layout(img_data.layout)
                    .src_access_mask(self.last_op_access)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .image(img_data.img)
                    .subresource_range(view_data.range)
                    .build()],
            );
        }
    }

    pub fn dispatch_compute(&mut self, cmd: Dispatch) {
        unsafe {
            // Bind pipeline
            let pipeline = self
                .ctx_ref()
                .compute_pipelines
                .get_ref(cmd.compute)
                .unwrap();
            let layout = self
                .ctx_ref()
                .compute_pipeline_layouts
                .get_ref(pipeline.layout)
                .unwrap()
                .layout;
            self.ctx_ref().device.cmd_bind_pipeline(
                self.cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.raw,
            );

            // Bind descriptor sets with dynamic offsets
            for (set_idx, bg_opt) in cmd.bind_groups.iter().enumerate() {
                if let Some(bg) = bg_opt {
                    let bg_data = self.ctx_ref().bind_groups.get_ref(*bg).unwrap();
                    let offsets: Vec<u32> = cmd
                        .dynamic_buffers
                        .get(set_idx)
                        .iter()
                        .filter_map(|&d| d.map(|b| b.alloc.offset))
                        .collect();
                    self.ctx_ref().device.cmd_bind_descriptor_sets(
                        self.cmd_buf,
                        vk::PipelineBindPoint::COMPUTE,
                        layout,
                        bg_data.set_id,
                        &[bg_data.set],
                        &offsets,
                    );
                }
            }

            // Dispatch
            self.ctx_ref().device.cmd_dispatch(
                self.cmd_buf,
                cmd.workgroup_size[0],
                cmd.workgroup_size[1],
                cmd.workgroup_size[2],
            );
            self.update_last_access(
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_WRITE,
            );
        }
    }

    // Helper: begin render pass if needed
    pub fn begin_render_pass(&mut self, info: &RenderPassBegin) -> Result<(), GPUError> {
        if self.curr_rp.map_or(false, |rp| rp == info.render_pass) {
            return Ok(());
        }
        // end previous pass
        if self.curr_rp.is_some() {
            unsafe {
                self.ctx_ref().device.cmd_end_render_pass(self.cmd_buf);
            }
        }
        self.curr_rp = Some(info.render_pass);

        unsafe {
            let rp_obj = self
                .ctx_ref()
                .render_passes
                .get_mut_ref(info.render_pass)
                .unwrap();
            let fb_info = self.create_or_cache_framebuffer(rp_obj, info)?.clone();
            self.ctx_ref().device.cmd_begin_render_pass(
                self.cmd_buf,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(rp_obj.raw)
                    .framebuffer(fb_info.fb)
                    .render_area(convert_rect2d_to_vulkan(info.viewport.scissor))
                    .clear_values(&fb_info.clear_values)
                    .build(),
                vk::SubpassContents::INLINE,
            );
            self.update_last_access(vk::PipelineStageFlags::NONE, vk::AccessFlags::empty());
        }
        Ok(())
    }

    pub fn draw(&mut self, cmd: Draw) {
        unsafe {
            self.bind_draw_descriptor_sets(&cmd.dynamic_buffers, &cmd.bind_groups);
            let buf = self.ctx_ref().buffers.get_ref(cmd.vertices).unwrap();
            static OFFSET: vk::DeviceSize = 0;
            self.ctx_ref()
                .device
                .cmd_bind_vertex_buffers(self.cmd_buf, 0, &[buf.buf], &[OFFSET]);
            self.ctx_ref()
                .device
                .cmd_draw(self.cmd_buf, cmd.count, cmd.instance_count, 0, 0);
            self.update_last_access(
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            );
        }
    }

    pub fn draw_indexed(&mut self, cmd: DrawIndexed) {
        unsafe {
            self.bind_draw_descriptor_sets(&cmd.dynamic_buffers, &cmd.bind_groups);
            let v_buf = self.ctx_ref().buffers.get_ref(cmd.vertices).unwrap();
            let i_buf = self.ctx_ref().buffers.get_ref(cmd.indices).unwrap();
            static OFFSET: vk::DeviceSize = 0;
            self.ctx_ref()
                .device
                .cmd_bind_vertex_buffers(self.cmd_buf, 0, &[v_buf.buf], &[OFFSET]);
            self.ctx_ref().device.cmd_bind_index_buffer(
                self.cmd_buf,
                i_buf.buf,
                OFFSET,
                vk::IndexType::UINT32,
            );
            self.ctx_ref().device.cmd_draw_indexed(
                self.cmd_buf,
                cmd.index_count,
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

    pub fn draw_indexed_dynamic(&mut self, cmd: DrawIndexedDynamic) {
        unsafe {
            self.bind_draw_descriptor_sets(&cmd.dynamic_buffers, &cmd.bind_groups);
            let v_buf = self
                .ctx_ref()
                .buffers
                .get_ref(cmd.vertices.handle())
                .unwrap();
            let i_buf = self
                .ctx_ref()
                .buffers
                .get_ref(cmd.indices.handle())
                .unwrap();
            let vb_offset = cmd.vertices.offset() as u64;
            let ib_offset = cmd.indices.offset() as u64;
            self.ctx_ref().device.cmd_bind_vertex_buffers(
                self.cmd_buf,
                0,
                &[v_buf.buf],
                &[vb_offset],
            );
            self.ctx_ref().device.cmd_bind_index_buffer(
                self.cmd_buf,
                i_buf.buf,
                ib_offset,
                vk::IndexType::UINT32,
            );
            self.ctx_ref().device.cmd_draw_indexed(
                self.cmd_buf,
                cmd.index_count,
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

    pub fn draw_indexed_indirect(&mut self, cmd: DrawIndexedIndirect) {
        unsafe {
            let buf = self.ctx_ref().buffers.get_ref(cmd.draw_params).unwrap();
            self.ctx_ref().device.cmd_draw_indexed_indirect(
                self.cmd_buf,
                buf.buf,
                cmd.offset as u64,
                cmd.draw_count,
                cmd.stride,
            );
            self.update_last_access(
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            );
        }
    }

    pub fn draw_indirect(&mut self, cmd: DrawIndirect) {
        unsafe {
            let buf = self.ctx_ref().buffers.get_ref(cmd.draw_params).unwrap();
            self.ctx_ref().device.cmd_draw_indirect(
                self.cmd_buf,
                buf.buf,
                cmd.offset as u64,
                cmd.draw_count,
                cmd.stride,
            );
            self.update_last_access(
                vk::PipelineStageFlags::VERTEX_SHADER,
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            );
        }
    }

    // Helper: bind descriptor sets for drawing
    pub fn bind_draw_descriptor_sets(
        &mut self,
        dyn_bufs: &[Option<DynamicBuffer>; 4],
        bgs: &[Option<Handle<BindGroup>>; 4],
    ) {
        for (i, bg_opt) in bgs.iter().enumerate() {
            if let Some(bg) = bg_opt {
                let bg_data = self.ctx_ref().bind_groups.get_ref(*bg).unwrap();
                let offsets: Vec<u32> = dyn_bufs
                    .iter()
                    .filter_map(|&d| d.map(|b| b.alloc.offset))
                    .collect();
                let p = self
                    .ctx_ref()
                    .gfx_pipelines
                    .get_ref(self.curr_pipeline.unwrap())
                    .unwrap();
                let p_layout = self
                    .ctx_ref()
                    .gfx_pipeline_layouts
                    .get_ref(p.layout)
                    .unwrap()
                    .layout;
                unsafe {
                    self.ctx_ref().device.cmd_bind_descriptor_sets(
                        self.cmd_buf,
                        vk::PipelineBindPoint::GRAPHICS,
                        p_layout,
                        bg_data.set_id,
                        &[bg_data.set],
                        &offsets,
                    );
                }
            }
        }
    }

    pub fn begin_drawing(&mut self, info: &DrawBegin) -> Result<(), GPUError> {
        let pipeline = info.pipeline;
        if let Some(gfx) = self.curr_pipeline {
            if pipeline == gfx {
                return Ok(());
            }
        }

        unsafe {
            self.curr_pipeline = Some(pipeline);
            let gfx = (*self.ctx).gfx_pipelines.get_ref(pipeline).unwrap();
            self.begin_render_pass(&RenderPassBegin {
                render_pass: gfx.render_pass,
                viewport: info.viewport,
                attachments: info.attachments,
            })?;
            (*self.ctx).device.cmd_bind_pipeline(
                self.cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                gfx.raw,
            );
        }

        return Ok(());
    }

    pub fn end_drawing(&mut self) -> Result<(), GPUError> {
        match self.curr_rp {
            Some(_) => {
                unsafe { (*self.ctx).device.cmd_end_render_pass(self.cmd_buf) };
                self.last_op_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
                self.last_op_access = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
                Ok(())
            }
            None => return Err(GPUError::LibraryError()),
        }
    }

    pub fn next_subpass(&mut self) -> Result<(), GPUError> {
        if self.curr_rp.is_none() {
            return Err(GPUError::LibraryError());
        }

        unsafe {
            (*self.ctx)
                .device
                .cmd_next_subpass(self.cmd_buf, vk::SubpassContents::INLINE);
        }

        self.last_op_stage = vk::PipelineStageFlags::NONE;
        self.last_op_access = vk::AccessFlags::NONE;

        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), GPUError> {
        unsafe {
            (*self.ctx)
                .device
                .reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty())?;

            (*self.ctx).device.begin_command_buffer(
                self.cmd_buf,
                &vk::CommandBufferBeginInfo::builder().build(),
            )?;
        }

        self.dirty = true;
        self.curr_rp = None;
        self.curr_pipeline = None;
        Ok(())
    }

    /// Transition image with stage/access based on last_op
    fn transition_image_layout(
        &mut self,
        view: Handle<ImageView>,
        layout: vk::ImageLayout,
        new_stage: vk::PipelineStageFlags,
        new_access: vk::AccessFlags,
    ) {
        unsafe {
            let view_data = self.ctx_ref().image_views.get_ref(view).unwrap();
            let img_data = self.ctx_ref().images.get_mut_ref(view_data.img).unwrap();
            let (src_stage, src_access, dst_stage, dst_access) = self
                .ctx_ref()
                .barrier_masks_for_transition(img_data.layout, layout);
            self.ctx_ref().device.cmd_pipeline_barrier(
                self.cmd_buf,
                src_stage,
                dst_stage,
                vk::DependencyFlags::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .old_layout(img_data.layout)
                    .new_layout(layout)
                    .src_access_mask(src_access)
                    .dst_access_mask(dst_access)
                    .image(img_data.img)
                    .subresource_range(view_data.range)
                    .build()],
            );
            img_data.layout = layout;
            self.last_op_stage = new_stage;
            self.last_op_access = new_access;
        }
    }

    /// Transition image with stage/access based on last_op
    fn transition_image(
        &mut self,
        view: Handle<ImageView>,
        new_stage: vk::PipelineStageFlags,
        new_access: vk::AccessFlags,
    ) {
        unsafe {
            let view_data = self.ctx_ref().image_views.get_ref(view).unwrap();
            let img_data = self.ctx_ref().images.get_ref(view_data.img).unwrap();
            self.ctx_ref().device.cmd_pipeline_barrier(
                self.cmd_buf,
                self.last_op_stage,
                new_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .old_layout(img_data.layout)
                    .new_layout(img_data.layout)
                    .src_access_mask(self.last_op_access)
                    .dst_access_mask(new_access)
                    .image(img_data.img)
                    .subresource_range(view_data.range)
                    .build()],
            );
            self.last_op_stage = new_stage;
            self.last_op_access = new_access;
        }
    }

    fn update_last_access(&mut self, stage: vk::PipelineStageFlags, access: vk::AccessFlags) {
        self.last_op_stage = stage;
        self.last_op_access = access;
    }

    fn ctx_ref(&self) -> &'static mut Context {
        unsafe { &mut *self.ctx }
    }

    /// Creates or retrieves a cached framebuffer and clear values
    fn create_or_cache_framebuffer(
        &mut self,
        rp: &mut RenderPass,
        info: &RenderPassBegin,
    ) -> Result<SubpassContainer, GPUError> {
        let mut hasher = DefaultHasher::new();
        for a in info.attachments {
            a.hash(&mut hasher);
        }
        let key = hasher.finish();
        if !rp.subpasses.contains_key(&key) {
            let fb = self.ctx_ref().make_framebuffer(info)?;
            let clear_vals = info
                .attachments
                .iter()
                .map(|a| a.to_vk_clear_value())
                .collect();
            rp.subpasses.insert(
                key,
                SubpassContainer {
                    attachments: info.attachments.to_vec(),
                    fb,
                    clear_values: clear_vals,
                },
            );
        }
        Ok(rp.subpasses.get(&key).unwrap().clone())
    }
}

//use std::hash::{DefaultHasher, Hash, Hasher};
//
//use crate::{
//    Attachment, AttachmentType, Context, IndexedBindGroup, IndexedIndirectCommand, IndirectCommand,
//    SubpassContainer,
//};
//
//use super::{
//    convert_barrier_point_vk, convert_rect2d_to_vulkan, BarrierPoint, Buffer, CommandList,
//    DynamicBuffer, Filter, GPUError, GraphicsPipeline, ImageView, Rect2D, Viewport,
//};
//use super::{BindGroup, Handle};
//use super::{ComputePipeline, RenderPass};
//use ash::*;
//
//#[derive(Clone)]
//pub struct BufferCopy {
//    pub src: Handle<Buffer>,
//    pub dst: Handle<Buffer>,
//    pub src_offset: usize,
//    pub dst_offset: usize,
//    pub size: usize,
//}
//
//impl Default for BufferCopy {
//    fn default() -> Self {
//        Self {
//            src: Default::default(),
//            dst: Default::default(),
//            src_offset: 0,
//            dst_offset: 0,
//            size: 0,
//        }
//    }
//}
//
//#[derive(Clone, Default)]
//pub struct ImageBufferCopy {
//    pub src: Handle<ImageView>,
//    pub dst: Handle<Buffer>,
//    pub dst_offset: usize,
//}
//
//#[derive(Clone, Default)]
//pub struct BufferImageCopy {
//    pub src: Handle<Buffer>,
//    pub dst: Handle<ImageView>,
//    pub src_offset: usize,
//}
//
//#[derive(Clone)]
//pub struct ImageBlit {
//    pub src: Handle<ImageView>,
//    pub dst: Handle<ImageView>,
//    pub filter: Filter,
//    pub src_region: Rect2D,
//    pub dst_region: Rect2D,
//}
//
//impl Default for ImageBlit {
//    fn default() -> Self {
//        Self {
//            src: Default::default(),
//            dst: Default::default(),
//            filter: Filter::Nearest,
//            src_region: Default::default(),
//            dst_region: Default::default(),
//        }
//    }
//}
//
//#[derive(Clone)]
//pub struct Dispatch {
//    pub compute: Handle<ComputePipeline>,
//    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
//    pub bind_groups: [Option<Handle<BindGroup>>; 4],
//    pub workgroup_size: [u32; 3],
//}
//
//impl Default for Dispatch {
//    fn default() -> Self {
//        Self {
//            compute: Default::default(),
//            dynamic_buffers: Default::default(),
//            bind_groups: Default::default(),
//            workgroup_size: Default::default(),
//        }
//    }
//}
//
//#[derive(Clone)]
//pub struct DrawBegin<'a> {
//    pub viewport: Viewport,
//    pub pipeline: Handle<GraphicsPipeline>,
//    pub attachments: &'a [Attachment],
//}
//
//#[derive(Clone, Default)]
//pub struct Draw {
//    pub vertices: Handle<Buffer>,
//    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
//    pub instance_count: u32,
//    pub count: u32,
//}
//
//#[derive(Clone)]
//pub struct DrawIndexedDynamic {
//    pub vertices: DynamicBuffer,
//    pub indices: DynamicBuffer,
//    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
//    pub bind_groups: [Option<Handle<BindGroup>>; 4],
//    pub index_count: u32,
//    pub instance_count: u32,
//    pub first_instance: u32,
//}
//
//impl Default for DrawIndexedDynamic {
//    fn default() -> Self {
//        Self {
//            vertices: Default::default(),
//            indices: Default::default(),
//            index_count: Default::default(),
//            instance_count: 1,
//            first_instance: 0,
//            bind_groups: [None, None, None, None],
//            dynamic_buffers: [None, None, None, None],
//        }
//    }
//}
//
//#[derive(Clone)]
//pub struct BindIndexedBG {
//    pub bind_groups: [Option<Handle<IndexedBindGroup>>; 4],
//}
//
//#[derive(Clone, Default)]
//pub struct RenderPassBegin<'a> {
//    pub render_pass: Handle<RenderPass>,
//    pub viewport: Viewport,
//    pub attachments: &'a [Attachment],
//}
//
//#[derive(Clone)]
//pub struct DrawIndexed {
//    pub vertices: Handle<Buffer>,
//    pub indices: Handle<Buffer>,
//    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
//    pub bind_groups: [Option<Handle<BindGroup>>; 4],
//    pub index_count: u32,
//    pub instance_count: u32,
//    pub first_instance: u32,
//}
//
//impl Default for DrawIndexed {
//    fn default() -> Self {
//        Self {
//            vertices: Default::default(),
//            indices: Default::default(),
//            index_count: Default::default(),
//            instance_count: 1,
//            first_instance: 0,
//            bind_groups: [None, None, None, None],
//            dynamic_buffers: [None, None, None, None],
//        }
//    }
//}
//
//#[derive(Clone)]
//pub struct DrawIndexedIndirect {
//    pub draw_params: Handle<Buffer>,
//    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
//    pub bind_groups: [Option<Handle<BindGroup>>; 4],
//    pub offset: u32,
//    pub draw_count: u32,
//    pub stride: u32,
//}
//
//impl Default for DrawIndexedIndirect {
//    fn default() -> Self {
//        Self {
//            draw_params: Default::default(),
//            bind_groups: [None, None, None, None],
//            dynamic_buffers: [None, None, None, None],
//            offset: 0,
//            draw_count: 0,
//            stride: std::mem::size_of::<IndexedIndirectCommand>() as u32,
//        }
//    }
//}
//
//#[derive(Clone)]
//pub struct DrawIndirect {
//    pub draw_params: Handle<Buffer>,
//    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
//    pub bind_groups: [Option<Handle<BindGroup>>; 4],
//    pub offset: u32,
//    pub draw_count: u32,
//    pub stride: u32,
//}
//
//impl Default for DrawIndirect {
//    fn default() -> Self {
//        Self {
//            draw_params: Default::default(),
//            bind_groups: [None, None, None, None],
//            dynamic_buffers: [None, None, None, None],
//            offset: 0,
//            draw_count: 0,
//            stride: std::mem::size_of::<IndirectCommand>() as u32,
//        }
//    }
//}
//
//#[derive(Clone, Copy)]
//pub struct BindPipeline {
//    pub gfx: Option<Handle<GraphicsPipeline>>,
//    pub compute: Option<Handle<ComputePipeline>>,
//    pub bg: Handle<BindGroup>,
//}
//
//#[derive(Clone, Copy)]
//pub struct ImageBarrier {
//    pub view: Handle<ImageView>,
//    pub src: BarrierPoint,
//    pub dst: BarrierPoint,
//}
//
//#[derive(Clone)]
//pub enum Command {
//    BufferCopyCommand(BufferCopy),
//    BufferImageCopyCommand(BufferImageCopy),
//    DrawCommand(Draw),
//    DrawIndexedCommand(DrawIndexed),
//    DrawIndexedDynamicCommand(DrawIndexedDynamic),
//    BlitCommand(ImageBlit),
//    ImageBarrierCommand(ImageBarrier),
//    DispatchCommand(Dispatch),
//}
//
//impl CommandList {
//    pub fn dispatch(&mut self, cmd: &Dispatch) {
//        unsafe {
//            let compute = (*self.ctx).compute_pipelines.get_ref(cmd.compute).unwrap();
//            // TODO probably should cache this. But I'm lazy. And this should work for now.
//            //
//
//            let mut dynamic_offsets = Vec::new();
//            for dynamic in cmd.dynamic_buffers {
//                if let Some(buf) = dynamic {
//                    dynamic_offsets.push(buf.alloc.offset);
//                }
//            }
//
//            // TODO dont do this in a loop, collect and send all at once.
//            for (i, opt) in cmd.bind_groups.iter().enumerate() {
//                if let Some(bg) = opt {
//                    let b = (*self.ctx).bind_groups.get_ref(*bg).unwrap();
//                    let pl = (*self.ctx)
//                        .compute_pipeline_layouts
//                        .get_ref(compute.layout)
//                        .unwrap();
//
//                    let dyn_buff = if let Some(d) = cmd.dynamic_buffers[i] {
//                        vec![d.alloc.offset]
//                    } else {
//                        vec![]
//                    };
//
//                    (*self.ctx).device.cmd_bind_descriptor_sets(
//                        self.cmd_buf,
//                        vk::PipelineBindPoint::COMPUTE,
//                        pl.layout,
//                        b.set_id,
//                        &[b.set],
//                        &dyn_buff,
//                    );
//                }
//            }
//            self.curr_pipeline = None;
//            (*self.ctx).device.cmd_bind_pipeline(
//                self.cmd_buf,
//                vk::PipelineBindPoint::COMPUTE,
//                compute.raw,
//            );
//
//            (*self.ctx).device.cmd_dispatch(
//                self.cmd_buf,
//                cmd.workgroup_size[0],
//                cmd.workgroup_size[1],
//                cmd.workgroup_size[2],
//            );
//
//            self.last_op_stage = vk::PipelineStageFlags::COMPUTE_SHADER;
//            self.last_op_access = vk::AccessFlags::SHADER_WRITE;
//        }
//    }
//
//    pub fn draw_dynamic_indexed(&mut self, cmd: &DrawIndexedDynamic) {
//        self.append(Command::DrawIndexedDynamicCommand(cmd.clone()));
//    }
//    pub fn draw_indexed(&mut self, cmd: &DrawIndexed) {
//        self.append(Command::DrawIndexedCommand(cmd.clone()));
//    }
//
//    pub fn draw_indexed_indirect(&mut self, cmd: &DrawIndexedIndirect) {
//        unsafe {
//            let buff = (*self.ctx).buffers.get_ref(cmd.draw_params).unwrap();
//            (*self.ctx).device.cmd_draw_indexed_indirect(
//                self.cmd_buf,
//                buff.buf,
//                cmd.offset as u64,
//                cmd.draw_count,
//                cmd.stride,
//            );
//
//            self.last_op_stage = vk::PipelineStageFlags::VERTEX_SHADER;
//            self.last_op_access = vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
//        }
//    }
//
//    pub fn draw_indirect(&mut self, cmd: &DrawIndirect) {
//        unsafe {
//            let buff = (*self.ctx).buffers.get_ref(cmd.draw_params).unwrap();
//            (*self.ctx).device.cmd_draw_indirect(
//                self.cmd_buf,
//                buff.buf,
//                cmd.offset as u64,
//                cmd.draw_count,
//                cmd.stride,
//            );
//
//            self.last_op_stage = vk::PipelineStageFlags::VERTEX_SHADER;
//            self.last_op_access = vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
//        }
//    }
//
//    pub fn blit(&mut self, cmd: ImageBlit) {
//        self.append(Command::BlitCommand(cmd));
//    }
//
//    pub fn copy_buffers(&mut self, info: &BufferCopy) {
//        self.append(Command::BufferCopyCommand(info.clone()));
//    }
//
//    pub fn copy_buffer_to_image(&mut self, rec: &BufferImageCopy) {
//        unsafe {
//            let view = (*self.ctx).image_views.get_ref(rec.dst).unwrap();
//            let img = (*self.ctx).images.get_ref(view.img).unwrap();
//            let old_layout = img.layout;
//            (*self.ctx).transition_image_stages(
//                self.cmd_buf,
//                rec.dst,
//                vk::ImageLayout::GENERAL,
//                self.last_op_stage,
//                vk::PipelineStageFlags::TRANSFER,
//                self.last_op_access,
//                vk::AccessFlags::TRANSFER_WRITE,
//            );
//
//            (*self.ctx).device.cmd_copy_buffer_to_image(
//                self.cmd_buf,
//                (*self.ctx).buffers.get_ref(rec.src).unwrap().buf,
//                img.img,
//                img.layout,
//                &[vk::BufferImageCopy {
//                    buffer_offset: rec.src_offset as u64,
//                    image_subresource: vk::ImageSubresourceLayers {
//                        aspect_mask: img.sub_layers.aspect_mask,
//                        mip_level: view.range.base_mip_level,
//                        base_array_layer: view.range.base_array_layer,
//                        layer_count: view.range.layer_count,
//                    },
//                    image_extent: img.extent,
//
//                    ..Default::default()
//                }],
//            );
//
//            (*self.ctx).transition_image_stages(
//                self.cmd_buf,
//                rec.dst,
//                old_layout,
//                vk::PipelineStageFlags::TRANSFER,
//                vk::PipelineStageFlags::TRANSFER,
//                vk::AccessFlags::TRANSFER_WRITE,
//                vk::AccessFlags::TRANSFER_READ,
//            );
//
//            self.last_op_stage = vk::PipelineStageFlags::TRANSFER;
//            self.last_op_access = vk::AccessFlags::MEMORY_WRITE;
//        };
//    }
//
//    pub fn copy_image_to_buffer(&mut self, rec: &ImageBufferCopy) {
//        unsafe {
//            let view = (*self.ctx).image_views.get_ref(rec.src).unwrap();
//            let img = (*self.ctx).images.get_ref(view.img).unwrap();
//            let old_layout = img.layout;
//            (*self.ctx).transition_image_stages(
//                self.cmd_buf,
//                rec.src,
//                vk::ImageLayout::GENERAL,
//                self.last_op_stage,
//                vk::PipelineStageFlags::TRANSFER,
//                self.last_op_access,
//                vk::AccessFlags::TRANSFER_READ,
//            );
//
//            (*self.ctx).device.cmd_copy_image_to_buffer(
//                self.cmd_buf,
//                img.img,
//                img.layout,
//                (*self.ctx).buffers.get_ref(rec.dst).unwrap().buf,
//                &[vk::BufferImageCopy {
//                    buffer_offset: rec.dst_offset as u64,
//                    image_subresource: vk::ImageSubresourceLayers {
//                        aspect_mask: img.sub_layers.aspect_mask,
//                        mip_level: view.range.base_mip_level,
//                        base_array_layer: view.range.base_array_layer,
//                        layer_count: view.range.layer_count,
//                    },
//                    image_extent: img.extent,
//
//                    ..Default::default()
//                }],
//            );
//
//            (*self.ctx).transition_image_stages(
//                self.cmd_buf,
//                rec.src,
//                old_layout,
//                vk::PipelineStageFlags::TRANSFER,
//                vk::PipelineStageFlags::TRANSFER,
//                vk::AccessFlags::TRANSFER_READ,
//                vk::AccessFlags::TRANSFER_READ,
//            );
//
//            self.last_op_stage = vk::PipelineStageFlags::TRANSFER;
//            self.last_op_access = vk::AccessFlags::MEMORY_WRITE;
//        };
//    }
//
//    pub fn image_barrier(&mut self, barrier: &ImageBarrier) {
//        self.append(Command::ImageBarrierCommand(barrier.clone()));
//    }
//
//    pub fn bind_pipeline(&mut self, info: &BindPipeline) {
//        unsafe {
//            let b = (*self.ctx).bind_groups.get_ref(info.bg).unwrap();
//
//            let mut _layout = Default::default();
//            if let Some(gfx) = info.gfx {
//                let p = (*self.ctx).gfx_pipelines.get_ref(gfx).unwrap();
//                let pl = (*self.ctx).gfx_pipeline_layouts.get_ref(p.layout).unwrap();
//                _layout = pl.layout;
//            } else if let Some(cpt) = info.compute {
//                let p = (*self.ctx).compute_pipelines.get_ref(cpt).unwrap();
//                let pl = (*self.ctx)
//                    .compute_pipeline_layouts
//                    .get_ref(p.layout)
//                    .unwrap();
//                _layout = pl.layout;
//            } else {
//                return;
//            }
//
//            (*self.ctx).device.cmd_bind_descriptor_sets(
//                self.cmd_buf,
//                vk::PipelineBindPoint::GRAPHICS,
//                _layout,
//                b.set_id,
//                &[b.set],
//                &[],
//            );
//        }
//    }
//
//    fn get_ctx(&mut self) -> &mut Context {
//        unsafe { &mut *(self.ctx) }
//    }
//
//    fn begin_render_pass<'a>(&mut self, info: &RenderPassBegin) -> Result<(), GPUError> {
//        if let Some(curr_rp) = self.curr_rp {
//            if curr_rp == info.render_pass {
//                return Ok(());
//            } else {
//                self.end_drawing()?;
//            }
//        }
//
//        self.curr_rp = Some(info.render_pass);
//        unsafe {
//            let rp = (*self.ctx)
//                .render_passes
//                .get_mut_ref(self.curr_rp.unwrap())
//                .unwrap();
//
//            let mut _fb = Default::default();
//            let mut _clear_values: &[vk::ClearValue] = &[];
//            let mut hasher = DefaultHasher::new();
//            for attach in info.attachments {
//                attach.hash(&mut hasher);
//            }
//
//            let hash = hasher.finish();
//            if let Some(entry) = rp.subpasses.get(&hash) {
//                _fb = entry.fb.clone();
//                _clear_values = &entry.clear_values;
//            } else {
//                _fb = self.get_ctx().make_framebuffer(&info)?;
//                let mut cv = Vec::new();
//                for a in info.attachments {
//                    match a.clear {
//                        crate::ClearValue::Color(c) => {
//                            cv.push(vk::ClearValue {
//                                color: vk::ClearColorValue { float32: c },
//                            });
//                        }
//                        crate::ClearValue::IntColor(c) => {
//                            cv.push(vk::ClearValue {
//                                color: vk::ClearColorValue { int32: c },
//                            });
//                        }
//                        crate::ClearValue::UintColor(c) => {
//                            cv.push(vk::ClearValue {
//                                color: vk::ClearColorValue { uint32: c },
//                            });
//                        }
//                        crate::ClearValue::DepthStencil { depth, stencil } => {
//                            cv.push(vk::ClearValue {
//                                depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
//                            });
//                        }
//                    }
//                }
//                rp.subpasses.insert(
//                    hash,
//                    SubpassContainer {
//                        attachments: info.attachments.to_vec(),
//                        fb: _fb,
//                        clear_values: cv,
//                    },
//                );
//                _clear_values = &rp.subpasses.get(&hash).unwrap().clear_values;
//            }
//
//            (*self.ctx).device.cmd_begin_render_pass(
//                self.cmd_buf,
//                &vk::RenderPassBeginInfo::builder()
//                    .render_pass(rp.raw)
//                    .framebuffer(_fb)
//                    .clear_values(&_clear_values)
//                    .render_area(convert_rect2d_to_vulkan(info.viewport.scissor))
//                    .build(),
//                vk::SubpassContents::INLINE,
//            );
//
//            self.last_op_stage = vk::PipelineStageFlags::NONE;
//            self.last_op_access = vk::AccessFlags::NONE;
//        }
//
//        Ok(())
//    }
//
//    pub fn begin_drawing(&mut self, info: &DrawBegin) -> Result<(), GPUError> {
//        let pipeline = info.pipeline;
//        if let Some(gfx) = self.curr_pipeline {
//            if pipeline == gfx {
//                return Ok(());
//            }
//        }
//
//        unsafe {
//            self.curr_pipeline = Some(pipeline);
//            let gfx = (*self.ctx).gfx_pipelines.get_ref(pipeline).unwrap();
//            self.begin_render_pass(&RenderPassBegin {
//                render_pass: gfx.render_pass,
//                viewport: info.viewport,
//                attachments: info.attachments,
//            })?;
//            (*self.ctx).device.cmd_bind_pipeline(
//                self.cmd_buf,
//                vk::PipelineBindPoint::GRAPHICS,
//                gfx.raw,
//            );
//        }
//
//        return Ok(());
//    }
//
//    pub fn end_drawing(&mut self) -> Result<(), GPUError> {
//        match self.curr_rp {
//            Some(_) => {
//                unsafe { (*self.ctx).device.cmd_end_render_pass(self.cmd_buf) };
//                self.last_op_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
//                self.last_op_access = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
//                Ok(())
//            }
//            None => return Err(GPUError::LibraryError()),
//        }
//    }
//
//    pub fn reset(&mut self) -> Result<(), GPUError> {
//        unsafe {
//            (*self.ctx)
//                .device
//                .reset_command_buffer(self.cmd_buf, vk::CommandBufferResetFlags::empty())?;
//
//            (*self.ctx).device.begin_command_buffer(
//                self.cmd_buf,
//                &vk::CommandBufferBeginInfo::builder().build(),
//            )?;
//        }
//
//        self.curr_rp = None;
//        self.curr_pipeline = None;
//        Ok(())
//    }
//
//    pub fn next_subpass(&mut self) -> Result<(), GPUError> {
//        if self.curr_rp.is_none() {
//            return Err(GPUError::LibraryError());
//        }
//
//        unsafe {
//            (*self.ctx)
//                .device
//                .cmd_next_subpass(self.cmd_buf, vk::SubpassContents::INLINE);
//        }
//
//        self.last_op_stage = vk::PipelineStageFlags::NONE;
//        self.last_op_access = vk::AccessFlags::NONE;
//
//        Ok(())
//    }
//
//    pub fn append(&mut self, cmd: Command) {
//        if self.ctx == std::ptr::null_mut() {
//            return;
//        };
//        match cmd {
//            Command::BufferCopyCommand(rec) => {
//                unsafe {
//                    let src_buf = (*self.ctx).buffers.get_ref(rec.src).unwrap();
//                    let dst_buf = (*self.ctx).buffers.get_ref(rec.dst).unwrap();
//
//                    (*self.ctx).device.cmd_copy_buffer(
//                        self.cmd_buf,
//                        src_buf.buf,
//                        dst_buf.buf,
//                        &[vk::BufferCopy {
//                            src_offset: rec.src_offset as u64,
//                            dst_offset: rec.dst_offset as u64,
//                            size: match rec.size == 0 {
//                                true => std::cmp::min(src_buf.size, dst_buf.size) as u64,
//                                false => rec.size as u64,
//                            },
//                        }],
//                    )
//                };
//
//                self.last_op_stage = vk::PipelineStageFlags::TRANSFER;
//                self.last_op_access = vk::AccessFlags::MEMORY_WRITE;
//            }
//            Command::BufferImageCopyCommand(rec) => {
//                unsafe {
//                    let view = (*self.ctx).image_views.get_ref(rec.dst).unwrap();
//                    let img = (*self.ctx).images.get_ref(view.img).unwrap();
//                    let old_layout = img.layout;
//                    (*self.ctx).transition_image_stages(
//                        self.cmd_buf,
//                        rec.dst,
//                        vk::ImageLayout::GENERAL,
//                        self.last_op_stage,
//                        vk::PipelineStageFlags::TRANSFER,
//                        self.last_op_access,
//                        vk::AccessFlags::TRANSFER_WRITE,
//                    );
//
//                    (*self.ctx).device.cmd_copy_buffer_to_image(
//                        self.cmd_buf,
//                        (*self.ctx).buffers.get_ref(rec.src).unwrap().buf,
//                        img.img,
//                        img.layout,
//                        &[vk::BufferImageCopy {
//                            buffer_offset: rec.src_offset as u64,
//                            image_subresource: vk::ImageSubresourceLayers {
//                                aspect_mask: img.sub_layers.aspect_mask,
//                                mip_level: view.range.base_mip_level,
//                                base_array_layer: view.range.base_array_layer,
//                                layer_count: view.range.layer_count,
//                            },
//                            image_extent: img.extent,
//
//                            ..Default::default()
//                        }],
//                    );
//
//                    (*self.ctx).transition_image_stages(
//                        self.cmd_buf,
//                        rec.dst,
//                        old_layout,
//                        vk::PipelineStageFlags::TRANSFER,
//                        vk::PipelineStageFlags::TRANSFER,
//                        vk::AccessFlags::TRANSFER_WRITE,
//                        vk::AccessFlags::TRANSFER_READ,
//                    );
//
//                    self.last_op_stage = vk::PipelineStageFlags::TRANSFER;
//                    self.last_op_access = vk::AccessFlags::MEMORY_WRITE;
//                };
//            }
//            Command::DrawCommand(rec) => unsafe {
//                static DEVICE_SIZES: vk::DeviceSize = vk::DeviceSize::MIN;
//                let buff = (*self.ctx).buffers.get_ref(rec.vertices).unwrap();
//                (*self.ctx).device.cmd_bind_vertex_buffers(
//                    self.cmd_buf,
//                    0,
//                    &[buff.buf],
//                    &[DEVICE_SIZES],
//                );
//                (*self.ctx)
//                    .device
//                    .cmd_draw(self.cmd_buf, rec.count, rec.instance_count, 0, 0);
//
//                self.last_op_stage = vk::PipelineStageFlags::VERTEX_SHADER;
//                self.last_op_access = vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
//            },
//            Command::DrawIndexedCommand(rec) => unsafe {
//                static DEVICE_SIZES: vk::DeviceSize = vk::DeviceSize::MIN;
//                let v = (*self.ctx).buffers.get_ref(rec.vertices).unwrap();
//                let i = (*self.ctx).buffers.get_ref(rec.indices).unwrap();
//
//                let mut dynamic_offsets = Vec::new();
//                for dynamic in rec.dynamic_buffers {
//                    if let Some(buf) = dynamic {
//                        dynamic_offsets.push(buf.alloc.offset);
//                    }
//                }
//
//                // TODO dont do this in a loop, collect and send all at once.
//                for opt in &rec.bind_groups {
//                    if let Some(bg) = opt {
//                        let b = (*self.ctx).bind_groups.get_ref(*bg).unwrap();
//                        let p = (*self.ctx)
//                            .gfx_pipelines
//                            .get_ref(self.curr_pipeline.unwrap())
//                            .unwrap();
//                        let pl = (*self.ctx).gfx_pipeline_layouts.get_ref(p.layout).unwrap();
//
//                        (*self.ctx).device.cmd_bind_descriptor_sets(
//                            self.cmd_buf,
//                            vk::PipelineBindPoint::GRAPHICS,
//                            pl.layout,
//                            0,
//                            &[b.set],
//                            &dynamic_offsets,
//                        );
//                    }
//                }
//
//                (*self.ctx).device.cmd_bind_vertex_buffers(
//                    self.cmd_buf,
//                    0,
//                    &[v.buf],
//                    &[DEVICE_SIZES],
//                );
//
//                (*self.ctx).device.cmd_bind_index_buffer(
//                    self.cmd_buf,
//                    i.buf,
//                    DEVICE_SIZES,
//                    vk::IndexType::UINT32,
//                );
//                (*self.ctx).device.cmd_draw_indexed(
//                    self.cmd_buf,
//                    rec.index_count,
//                    rec.instance_count,
//                    0,
//                    0,
//                    rec.first_instance,
//                );
//                self.last_op_stage = vk::PipelineStageFlags::VERTEX_SHADER;
//                self.last_op_access = vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
//            },
//            Command::BlitCommand(rec) => unsafe {
//                let src = (*self.ctx).image_views.get_ref(rec.src).unwrap();
//                let src_img = (*self.ctx).images.get_ref(src.img).unwrap();
//                let dst = (*self.ctx).image_views.get_ref(rec.dst).unwrap();
//                let dst_img = (*self.ctx).images.get_ref(dst.img).unwrap();
//                let src_layout = src_img.layout;
//                let dst_layout = dst_img.layout;
//
//                let src_width = if rec.src_region.w == 0 {
//                    src_img.dim[0]
//                } else {
//                    rec.src_region.w
//                };
//                let src_height = if rec.src_region.h == 0 {
//                    src_img.dim[1]
//                } else {
//                    rec.src_region.h
//                };
//                let dst_width = if rec.dst_region.w == 0 {
//                    dst_img.dim[0]
//                } else {
//                    rec.dst_region.w
//                };
//                let dst_height = if rec.dst_region.h == 0 {
//                    dst_img.dim[1]
//                } else {
//                    rec.dst_region.h
//                };
//                let regions = &[vk::ImageBlit {
//                    src_subresource: src_img.sub_layers,
//                    src_offsets: [
//                        vk::Offset3D {
//                            x: rec.src_region.x as i32,
//                            y: rec.src_region.y as i32,
//                            z: 0,
//                        },
//                        vk::Offset3D {
//                            x: src_width as i32,
//                            y: src_height as i32,
//                            z: 1,
//                        },
//                    ],
//                    dst_subresource: dst_img.sub_layers,
//                    dst_offsets: [
//                        vk::Offset3D {
//                            x: rec.dst_region.x as i32,
//                            y: rec.dst_region.y as i32,
//                            z: 0,
//                        },
//                        vk::Offset3D {
//                            x: dst_width as i32,
//                            y: dst_height as i32,
//                            z: 1,
//                        },
//                    ],
//                }];
//
//                (*self.ctx).transition_image_stages(
//                    self.cmd_buf,
//                    rec.src,
//                    vk::ImageLayout::GENERAL,
//                    self.last_op_stage,
//                    vk::PipelineStageFlags::TRANSFER,
//                    self.last_op_access,
//                    vk::AccessFlags::TRANSFER_READ,
//                );
//                (*self.ctx).transition_image_stages(
//                    self.cmd_buf,
//                    rec.dst,
//                    vk::ImageLayout::GENERAL,
//                    self.last_op_stage,
//                    vk::PipelineStageFlags::TRANSFER,
//                    self.last_op_access,
//                    vk::AccessFlags::TRANSFER_WRITE,
//                );
//                (*self.ctx).device.cmd_blit_image(
//                    self.cmd_buf,
//                    src_img.img,
//                    src_img.layout,
//                    dst_img.img,
//                    dst_img.layout,
//                    regions,
//                    rec.filter.into(),
//                );
//                (*self.ctx).transition_image_stages(
//                    self.cmd_buf,
//                    rec.src,
//                    src_layout,
//                    vk::PipelineStageFlags::TRANSFER,
//                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
//                    vk::AccessFlags::TRANSFER_READ,
//                    vk::AccessFlags::NONE,
//                );
//                (*self.ctx).transition_image_stages(
//                    self.cmd_buf,
//                    rec.dst,
//                    dst_layout,
//                    vk::PipelineStageFlags::TRANSFER,
//                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
//                    vk::AccessFlags::TRANSFER_WRITE,
//                    vk::AccessFlags::NONE,
//                );
//                self.last_op_stage = vk::PipelineStageFlags::TRANSFER;
//                self.last_op_access = vk::AccessFlags::TRANSFER_WRITE;
//            },
//            Command::ImageBarrierCommand(rec) => unsafe {
//                let view = (*self.ctx).image_views.get_ref(rec.view).unwrap();
//                let img = (*self.ctx).images.get_ref(view.img).unwrap();
//
//                let dst_flag = convert_barrier_point_vk(rec.dst);
//                (*self.ctx).device.cmd_pipeline_barrier(
//                    self.cmd_buf,
//                    self.last_op_stage,
//                    dst_flag,
//                    vk::DependencyFlags::default(),
//                    &[],
//                    &[],
//                    &[vk::ImageMemoryBarrier::builder()
//                        .new_layout(img.layout)
//                        .old_layout(img.layout)
//                        .src_access_mask(self.last_op_access)
//                        .image(img.img)
//                        .subresource_range(view.range)
//                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
//                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
//                        .build()],
//                );
//            },
//            Command::DrawIndexedDynamicCommand(rec) => unsafe {
//                let v = (*self.ctx).buffers.get_ref(rec.vertices.handle()).unwrap();
//                let i = (*self.ctx).buffers.get_ref(rec.indices.handle()).unwrap();
//
//                let mut dynamic_offsets = Vec::new();
//                for dynamic in rec.dynamic_buffers {
//                    if let Some(buf) = dynamic {
//                        dynamic_offsets.push(buf.alloc.offset);
//                    }
//                }
//
//                // TODO dont do this in a loop, collect and send all at once.
//                for opt in &rec.bind_groups {
//                    if let Some(bg) = opt {
//                        let b = (*self.ctx).bind_groups.get_ref(*bg).unwrap();
//                        let p = (*self.ctx)
//                            .gfx_pipelines
//                            .get_ref(self.curr_pipeline.unwrap())
//                            .unwrap();
//                        let pl = (*self.ctx).gfx_pipeline_layouts.get_ref(p.layout).unwrap();
//
//                        (*self.ctx).device.cmd_bind_descriptor_sets(
//                            self.cmd_buf,
//                            vk::PipelineBindPoint::GRAPHICS,
//                            pl.layout,
//                            0,
//                            &[b.set],
//                            &dynamic_offsets,
//                        );
//                    }
//                }
//
//                (*self.ctx).device.cmd_bind_vertex_buffers(
//                    self.cmd_buf,
//                    0,
//                    &[v.buf],
//                    &[rec.vertices.offset() as u64],
//                );
//
//                (*self.ctx).device.cmd_bind_index_buffer(
//                    self.cmd_buf,
//                    i.buf,
//                    rec.indices.offset() as u64,
//                    vk::IndexType::UINT32,
//                );
//
//                (*self.ctx).device.cmd_draw_indexed(
//                    self.cmd_buf,
//                    rec.index_count,
//                    rec.instance_count,
//                    0,
//                    0,
//                    rec.first_instance,
//                );
//                self.last_op_stage = vk::PipelineStageFlags::VERTEX_SHADER;
//                self.last_op_access = vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
//            },
//            Command::DispatchCommand(_) => todo!(),
//        }
//        self.dirty = true;
//    }
//}
