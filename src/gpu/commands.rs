use std::hash::{DefaultHasher, Hash, Hasher};

use crate::{
    Context, IndexedBindGroup, IndexedIndirectCommand, IndirectCommand, Subpass, SubpassContainer,
};

use super::{
    convert_barrier_point_vk, convert_rect2d_to_vulkan, BarrierPoint, Buffer, CommandList,
    DynamicBuffer, Filter, GPUError, GraphicsPipeline, ImageView, Rect2D, Viewport,
};
use super::{BindGroup, Handle};
use super::{ComputePipeline, RenderPass};
use ash::*;

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
#[derive(Clone)]
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
    pub subpass: Subpass<'a>,
}

#[derive(Clone)]
pub struct Draw {
    pub vertices: Handle<Buffer>,
    pub dynamic_buffers: [Option<DynamicBuffer>; 4],
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
    pub subpass: Subpass<'a>,
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
    BufferCopyCommand(BufferCopy),
    BufferImageCopyCommand(BufferImageCopy),
    DrawCommand(Draw),
    DrawIndexedCommand(DrawIndexed),
    DrawIndexedDynamicCommand(DrawIndexedDynamic),
    BlitCommand(ImageBlit),
    ImageBarrierCommand(ImageBarrier),
    DispatchCommand(Dispatch),
}

impl CommandList {
    pub fn dispatch(&mut self, cmd: &Dispatch) {
        unsafe {
            let compute = (*self.ctx).compute_pipelines.get_ref(cmd.compute).unwrap();
            // TODO probably should cache this. But I'm lazy. And this should work for now.
            //

            let mut dynamic_offsets = Vec::new();
            for dynamic in cmd.dynamic_buffers {
                if let Some(buf) = dynamic {
                    dynamic_offsets.push(buf.alloc.offset);
                }
            }

            // TODO dont do this in a loop, collect and send all at once.
            for (i, opt) in cmd.bind_groups.iter().enumerate() {
                if let Some(bg) = opt {
                    let b = (*self.ctx).bind_groups.get_ref(*bg).unwrap();
                    let pl = (*self.ctx)
                        .compute_pipeline_layouts
                        .get_ref(compute.layout)
                        .unwrap();

                    let dyn_buff = if let Some(d) = cmd.dynamic_buffers[i] {
                        vec![d.alloc.offset]
                    } else {
                        vec![]
                    };

                    (*self.ctx).device.cmd_bind_descriptor_sets(
                        self.cmd_buf,
                        vk::PipelineBindPoint::COMPUTE,
                        pl.layout,
                        b.set_id,
                        &[b.set],
                        &dyn_buff,
                    );
                }
            }
            self.curr_pipeline = None;
            (*self.ctx).device.cmd_bind_pipeline(
                self.cmd_buf,
                vk::PipelineBindPoint::COMPUTE,
                compute.raw,
            );

            (*self.ctx).device.cmd_dispatch(
                self.cmd_buf,
                cmd.workgroup_size[0],
                cmd.workgroup_size[1],
                cmd.workgroup_size[2],
            );

            self.last_op_stage = vk::PipelineStageFlags::COMPUTE_SHADER;
            self.last_op_access = vk::AccessFlags::SHADER_WRITE;
        }
    }

    pub fn draw_dynamic_indexed(&mut self, cmd: &DrawIndexedDynamic) {
        self.append(Command::DrawIndexedDynamicCommand(cmd.clone()));
    }
    pub fn draw_indexed(&mut self, cmd: &DrawIndexed) {
        self.append(Command::DrawIndexedCommand(cmd.clone()));
    }

    pub fn draw_indexed_indirect(&mut self, cmd: &DrawIndexedIndirect) {
        unsafe {
            static DEVICE_SIZES: vk::DeviceSize = vk::DeviceSize::MIN;
            let buff = (*self.ctx).buffers.get_ref(cmd.draw_params).unwrap();
            (*self.ctx).device.cmd_draw_indexed_indirect(
                self.cmd_buf,
                buff.buf,
                cmd.offset as u64,
                cmd.draw_count,
                cmd.stride,
            );

            self.last_op_stage = vk::PipelineStageFlags::VERTEX_SHADER;
            self.last_op_access = vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
        }
    }

    pub fn draw_indirect(&mut self, cmd: &DrawIndirect) {
        unsafe {
            static DEVICE_SIZES: vk::DeviceSize = vk::DeviceSize::MIN;
            let buff = (*self.ctx).buffers.get_ref(cmd.draw_params).unwrap();
            (*self.ctx).device.cmd_draw_indirect(
                self.cmd_buf,
                buff.buf,
                cmd.offset as u64,
                cmd.draw_count,
                cmd.stride,
            );

            self.last_op_stage = vk::PipelineStageFlags::VERTEX_SHADER;
            self.last_op_access = vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
        }
    }

    pub fn blit(&mut self, cmd: ImageBlit) {
        self.append(Command::BlitCommand(cmd));
    }

    pub fn copy_buffers(&mut self, info: &BufferCopy) {
        self.append(Command::BufferCopyCommand(info.clone()));
    }

    pub fn image_barrier(&mut self, barrier: &ImageBarrier) {
        self.append(Command::ImageBarrierCommand(barrier.clone()));
    }

    pub fn bind_pipeline(&mut self, info: &BindPipeline) {
        unsafe {
            let b = (*self.ctx).bind_groups.get_ref(info.bg).unwrap();

            let mut layout = Default::default();
            if let Some(gfx) = info.gfx {
                let p = (*self.ctx).gfx_pipelines.get_ref(gfx).unwrap();
                let pl = (*self.ctx).gfx_pipeline_layouts.get_ref(p.layout).unwrap();
                layout = pl.layout;
            } else if let Some(cpt) = info.compute {
                let p = (*self.ctx).compute_pipelines.get_ref(cpt).unwrap();
                let pl = (*self.ctx)
                    .compute_pipeline_layouts
                    .get_ref(p.layout)
                    .unwrap();
                layout = pl.layout;
            } else {
                return;
            }

            (*self.ctx).device.cmd_bind_descriptor_sets(
                self.cmd_buf,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                b.set_id,
                &[b.set],
                &[],
            );
        }
    }

    fn get_ctx(&mut self) -> &mut Context {
        unsafe { &mut *(self.ctx) }
    }

    fn begin_render_pass<'a>(&mut self, info: &RenderPassBegin) -> Result<(), GPUError> {
        if let Some(curr_rp) = self.curr_rp {
            if curr_rp == info.render_pass {
                return Ok(());
            } else {
                self.end_drawing()?;
            }
        }

        self.curr_rp = Some(info.render_pass);
        unsafe {
            let rp = (*self.ctx)
                .render_passes
                .get_mut_ref(self.curr_rp.unwrap())
                .unwrap();

            let mut fb = Default::default();
            let mut clear_values: &[vk::ClearValue] = &[];
            let mut hasher = DefaultHasher::new();
            info.subpass.hash(&mut hasher);
            let hash = hasher.finish();
            if let Some(entry) = rp.subpasses.get(&hash) {
                fb = entry.fb.clone();
                clear_values = &entry.clear_values;
            } else {
                fb = self.get_ctx().make_framebuffer(&info)?;
                let mut cv = Vec::new();
                for color in info.subpass.colors {
                    cv.push(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: color.clear_color,
                        },
                    });
                }
                if let Some(c) = &info.subpass.depth {
                    cv.push(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: c.clear_color[0],
                            stencil: c.clear_color[1] as u32,
                        },
                    });
                }

                rp.subpasses.insert(
                    hash,
                    SubpassContainer {
                        colors: info.subpass.colors.to_vec(),
                        depth: info.subpass.depth.clone(),
                        fb,
                        clear_values: cv,
                    },
                );
                clear_values = &rp.subpasses.get(&hash).unwrap().clear_values;
            }

            (*self.ctx).device.cmd_begin_render_pass(
                self.cmd_buf,
                &vk::RenderPassBeginInfo::builder()
                    .render_pass(rp.raw)
                    .framebuffer(fb)
                    .clear_values(&clear_values)
                    .render_area(convert_rect2d_to_vulkan(info.viewport.scissor))
                    .build(),
                vk::SubpassContents::INLINE,
            );

            self.last_op_stage = vk::PipelineStageFlags::NONE;
            self.last_op_access = vk::AccessFlags::NONE;
        }

        Ok(())
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
                subpass: info.subpass.clone(),
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

        self.curr_rp = None;
        self.curr_pipeline = None;
        Ok(())
    }

    pub fn append(&mut self, cmd: Command) {
        if self.ctx == std::ptr::null_mut() {
            return;
        };
        match cmd {
            Command::BufferCopyCommand(rec) => {
                unsafe {
                    let src_buf = (*self.ctx).buffers.get_ref(rec.src).unwrap();
                    let dst_buf = (*self.ctx).buffers.get_ref(rec.dst).unwrap();

                    (*self.ctx).device.cmd_copy_buffer(
                        self.cmd_buf,
                        src_buf.buf,
                        dst_buf.buf,
                        &[vk::BufferCopy {
                            src_offset: rec.src_offset as u64,
                            dst_offset: rec.dst_offset as u64,
                            size: match rec.size == 0 {
                                true => std::cmp::min(src_buf.size, dst_buf.size) as u64,
                                false => rec.size as u64,
                            },
                        }],
                    )
                };

                self.last_op_stage = vk::PipelineStageFlags::TRANSFER;
                self.last_op_access = vk::AccessFlags::MEMORY_WRITE;
            }
            Command::BufferImageCopyCommand(rec) => {
                unsafe {
                    let view = (*self.ctx).image_views.get_ref(rec.dst).unwrap();
                    let img = (*self.ctx).images.get_ref(view.img).unwrap();
                    let old_layout = img.layout;
                    (*self.ctx).transition_image_stages(
                        self.cmd_buf,
                        rec.dst,
                        vk::ImageLayout::GENERAL,
                        self.last_op_stage,
                        vk::PipelineStageFlags::TRANSFER,
                        self.last_op_access,
                        vk::AccessFlags::TRANSFER_WRITE,
                    );

                    (*self.ctx).device.cmd_copy_buffer_to_image(
                        self.cmd_buf,
                        (*self.ctx).buffers.get_ref(rec.src).unwrap().buf,
                        img.img,
                        img.layout,
                        &[vk::BufferImageCopy {
                            buffer_offset: rec.src_offset as u64,
                            image_subresource: vk::ImageSubresourceLayers {
                                aspect_mask: img.sub_layers.aspect_mask,
                                mip_level: view.range.base_mip_level,
                                base_array_layer: view.range.base_array_layer,
                                layer_count: view.range.layer_count,
                            },
                            image_extent: img.extent,

                            ..Default::default()
                        }],
                    );

                    (*self.ctx).transition_image_stages(
                        self.cmd_buf,
                        rec.dst,
                        old_layout,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    );

                    self.last_op_stage = vk::PipelineStageFlags::TRANSFER;
                    self.last_op_access = vk::AccessFlags::MEMORY_WRITE;
                };
            }
            Command::DrawCommand(rec) => unsafe {
                static DEVICE_SIZES: vk::DeviceSize = vk::DeviceSize::MIN;
                let buff = (*self.ctx).buffers.get_ref(rec.vertices).unwrap();
                (*self.ctx).device.cmd_bind_vertex_buffers(
                    self.cmd_buf,
                    0,
                    &[buff.buf],
                    &[DEVICE_SIZES],
                );
                (*self.ctx)
                    .device
                    .cmd_draw(self.cmd_buf, rec.count, rec.instance_count, 0, 0);

                self.last_op_stage = vk::PipelineStageFlags::VERTEX_SHADER;
                self.last_op_access = vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
            },
            Command::DrawIndexedCommand(rec) => unsafe {
                static DEVICE_SIZES: vk::DeviceSize = vk::DeviceSize::MIN;
                let v = (*self.ctx).buffers.get_ref(rec.vertices).unwrap();
                let i = (*self.ctx).buffers.get_ref(rec.indices).unwrap();

                let mut dynamic_offsets = Vec::new();
                for dynamic in rec.dynamic_buffers {
                    if let Some(buf) = dynamic {
                        dynamic_offsets.push(buf.alloc.offset);
                    }
                }

                // TODO dont do this in a loop, collect and send all at once.
                for opt in &rec.bind_groups {
                    if let Some(bg) = opt {
                        let b = (*self.ctx).bind_groups.get_ref(*bg).unwrap();
                        let p = (*self.ctx)
                            .gfx_pipelines
                            .get_ref(self.curr_pipeline.unwrap())
                            .unwrap();
                        let pl = (*self.ctx).gfx_pipeline_layouts.get_ref(p.layout).unwrap();

                        (*self.ctx).device.cmd_bind_descriptor_sets(
                            self.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            pl.layout,
                            0,
                            &[b.set],
                            &dynamic_offsets,
                        );
                    }
                }

                (*self.ctx).device.cmd_bind_vertex_buffers(
                    self.cmd_buf,
                    0,
                    &[v.buf],
                    &[DEVICE_SIZES],
                );

                (*self.ctx).device.cmd_bind_index_buffer(
                    self.cmd_buf,
                    i.buf,
                    DEVICE_SIZES,
                    vk::IndexType::UINT32,
                );
                (*self.ctx).device.cmd_draw_indexed(
                    self.cmd_buf,
                    rec.index_count,
                    rec.instance_count,
                    0,
                    0,
                    rec.first_instance,
                );
                self.last_op_stage = vk::PipelineStageFlags::VERTEX_SHADER;
                self.last_op_access = vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
            },
            Command::BlitCommand(rec) => unsafe {
                let src = (*self.ctx).image_views.get_ref(rec.src).unwrap();
                let src_img = (*self.ctx).images.get_ref(src.img).unwrap();
                let dst = (*self.ctx).image_views.get_ref(rec.dst).unwrap();
                let dst_img = (*self.ctx).images.get_ref(dst.img).unwrap();
                let src_layout = src_img.layout;
                let dst_layout = dst_img.layout;

                let src_width = if rec.src_region.w == 0 {
                    src_img.dim[0]
                } else {
                    rec.src_region.w
                };
                let src_height = if rec.src_region.h == 0 {
                    src_img.dim[1]
                } else {
                    rec.src_region.h
                };
                let dst_width = if rec.dst_region.w == 0 {
                    dst_img.dim[0]
                } else {
                    rec.dst_region.w
                };
                let dst_height = if rec.dst_region.h == 0 {
                    dst_img.dim[1]
                } else {
                    rec.dst_region.h
                };
                let regions = &[vk::ImageBlit {
                    src_subresource: src_img.sub_layers,
                    src_offsets: [
                        vk::Offset3D {
                            x: rec.src_region.x as i32,
                            y: rec.src_region.y as i32,
                            z: 0,
                        },
                        vk::Offset3D {
                            x: src_width as i32,
                            y: src_height as i32,
                            z: 1,
                        },
                    ],
                    dst_subresource: dst_img.sub_layers,
                    dst_offsets: [
                        vk::Offset3D {
                            x: rec.dst_region.x as i32,
                            y: rec.dst_region.y as i32,
                            z: 0,
                        },
                        vk::Offset3D {
                            x: dst_width as i32,
                            y: dst_height as i32,
                            z: 1,
                        },
                    ],
                }];

                (*self.ctx).transition_image_stages(
                    self.cmd_buf,
                    rec.src,
                    vk::ImageLayout::GENERAL,
                    self.last_op_stage,
                    vk::PipelineStageFlags::TRANSFER,
                    self.last_op_access,
                    vk::AccessFlags::TRANSFER_READ,
                );
                (*self.ctx).transition_image_stages(
                    self.cmd_buf,
                    rec.dst,
                    vk::ImageLayout::GENERAL,
                    self.last_op_stage,
                    vk::PipelineStageFlags::TRANSFER,
                    self.last_op_access,
                    vk::AccessFlags::TRANSFER_WRITE,
                );
                (*self.ctx).device.cmd_blit_image(
                    self.cmd_buf,
                    src_img.img,
                    src_img.layout,
                    dst_img.img,
                    dst_img.layout,
                    regions,
                    rec.filter.into(),
                );
                (*self.ctx).transition_image_stages(
                    self.cmd_buf,
                    rec.src,
                    src_layout,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::AccessFlags::NONE,
                );
                (*self.ctx).transition_image_stages(
                    self.cmd_buf,
                    rec.dst,
                    dst_layout,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::NONE,
                );
                self.last_op_stage = vk::PipelineStageFlags::TRANSFER;
                self.last_op_access = vk::AccessFlags::TRANSFER_WRITE;
            },
            Command::ImageBarrierCommand(rec) => unsafe {
                let view = (*self.ctx).image_views.get_ref(rec.view).unwrap();
                let img = (*self.ctx).images.get_ref(view.img).unwrap();

                let dst_flag = convert_barrier_point_vk(rec.dst);
                (*self.ctx).device.cmd_pipeline_barrier(
                    self.cmd_buf,
                    self.last_op_stage,
                    dst_flag,
                    vk::DependencyFlags::default(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier::builder()
                        .new_layout(img.layout)
                        .old_layout(img.layout)
                        .src_access_mask(self.last_op_access)
                        .image(img.img)
                        .subresource_range(view.range)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .build()],
                );
            },
            Command::DrawIndexedDynamicCommand(rec) => unsafe {
                let v = (*self.ctx).buffers.get_ref(rec.vertices.handle()).unwrap();
                let i = (*self.ctx).buffers.get_ref(rec.indices.handle()).unwrap();

                let mut dynamic_offsets = Vec::new();
                for dynamic in rec.dynamic_buffers {
                    if let Some(buf) = dynamic {
                        dynamic_offsets.push(buf.alloc.offset);
                    }
                }

                // TODO dont do this in a loop, collect and send all at once.
                for opt in &rec.bind_groups {
                    if let Some(bg) = opt {
                        let b = (*self.ctx).bind_groups.get_ref(*bg).unwrap();
                        let p = (*self.ctx)
                            .gfx_pipelines
                            .get_ref(self.curr_pipeline.unwrap())
                            .unwrap();
                        let pl = (*self.ctx).gfx_pipeline_layouts.get_ref(p.layout).unwrap();

                        (*self.ctx).device.cmd_bind_descriptor_sets(
                            self.cmd_buf,
                            vk::PipelineBindPoint::GRAPHICS,
                            pl.layout,
                            0,
                            &[b.set],
                            &dynamic_offsets,
                        );
                    }
                }

                (*self.ctx).device.cmd_bind_vertex_buffers(
                    self.cmd_buf,
                    0,
                    &[v.buf],
                    &[rec.vertices.offset() as u64],
                );

                (*self.ctx).device.cmd_bind_index_buffer(
                    self.cmd_buf,
                    i.buf,
                    rec.indices.offset() as u64,
                    vk::IndexType::UINT32,
                );

                (*self.ctx).device.cmd_draw_indexed(
                    self.cmd_buf,
                    rec.index_count,
                    rec.instance_count,
                    0,
                    0,
                    rec.first_instance,
                );
                self.last_op_stage = vk::PipelineStageFlags::VERTEX_SHADER;
                self.last_op_access = vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
            },
            Command::DispatchCommand(_) => todo!(),
        }
        self.dirty = true;
    }
}
