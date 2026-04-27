mod common;

use common::ValidationContext;
use dashi::driver::command::{CopyImageBuffer, Dispatch};
use dashi::gpu::cmd::CommandStream;
use dashi::gpu::vulkan::*;
use serial_test::serial;

#[test]
#[serial]
fn volume_texture_compute_round_trip() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();

    let voxels: Vec<u8> = vec![
        255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 0, 255, 255, 0, 255, 255,
        0, 255, 255, 255, 255, 255, 255, 255, 32, 64, 128, 255,
    ];

    let src = ctx
        .make_image(&ImageInfo {
            debug_name: "volume_src",
            dim: [2, 2, 2],
            format: Format::RGBA8Unorm,
            initial_data: Some(&voxels),
            ..Default::default()
        })
        .unwrap();

    let dst = ctx
        .make_image(&ImageInfo {
            debug_name: "volume_dst",
            dim: [2, 2, 2],
            format: Format::RGBA8Unorm,
            storage: true,
            ..Default::default()
        })
        .unwrap();

    let readback = ctx
        .make_buffer(&BufferInfo {
            debug_name: "volume_readback",
            byte_size: voxels.len() as u32,
            visibility: MemoryVisibility::CpuAndGpu,
            ..Default::default()
        })
        .unwrap();

    let sampler = ctx
        .make_sampler(&SamplerInfo {
            min_filter: Filter::Nearest,
            mag_filter: Filter::Nearest,
            address_mode_u: SamplerAddressMode::ClampToEdge,
            address_mode_v: SamplerAddressMode::ClampToEdge,
            address_mode_w: SamplerAddressMode::ClampToEdge,
            ..Default::default()
        })
        .unwrap();

    let table_layout = ctx
        .make_bind_table_layout(&BindTableLayoutInfo {
            debug_name: "volume_bind_table_layout",
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Compute,
                variables: &[
                    BindTableVariable {
                        var_type: BindTableVariableType::SampledImage,
                        binding: 0,
                        count: 1,
                    },
                    BindTableVariable {
                        var_type: BindTableVariableType::StorageImage,
                        binding: 1,
                        count: 1,
                    },
                ],
            }],
        })
        .unwrap();

    let bind_table = ctx
        .make_bind_table(&BindTableInfo {
            debug_name: "volume_bind_table",
            layout: table_layout,
            bindings: &[
                IndexedBindingInfo {
                    binding: 0,
                    resources: &[IndexedResource {
                        slot: 0,
                        resource: ShaderResource::SampledImage(
                            ImageView {
                                img: src,
                                view_type: ImageViewType::Type3D,
                                ..Default::default()
                            },
                            sampler,
                        ),
                    }],
                },
                IndexedBindingInfo {
                    binding: 1,
                    resources: &[IndexedResource {
                        slot: 0,
                        resource: ShaderResource::Image(ImageView {
                            img: dst,
                            view_type: ImageViewType::Type3D,
                            ..Default::default()
                        }),
                    }],
                },
            ],
            set: 0,
        })
        .unwrap();

    let pipeline_layout = ctx
        .make_compute_pipeline_layout(&ComputePipelineLayoutInfo {
            bt_layouts: [Some(table_layout), None, None, None],
            shader: &PipelineShaderInfo {
                stage: ShaderType::Compute,
                spirv: inline_spirv::inline_spirv!(
                    r#"
#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) uniform sampler3D src_tex;
layout(rgba8, set = 0, binding = 1) uniform writeonly image3D dst_img;

void main() {
    ivec3 coord = ivec3(gl_GlobalInvocationID.xyz);
    vec3 dims = vec3(imageSize(dst_img));
    vec3 uvw = (vec3(coord) + vec3(0.5)) / dims;
    vec4 sampled = texture(src_tex, uvw);
    imageStore(dst_img, coord, sampled);
}
"#,
                    comp
                ),
                entry_point: "main",
                specialization: &[],
            },
        })
        .unwrap();

    let pipeline = ctx
        .make_compute_pipeline(&ComputePipelineInfo {
            debug_name: "volume_compute_pipeline",
            layout: pipeline_layout,
        })
        .unwrap();

    let mut list = ctx
        .begin_command_queue(QueueType::Graphics, "volume_compute", false)
        .unwrap();

    let stream = CommandStream::new()
        .begin()
        .dispatch(&Dispatch {
            x: 2,
            y: 2,
            z: 2,
            pipeline,
            bind_tables: [Some(bind_table), None, None, None],
            dynamic_buffers: Default::default(),
        })
        .copy_image_to_buffer(&CopyImageBuffer {
            src: dst,
            dst: readback,
            range: Default::default(),
            dst_offset: 0,
        })
        .end();
    stream.append(&mut list).unwrap();

    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
    ctx.wait(fence).unwrap();
    ctx.destroy_cmd_queue(list);

    let result = ctx.map_buffer::<u8>(BufferView::new(readback)).unwrap().to_vec();
    ctx.unmap_buffer(readback).unwrap();

    assert_eq!(result, voxels);

    ctx.destroy_bind_table(bind_table);
    ctx.destroy_bind_table_layout(table_layout);
    ctx.destroy_buffer(readback);
    ctx.destroy_image(dst);
    ctx.destroy_image(src);
}

#[test]
#[serial]
fn volume_texture_generates_3d_mips() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();

    let base_level = vec![255u8, 0, 0, 255].repeat((4 * 4 * 4) as usize);
    let image = ctx
        .make_image(&ImageInfo {
            debug_name: "volume_mips",
            dim: [4, 4, 4],
            format: Format::RGBA8Unorm,
            mip_levels: 3,
            initial_data: Some(&base_level),
            ..Default::default()
        })
        .unwrap();

    let readback = ctx
        .make_buffer(&BufferInfo {
            debug_name: "volume_mip_readback",
            byte_size: 4,
            visibility: MemoryVisibility::CpuAndGpu,
            ..Default::default()
        })
        .unwrap();

    let mut list = ctx
        .begin_command_queue(QueueType::Graphics, "volume_mip_copy", false)
        .unwrap();
    let stream = CommandStream::new()
        .begin()
        .copy_image_to_buffer(&CopyImageBuffer {
            src: image,
            dst: readback,
            range: SubresourceRange {
                base_mip: 2,
                ..Default::default()
            },
            dst_offset: 0,
        })
        .end();
    stream.append(&mut list).unwrap();

    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
    ctx.wait(fence).unwrap();
    ctx.destroy_cmd_queue(list);

    let result = ctx.map_buffer::<u8>(BufferView::new(readback)).unwrap().to_vec();
    ctx.unmap_buffer(readback).unwrap();

    assert_eq!(result, vec![255, 0, 0, 255]);

    ctx.destroy_buffer(readback);
    ctx.destroy_image(image);
}
