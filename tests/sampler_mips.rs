mod common;

use common::ValidationContext;
use dashi::driver::command::{CopyImageBuffer, Dispatch};
use dashi::gpu::cmd::CommandStream;
use dashi::gpu::vulkan::*;
use serial_test::serial;

#[test]
#[serial]
fn normalized_sampler_reads_requested_nonzero_mip() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();

    let voxels: Vec<u8> = vec![255,0,0,255, 0,255,0,255, 0,0,255,255, 255,255,255,255];

    let src = ctx
        .make_image(&ImageInfo {
            debug_name: "volume_src",
            dim: [2, 2, 1],
            mip_levels: 2,
            format: Format::RGBA8Unorm,
            initial_data: Some(&voxels),
            ..Default::default()
        })
        .unwrap();

    let dst = ctx
        .make_image(&ImageInfo {
            debug_name: "volume_dst",
            dim: [1, 1, 1],
            format: Format::RGBA8Unorm,
            storage: true,
            ..Default::default()
        })
        .unwrap();

    let readback = ctx
        .make_buffer(&BufferInfo {
            debug_name: "volume_readback",
            byte_size: 4,
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
                                range: SubresourceRange::new(0,2,0,1),
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
                            view_type: ImageViewType::Type2D,
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
layout(set = 0, binding = 0) uniform sampler2D src_tex;
layout(rgba8, set = 0, binding = 1) uniform writeonly image2D dst_img;

void main() {
    imageStore(dst_img, ivec2(0), textureLod(src_tex,vec2(0.1),1.0));
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
            x: 1,
            y: 1,
            z: 1,
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

    let result = ctx
        .map_buffer::<u8>(BufferView::new(readback))
        .unwrap()
        .to_vec();
    ctx.unmap_buffer(readback).unwrap();

    assert_eq!(result.len(), 4);
    for channel in &result[..3] {
        assert!((*channel as i32 - 128).abs() <= 1, "requested mip must contain the average color, got {result:?}");
    }
    assert_eq!(result[3],255);

    ctx.destroy_compute_pipeline(pipeline);
    ctx.destroy_compute_pipeline_layout(pipeline_layout);
    ctx.destroy_bind_table(bind_table);
    ctx.destroy_bind_table_layout(table_layout);
    ctx.destroy_buffer(readback);
    ctx.destroy_image(dst);
    ctx.destroy_image(src);
}
