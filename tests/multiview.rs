#![cfg(feature = "vulkan")]
mod common;

use common::ValidationContext;
use dashi::driver::command::{BeginRenderPass, CopyImageBuffer, Draw};
use dashi::*;

#[test]
#[serial_test::serial]
fn multiview_writes_distinct_array_layers() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();
    assert!(
        ctx.multiview_max_views() >= 2,
        "test requires stereo multiview"
    );
    let format = Format::RGBA32Uint;
    let image = ctx
        .make_image(&ImageInfo {
            debug_name: "integer attachment",
            dim: [4, 2, 1],
            layers: 2,
            format,
            ..Default::default()
        })
        .unwrap();
    let viewport = Viewport {
        area: FRect2D {
            w: 4.0,
            h: 2.0,
            ..Default::default()
        },
        scissor: Rect2D {
            w: 4,
            h: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let pass = ctx
        .make_render_pass_multiview(
            &RenderPassInfo {
                debug_name: "integer output",
                viewport,
                subpasses: &[SubpassDescription {
                    color_attachments: &[AttachmentDescription {
                        format,
                        ..Default::default()
                    }],
                    depth_stencil_attachment: None,
                    subpass_dependencies: &[],
                }],
            },
            3,
        )
        .unwrap();
    let vertex = inline_spirv::inline_spirv!(
        r"#version 450
        #extension GL_EXT_multiview : require
        layout(location=0) flat out uint eye;
        vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));
        void main(){eye=gl_ViewIndex;gl_Position=vec4(p[gl_VertexIndex],0,1);}",
        vert
    );
    let fragment = inline_spirv::inline_spirv!(
        r"#version 450
        layout(location=0) flat in uint eye;
        layout(location=0) out uvec4 value;
        void main(){if(gl_FragCoord.x<2.0) discard;
        value=uvec4(eye+1u,0xffffffffu,0x7fc01234u,0x80000000u);}",
        frag
    );
    let layout = ctx
        .make_graphics_pipeline_layout(&GraphicsPipelineLayoutInfo {
            debug_name: "integer output",
            vertex_info: VertexDescriptionInfo {
                entries: &[],
                stride: 0,
                rate: VertexRate::Vertex,
            },
            bt_layouts: [None; 4],
            shaders: &[
                PipelineShaderInfo {
                    stage: ShaderType::Vertex,
                    spirv: vertex,
                    entry_point: "main",
                    specialization: &[],
                },
                PipelineShaderInfo {
                    stage: ShaderType::Fragment,
                    spirv: fragment,
                    entry_point: "main",
                    specialization: &[],
                },
            ],
            details: GraphicsPipelineDetails {
                color_blend_states: vec![ColorBlendState {
                    enable: false,
                    ..Default::default()
                }],
                ..Default::default()
            },
        })
        .unwrap();
    let subpass = ctx.render_pass_subpass_info(pass, 0).unwrap();
    let pipeline = ctx
        .make_graphics_pipeline_multiview(
            &GraphicsPipelineInfo {
                debug_name: "integer output",
                layout,
                attachment_formats: subpass.color_formats,
                depth_format: subpass.depth_format,
                subpass_samples: subpass.samples,
                ..Default::default()
            },
            3,
        )
        .unwrap();
    let vertices = ctx
        .make_buffer(&BufferInfo {
            byte_size: 4,
            usage: BufferUsage::VERTEX,
            initial_data: Some(&[0; 4]),
            ..Default::default()
        })
        .unwrap();
    let readback = ctx
        .make_buffer(&BufferInfo {
            byte_size: 2 * 4 * 2 * 16,
            visibility: MemoryVisibility::CpuAndGpu,
            ..Default::default()
        })
        .unwrap();
    let clear = [0xdeadbeef, 0x01000003, 0, u32::MAX];
    let mut colors = [None; 8];
    colors[0] = Some(ImageView {
        img: image,
        range: SubresourceRange::new(0, 1, 0, 2),
        view_type: ImageViewType::Type2DArray,
        ..Default::default()
    });
    let mut clears = [None; 8];
    clears[0] = Some(ClearValue::UintColor(clear));
    let mut queue = ctx
        .begin_command_queue(QueueType::Graphics, "integer output", false)
        .unwrap();
    CommandStream::new()
        .begin()
        .begin_render_pass(&BeginRenderPass {
            render_pass: pass,
            viewport,
            color_attachments: colors,
            clear_values: clears,
            ..Default::default()
        })
        .bind_graphics_pipeline(pipeline)
        .update_viewport(&viewport)
        .draw(&Draw {
            vertices,
            count: 3,
            instance_count: 1,
            ..Default::default()
        })
        .stop_drawing()
        .copy_image_to_buffer(&CopyImageBuffer {
            src: image,
            dst: readback,
            range: SubresourceRange::new(0, 1, 0, 2),
            ..Default::default()
        })
        .end()
        .append(&mut queue)
        .unwrap();
    let fence = ctx
        .submit_command_queue(&mut queue, &Default::default())
        .unwrap();
    ctx.wait_fence(fence).unwrap();
    ctx.invalidate_buffer(BufferView::new(readback)).unwrap();
    let words = ctx
        .map_buffer::<u32>(BufferView::new(readback))
        .unwrap()
        .to_vec();
    ctx.unmap_buffer(readback).unwrap();
    for (pixel, words) in words.chunks_exact(4).enumerate() {
        let expected = if pixel % 4 < 2 {
            clear
        } else {
            [(pixel / 8 + 1) as u32, u32::MAX, 0x7fc01234, 0x80000000]
        };
        assert_eq!(words, expected, "pixel {pixel}");
    }
    ctx.destroy_command_queue(queue);
    ctx.destroy_graphics_pipeline(pipeline);
    ctx.destroy_graphics_pipeline_layout(layout);
    ctx.destroy_buffer(readback);
    ctx.destroy_buffer(vertices);
    ctx.destroy_render_pass(pass);
    ctx.destroy_image(image);
}
