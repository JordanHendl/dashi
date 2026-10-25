#![cfg(feature = "vulkan")]
mod common;

use common::ValidationContext;
use dashi::driver::command::{BeginDrawing, BeginRenderPass, CopyImageBuffer};
use dashi::*;

#[test]
#[serial_test::serial]
fn loaded_attachment_does_not_shift_later_attachment_clears() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();
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
    let colors: Vec<_> = (0..3)
        .map(|_| {
            ctx.make_image(&ImageInfo {
                dim: [4, 2, 1],
                format: Format::RGBA32F,
                initial_data: Some(bytemuck::cast_slice(&[0.125f32; 32])),
                ..Default::default()
            })
            .unwrap()
        })
        .collect();
    let mut attachments = [None; 8];
    for (slot, &image) in colors.iter().enumerate() {
        attachments[slot] = Some(ImageView {
            img: image,
            ..Default::default()
        });
    }
    let pass = ctx
        .make_render_pass(&RenderPassInfo {
            debug_name: "partial attachment clears",
            viewport,
            subpasses: &[SubpassDescription {
                color_attachments: &[
                    AttachmentDescription {
                        format: Format::RGBA32F,
                        load_op: LoadOp::Load,
                        ..Default::default()
                    },
                    AttachmentDescription {
                        format: Format::RGBA32F,
                        ..Default::default()
                    },
                    AttachmentDescription {
                        format: Format::RGBA32F,
                        ..Default::default()
                    },
                ],
                depth_stencil_attachment: None,
                subpass_dependencies: &[],
            }],
        })
        .unwrap();
    let vertex = inline_spirv::inline_spirv!(
        "#version 450\nvoid main(){gl_Position=vec4(0,0,0,1);}",
        vert
    );
    let fragment = inline_spirv::inline_spirv!("#version 450\nlayout(location=0) out vec4 a;layout(location=1) out vec4 b;layout(location=2) out vec4 c;void main(){a=b=c=vec4(1);}",frag);
    let layout = ctx
        .make_graphics_pipeline_layout(&GraphicsPipelineLayoutInfo {
            debug_name: "independent attachment blends",
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
                color_blend_states: vec![
                    ColorBlendState {
                        enable: false,
                        ..Default::default()
                    },
                    ColorBlendState {
                        enable: true,
                        ..Default::default()
                    },
                    ColorBlendState {
                        enable: false,
                        ..Default::default()
                    },
                ],
                ..Default::default()
            },
        })
        .unwrap();
    let subpass = ctx.render_pass_subpass_info(pass, 0).unwrap();
    let pipeline = ctx
        .make_graphics_pipeline(&GraphicsPipelineInfo {
            layout,
            attachment_formats: subpass.color_formats,
            depth_format: subpass.depth_format,
            subpass_samples: subpass.samples,
            ..Default::default()
        })
        .unwrap();
    let outputs: Vec<_> = (0..3)
        .map(|_| {
            ctx.make_buffer(&BufferInfo {
                byte_size: 128,
                visibility: MemoryVisibility::CpuAndGpu,
                ..Default::default()
            })
            .unwrap()
        })
        .collect();
    for drawing in [false, true] {
        let mut values = [None; 8];
        values[1] = Some(ClearValue::Color([0.25; 4]));
        values[2] = Some(ClearValue::Color([0.75; 4]));
        let mut queue = ctx
            .begin_command_queue(QueueType::Graphics, "partial clear", false)
            .unwrap();
        let stream = CommandStream::new().begin();
        let drawing_stream = if drawing {
            stream.begin_drawing(&BeginDrawing {
                viewport,
                render_pass: pass,
                pipeline,
                color_attachments: attachments,
                clear_values: values,
                ..Default::default()
            })
        } else {
            stream
                .begin_render_pass(&BeginRenderPass {
                    viewport,
                    render_pass: pass,
                    color_attachments: attachments,
                    clear_values: values,
                    ..Default::default()
                })
                .bind_graphics_pipeline(pipeline)
        };
        let mut stream = drawing_stream.stop_drawing();
        for (&src, &dst) in colors.iter().zip(&outputs) {
            stream = stream.copy_image_to_buffer(&CopyImageBuffer {
                src,
                dst,
                ..Default::default()
            });
        }
        stream.end().append(&mut queue).unwrap();
        let fence = ctx
            .submit_command_queue(&mut queue, &Default::default())
            .unwrap();
        ctx.wait_fence(fence).unwrap();
        for (&output, expected) in outputs.iter().zip([0.125f32, 0.25, 0.75]) {
            ctx.invalidate_buffer(BufferView::new(output)).unwrap();
            let pixels = ctx
                .map_buffer::<f32>(BufferView::new(output))
                .unwrap()
                .to_vec();
            ctx.unmap_buffer(output).unwrap();
            assert!(
                pixels.iter().all(|&v| v == expected),
                "drawing={drawing} expected={expected} actual={pixels:?}"
            );
        }
        ctx.destroy_command_queue(queue);
    }
    ctx.destroy_graphics_pipeline(pipeline);
    ctx.destroy_graphics_pipeline_layout(layout);
    ctx.destroy_render_pass(pass);
    for image in colors {
        ctx.destroy_image(image);
    }
    for buffer in outputs {
        ctx.destroy_buffer(buffer);
    }
}
