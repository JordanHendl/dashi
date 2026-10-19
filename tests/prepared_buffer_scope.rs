mod common;

use common::ValidationContext;
use dashi::*;

#[test]
#[serial_test::serial]
fn prepared_storage_buffer_supports_narrower_pipeline_stages() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;

    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();

    let img = ctx
        .make_image(&ImageInfo {
            debug_name: "fb",
            dim: [WIDTH, HEIGHT, 1],
            format: Format::RGBA8,
            mip_levels: 1,
            initial_data: None,
            ..Default::default()
        })
        .unwrap();

    let view = ImageView {
        img,
        ..Default::default()
    };

    let rp = ctx
        .make_render_pass(&RenderPassInfo {
            debug_name: "rp",
            viewport: Viewport {
                area: FRect2D {
                    w: WIDTH as f32,
                    h: HEIGHT as f32,
                    ..Default::default()
                },
                scissor: Rect2D {
                    w: WIDTH,
                    h: HEIGHT,
                    ..Default::default()
                },
                ..Default::default()
            },
            subpasses: &[SubpassDescription {
                color_attachments: &[AttachmentDescription::default()],
                depth_stencil_attachment: None,
                subpass_dependencies: &[],
            }],
        })
        .unwrap();

    let feedback = ctx
        .make_buffer(&BufferInfo {
            debug_name: "shared feedback",
            byte_size: 8,
            usage: BufferUsage::STORAGE,
            visibility: MemoryVisibility::CpuAndGpu,
            initial_data: Some(&[1, 0, 0, 0, 0, 0, 0, 0]),
        })
        .unwrap();
    let mut tables = Vec::new();
    let mut layouts = Vec::new();
    for stages in [
        vec![ShaderType::Fragment],
        vec![ShaderType::Vertex, ShaderType::Fragment],
    ] {
        let variables = [BindTableVariable {
            var_type: BindTableVariableType::Storage,
            binding: 0,
            count: 1,
        }];
        let shaders: Vec<_> = stages
            .iter()
            .map(|&shader_type| ShaderInfo {
                shader_type,
                variables: &variables,
            })
            .collect();
        let layout = ctx
            .make_bind_table_layout(&BindTableLayoutInfo {
                debug_name: "feedback",
                shaders: &shaders,
            })
            .unwrap();
        tables.push(
            ctx.make_bind_table(&BindTableInfo {
                debug_name: "feedback",
                layout,
                bindings: &[IndexedBindingInfo {
                    binding: 0,
                    resources: &[IndexedResource {
                        slot: 0,
                        resource: ShaderResource::StorageBuffer(BufferView::new(feedback)),
                    }],
                }],
                ..Default::default()
            })
            .unwrap(),
        );
        layouts.push(layout);
    }
    let vert = inline_spirv::inline_spirv!(
        r"#version 450
        vec2 positions[3] = vec2[3](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.0,0.5));
        void main() {
            gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
        }
    ",
        vert
    );

    let vert_feedback = inline_spirv::inline_spirv!(
        r"#version 450
        layout(set=0,binding=0) readonly buffer Feedback { uint words[]; } feedback;
        vec2 positions[3] = vec2[3](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.0,0.5));
        void main() {
            gl_Position = vec4(positions[gl_VertexIndex] * float(feedback.words[0]), 0.0, 1.0);
        }",
        vert
    );
    let frag_red = inline_spirv::inline_spirv!(
        r"#version 450
        layout(set=0,binding=0) buffer Feedback { uint words[]; } feedback;
        layout(location=0) out vec4 color;
        void main() { atomicOr(feedback.words[1], 2u); color = vec4(1.0,0.0,0.0,1.0); }
    ",
        frag
    );

    let frag_green = inline_spirv::inline_spirv!(
        r"#version 450
        layout(set=0,binding=0) buffer Feedback { uint words[]; } feedback;
        layout(location=0) out vec4 color;
        void main() { atomicOr(feedback.words[1], 2u); color = vec4(0.0,1.0,0.0,1.0); }
    ",
        frag
    );

    let layout_red = ctx
        .make_graphics_pipeline_layout(&GraphicsPipelineLayoutInfo {
            debug_name: "layout_red",
            vertex_info: VertexDescriptionInfo {
                entries: &[],
                stride: 0,
                rate: VertexRate::Vertex,
            },
            bt_layouts: [Some(layouts[0]), None, None, None],
            shaders: &[
                PipelineShaderInfo {
                    stage: ShaderType::Vertex,
                    spirv: vert,
                    entry_point: "main",
                    specialization: &[],
                },
                PipelineShaderInfo {
                    stage: ShaderType::Fragment,
                    spirv: frag_red,
                    entry_point: "main",
                    specialization: &[],
                },
            ],
            details: Default::default(),
        })
        .unwrap();

    let layout_green = ctx
        .make_graphics_pipeline_layout(&GraphicsPipelineLayoutInfo {
            debug_name: "layout_green",
            vertex_info: VertexDescriptionInfo {
                entries: &[],
                stride: 0,
                rate: VertexRate::Vertex,
            },
            bt_layouts: [Some(layouts[1]), None, None, None],
            shaders: &[
                PipelineShaderInfo {
                    stage: ShaderType::Vertex,
                    spirv: vert_feedback,
                    entry_point: "main",
                    specialization: &[],
                },
                PipelineShaderInfo {
                    stage: ShaderType::Fragment,
                    spirv: frag_green,
                    entry_point: "main",
                    specialization: &[],
                },
            ],
            details: Default::default(),
        })
        .unwrap();

    let subpass_info = ctx
        .render_pass_subpass_info(rp, 0)
        .expect("render pass subpass info");

    let pipe_red = ctx
        .make_graphics_pipeline(&GraphicsPipelineInfo {
            debug_name: "pipe_red",
            layout: layout_red,
            attachment_formats: subpass_info.color_formats.clone(),
            depth_format: subpass_info.depth_format,
            subpass_samples: subpass_info.samples.clone(),
            ..Default::default()
        })
        .unwrap();

    let pipe_green = ctx
        .make_graphics_pipeline(&GraphicsPipelineInfo {
            debug_name: "pipe_green",
            layout: layout_green,
            attachment_formats: subpass_info.color_formats,
            depth_format: subpass_info.depth_format,
            subpass_samples: subpass_info.samples,
            ..Default::default()
        })
        .unwrap();

    let vb = ctx
        .make_buffer(&BufferInfo {
            debug_name: "vb",
            byte_size: 4,
            visibility: MemoryVisibility::Gpu,
            usage: BufferUsage::VERTEX,
            initial_data: Some(&[0u8; 4]),
        })
        .unwrap();

    let readback = ctx
        .make_buffer(&BufferInfo {
            debug_name: "readback",
            byte_size: WIDTH * HEIGHT * 4,
            visibility: MemoryVisibility::CpuAndGpu,
            ..Default::default()
        })
        .unwrap();
    let viewport = Viewport {
        area: FRect2D {
            w: WIDTH as f32,
            h: HEIGHT as f32,
            ..Default::default()
        },
        scissor: Rect2D {
            w: WIDTH,
            h: HEIGHT,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut encoder = driver::command::CommandEncoder::new(QueueType::Graphics);
    encoder.prepare_buffer(vb, UsageBits::VERTEX_READ, None);
    encoder.prepare_buffer_requirement(
        BufferStateRequirement {
            buffer: feedback,
            usage: UsageBits::STORAGE_READ | UsageBits::STORAGE_WRITE,
            stages: ShaderStageMask::VERTEX | ShaderStageMask::FRAGMENT,
        },
        None,
    );
    encoder.begin_render_pass(&driver::command::BeginRenderPass {
        render_pass: rp,
        viewport,
        color_attachments: [Some(view), None, None, None, None, None, None, None],
        clear_values: [
            Some(ClearValue::Color([0.0; 4])),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        ..Default::default()
    });
    for (pipeline, table) in [
        (pipe_red, tables[0]),
        (pipe_green, tables[1]),
        (pipe_red, tables[0]),
        (pipe_green, tables[1]),
    ] {
        encoder.bind_graphics_pipeline(pipeline);
        encoder.update_graphics_pipeline_state(&driver::command::GraphicsPipelineStateUpdate {
            viewport: Some(viewport),
        });
        encoder.draw(&driver::command::Draw {
            vertices: vb,
            bind_tables: [Some(table), None, None, None],
            count: 3,
            instance_count: 1,
            ..Default::default()
        });
    }
    encoder.end_drawing();
    encoder.copy_image_to_buffer(&driver::command::CopyImageBuffer {
        src: img,
        dst: readback,
        ..Default::default()
    });
    let mut ring = ctx
        .make_command_ring(&CommandQueueInfo2::default())
        .unwrap();
    encoder.prepare_buffer(feedback, UsageBits::HOST_READ, None);
    for _ in 0..7 {
        ring.record(|queue| {
            encoder.append(queue).unwrap();
        })
        .unwrap();
        ring.submit(&Default::default()).unwrap();
        ring.wait_all().unwrap();
        let pixels = ctx
            .map_buffer::<u8>(BufferView::new(readback))
            .unwrap()
            .to_vec();
        ctx.unmap_buffer(readback).unwrap();
        let center = ((HEIGHT / 2 * WIDTH + WIDTH / 2) * 4) as usize;
        assert_eq!(&pixels[center..center + 4], &[0, 255, 0, 255]);
        let words = ctx
            .map_buffer::<u32>(BufferView::new(feedback))
            .unwrap()
            .to_vec();
        ctx.unmap_buffer(feedback).unwrap();
        assert_eq!(words, vec![1, 2]);
    }
}

#[test]
#[serial_test::serial]
fn prepared_scope_rejects_missing_stages_accesses_and_queue_ownership() {
    use dashi::driver::command::{CommandSink, EndDrawing, PrepareBuffer};
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 64;

    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();

    let img = ctx
        .make_image(&ImageInfo {
            debug_name: "fb",
            dim: [WIDTH, HEIGHT, 1],
            format: Format::RGBA8,
            mip_levels: 1,
            initial_data: None,
            ..Default::default()
        })
        .unwrap();

    let view = ImageView {
        img,
        ..Default::default()
    };

    let rp = ctx
        .make_render_pass(&RenderPassInfo {
            debug_name: "rp",
            viewport: Viewport {
                area: FRect2D {
                    w: WIDTH as f32,
                    h: HEIGHT as f32,
                    ..Default::default()
                },
                scissor: Rect2D {
                    w: WIDTH,
                    h: HEIGHT,
                    ..Default::default()
                },
                ..Default::default()
            },
            subpasses: &[SubpassDescription {
                color_attachments: &[AttachmentDescription::default()],
                depth_stencil_attachment: None,
                subpass_dependencies: &[],
            }],
        })
        .unwrap();

    let viewport = Viewport {
        area: FRect2D {
            w: WIDTH as f32,
            h: HEIGHT as f32,
            ..Default::default()
        },
        scissor: Rect2D {
            w: WIDTH,
            h: HEIGHT,
            ..Default::default()
        },
        ..Default::default()
    };

    let rw = UsageBits::STORAGE_READ | UsageBits::STORAGE_WRITE;
    let both = ShaderStageMask::VERTEX | ShaderStageMask::FRAGMENT;
    for (prepared_usage, prepared_stages, usage, stages, owner, accepted) in [
        (
            rw,
            both,
            UsageBits::STORAGE_READ,
            ShaderStageMask::FRAGMENT,
            QueueType::Graphics,
            true,
        ),
        (
            rw,
            ShaderStageMask::FRAGMENT,
            rw,
            ShaderStageMask::VERTEX,
            QueueType::Graphics,
            false,
        ),
        (
            UsageBits::STORAGE_READ,
            both,
            rw,
            ShaderStageMask::FRAGMENT,
            QueueType::Graphics,
            false,
        ),
        (
            rw,
            both,
            rw,
            ShaderStageMask::FRAGMENT,
            QueueType::Compute,
            false,
        ),
    ] {
        let buffer = ctx
            .make_buffer(&BufferInfo {
                debug_name: "prepared scope",
                byte_size: 4,
                usage: BufferUsage::STORAGE,
                visibility: MemoryVisibility::Gpu,
                ..Default::default()
            })
            .unwrap();
        let mut ring = ctx
            .make_command_ring(&CommandQueueInfo2::default())
            .unwrap();
        ring.record(|queue| {
            queue
                .prepare_buffer(&PrepareBuffer {
                    buffer,
                    usage: prepared_usage,
                    stages: prepared_stages,
                    queue: QueueType::Graphics,
                })
                .unwrap();
            queue
                .begin_render_pass(&driver::command::BeginRenderPass {
                    render_pass: rp,
                    viewport,
                    color_attachments: [Some(view), None, None, None, None, None, None, None],
                    clear_values: [
                        Some(ClearValue::Color([0.0; 4])),
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ],
                    ..Default::default()
                })
                .unwrap();

            let result = queue.prepare_buffer(&PrepareBuffer {
                buffer,
                usage,
                stages,
                queue: owner,
            });
            if accepted {
                result.unwrap();
                // The narrow read must not discard the prepared vertex/write scope.
                queue
                    .prepare_buffer(&PrepareBuffer {
                        buffer,
                        usage: rw,
                        stages: both,
                        queue: QueueType::Graphics,
                    })
                    .unwrap();
            } else {
                assert!(matches!(
                    result,
                    Err(GPUError::BufferBarrierInsideRenderPass { .. })
                ));
            }
            queue.end_drawing(&EndDrawing::default()).unwrap();
        })
        .unwrap();
        ring.submit(&Default::default()).unwrap();
        ring.wait_all().unwrap();
        ctx.destroy_buffer(buffer);
    }
}
