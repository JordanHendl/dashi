mod common;

use common::ValidationContext;
use dashi::*;

#[test]
fn pipeline_switch() {
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

    let vert = inline_spirv::inline_spirv!(
        r"#version 450
        vec2 positions[3] = vec2[3](vec2(-0.5,-0.5), vec2(0.5,-0.5), vec2(0.0,0.5));
        void main() {
            gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
        }
    ",
        vert
    );

    let frag_red = inline_spirv::inline_spirv!(
        r"#version 450
        layout(location=0) out vec4 color;
        void main() { color = vec4(1.0,0.0,0.0,1.0); }
    ",
        frag
    );

    let frag_green = inline_spirv::inline_spirv!(
        r"#version 450
        layout(location=0) out vec4 color;
        void main() { color = vec4(0.0,1.0,0.0,1.0); }
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
            bt_layouts: [None, None, None, None],
            shaders: &[
                PipelineShaderInfo {
                    stage: ShaderType::Vertex,
                    spirv: vert,
                    specialization: &[],
                },
                PipelineShaderInfo {
                    stage: ShaderType::Fragment,
                    spirv: frag_red,
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
            bt_layouts: [None, None, None, None],
            shaders: &[
                PipelineShaderInfo {
                    stage: ShaderType::Vertex,
                    spirv: vert,
                    specialization: &[],
                },
                PipelineShaderInfo {
                    stage: ShaderType::Fragment,
                    spirv: frag_green,
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

    //    let mut list = ctx
    //        .begin_command_queue(&CommandQueueInfo { debug_name: "draw", ..Default::default() })
    //        .unwrap();
    //
    //    list.begin_drawing(&DrawBegin {
    //        viewport: Viewport {
    //            area: FRect2D { w: WIDTH as f32, h: HEIGHT as f32, ..Default::default() },
    //            scissor: Rect2D { w: WIDTH, h: HEIGHT, ..Default::default() },
    //            ..Default::default()
    //        },
    //        pipeline: pipe_red,
    //        render_target: rt,
    //        clear_values: &[ClearValue::Color([0.0,0.0,0.0,1.0])],
    //    }).unwrap();
    //
    //    list.append(Command::Draw(Draw { vertices: vb, count: 3, ..Default::default() }));
    //
    //    list.bind_pipeline(pipe_green).unwrap();
    //    list.append(Command::Draw(Draw { vertices: vb, count: 3, ..Default::default() }));
    //
    //    list.end_drawing().unwrap();
    //
    //    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
    //    ctx.wait(fence).unwrap();

    //    ctx.destroy_cmd_queue(list);
}
