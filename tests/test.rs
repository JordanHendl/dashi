use dashi::*;
use sdl2::{event::Event, keyboard::Keycode};

#[cfg(feature = "dashi-tests")]
fn main() {
    let mut ctx = gpu::Context::new(&Default::default()).unwrap();

    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 1024;

    const VERTICES: [[f32; 2]; 3] = [
        [0.0, -0.5], // Vertex 0: Bottom
        [0.5, 0.5],  // Vertex 1: Top right
        [-0.5, 0.5], // Vertex 2: Top left
    ];

    const INDICES: [u32; 3] = [
        0, 1, 2, // Triangle: uses vertices 0, 1, and 2
    ];

    let vertices = ctx
        .make_buffer(&BufferInfo {
            debug_name: "vertices",
            byte_size: (VERTICES.len() * std::mem::size_of::<f32>() * 2) as u32,
            visibility: MemoryVisibility::Gpu,
            initial_data: unsafe { Some(VERTICES.align_to::<u8>().1) },
        })
        .unwrap();

    let indices = ctx
        .make_buffer(&BufferInfo {
            debug_name: "indices",
            byte_size: (INDICES.len() * std::mem::size_of::<u32>()) as u32,
            visibility: MemoryVisibility::Gpu,
            initial_data: unsafe { Some(INDICES.align_to::<u8>().1) },
        })
        .unwrap();

    let fb = ctx
        .make_image(&ImageInfo {
            debug_name: "color_attachment",
            dim: [WIDTH, HEIGHT, 1],
            format: Format::RGBA8,
            mip_levels: 1,
            initial_data: None,
        })
        .unwrap();

    let fb_view = ctx
        .make_image_view(&ImageViewInfo {
            img: fb,
            ..Default::default()
        })
        .unwrap();

    let bg_layout = ctx
        .make_bind_group_layout(&BindGroupLayoutInfo {
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Vertex,
                variables: &[BindGroupVariable {
                    var_type: BindGroupVariableType::Uniform,
                    binding: 0,
                }],
            }],
        })
        .unwrap();

    let pipeline_layout = ctx
        .make_graphics_pipeline_layout(&GraphicsPipelineLayoutInfo {
            vertex_info: VertexDescriptionInfo {
                entries: &[VertexEntryInfo {
                    format: ShaderPrimitiveType::Vec2,
                    location: 0,
                    offset: 0,
                }],
                stride: 8,
                rate: VertexRate::Vertex,
            },
            bg_layout,
            shaders: &[
                PipelineShaderInfo {
                    stage: ShaderType::Vertex,
                    spirv: inline_spirv::inline_spirv!(
                        r#"
#version 450
layout(location = 0) in vec2 inPosition;
layout(location = 0) out vec2 frag_color;
void main() {
    frag_color = inPosition;
    gl_Position = vec4(inPosition, 0.0, 1.0);
}
"#,
                        vert
                    ),
                    specialization: &[],
                },
                PipelineShaderInfo {
                    stage: ShaderType::Fragment,
                    spirv: inline_spirv::inline_spirv!(
                        r#"
    #version 450 core
    layout(location = 0) in vec2 frag_color;
    layout(location = 0) out vec4 out_color;
    void main() { out_color = vec4(frag_color.xy, 0, 1); }
"#,
                        frag
                    ),
                    specialization: &[],
                },
            ],
            details: Default::default(),
        })
        .expect("Unable to create GFX Pipeline Layout!");

    let render_pass = ctx
        .make_render_pass(&RenderPassInfo {
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
            pipeline_layout,
            color_attachments: &[Attachment {
                view: fb_view,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                ..Default::default()
            }],
            depth_stencil_attachment: None,
        })
        .unwrap();

    let graphics_pipeline = ctx
        .make_graphics_pipeline(&GraphicsPipelineInfo {
            layout: pipeline_layout,
            render_pass,
        })
        .unwrap();

    let mut event_pump = ctx.get_sdl_event();
    let mut display = ctx.make_display(&Default::default()).unwrap();
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    break 'running;
                }
                _ => {}
            }
        }

        let (img, sem, _idx, _good) = ctx.acquire_new_image(&mut display).unwrap();
        let mut list = ctx.begin_command_list(&Default::default()).unwrap();
        list.begin_render_pass(&RenderPassBegin {
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
            render_pass,
        })
        .unwrap();
        
        list.bind_graphics_pipeline(graphics_pipeline);
        list.append(Command::DrawIndexedCommand(DrawIndexed {
            vertices,
            indices,
            index_count: INDICES.len() as u32,
            ..Default::default()
        }));

        list.end_render_pass().expect("Error ending render pass!");

        list.blit(ImageBlit {
            src: fb_view,
            dst: img,
            filter: Filter::Nearest,
            ..Default::default()
        });
        let (sem, fence) = ctx.submit(& mut list, Some(&[sem])).unwrap();
        ctx.present_display(&display, &[sem]).unwrap();
    }
}

#[cfg(not(feature = "dashi-tests"))]
fn main() { //none
}
