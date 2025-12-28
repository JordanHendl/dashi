use bytemuck::{Pod, Zeroable};
use dashi::builders::{
    BindTableBuilder, BindTableLayoutBuilder, GraphicsPipelineBuilder,
    GraphicsPipelineLayoutBuilder, RenderPassBuilder,
};
use dashi::driver::command::{BeginDrawing, CommandSink, DrawIndexed, EndDrawing};
use dashi::*;
use driver::command::MSImageResolve;
use glam::{Mat4, Quat, Vec3};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::platform::run_return::EventLoopExtRunReturn;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
}

const NUM_TRANSFORMS: usize = 1024;
fn main() -> Result<(), GPUError> {
    // Create a headless context.
    let mut ctx = gpu::Context::new(&ContextInfo::default())?;

    // Describe storage buffers for the per-instance transforms.
    let shader_info = ShaderInfo {
        shader_type: ShaderType::All,
        variables: &[
            BindTableVariable {
                var_type: BindTableVariableType::Storage,
                binding: 0,
                count: NUM_TRANSFORMS as u32,
            },
            BindTableVariable {
                var_type: BindTableVariableType::SampledImage,
                binding: 1,
                count: NUM_TRANSFORMS as u32,
            },
        ],
    };

    // Build the layout and table.
    let layout = BindTableLayoutBuilder::new("bindless_layout")
        .shader(shader_info)
        .build(&mut ctx)?;

    let r = ctx.make_image(&ImageInfo {
        debug_name: "bindless_triangle_r",
        dim: [1, 1, 1],
        format: Format::RGBA8,
        mip_levels: 1,
        initial_data: Some(&[255, 0, 0, 255]),
        ..Default::default()
    })?;

    let g = ctx.make_image(&ImageInfo {
        debug_name: "bindless_triangle_g",
        dim: [1, 1, 1],
        format: Format::RGBA8,
        mip_levels: 1,
        initial_data: Some(&[0, 255, 0, 255]),
        ..Default::default()
    })?;

    let b = ctx.make_image(&ImageInfo {
        debug_name: "bindless_triangle_b",
        dim: [1, 1, 1],
        format: Format::RGBA8,
        mip_levels: 1,
        initial_data: Some(&[0, 0, 255, 255]),
        ..Default::default()
    })?;

    // Allocate a dynamic buffer and bind it.
    let mut allocator = ctx.make_dynamic_allocator(&Default::default())?;
    let mut transforms = Vec::with_capacity(NUM_TRANSFORMS);
    let mut rng_state = 0x4d59_5df4_d0f3_3173u64;
    let mut rand = || {
        rng_state = rng_state
            .wrapping_mul(6364_1362_2384_6793_005)
            .wrapping_add(1);
        ((rng_state >> 32) as f32) / (u32::MAX as f32)
    };

    for _ in 0..NUM_TRANSFORMS {
        // Random uniform scale in [0.05, 0.15]
        let s = 0.05 + rand() * 0.10;
        let scale = Vec3::new(s, s, 1.0);

        // Triangle is roughly in [-0.5, 0.5] in both axes.
        // So its half-extent in clip space is about 0.5 * s.
        let half_extent = 0.5 * s;

        // Keep the *center* inside [-1 + half_extent, 1 - half_extent]
        let max_x = 1.0 - half_extent;
        let max_y = 1.0 - half_extent;

        let translation = Vec3::new(
            rand() * 2.0 * max_x - max_x, // [-max_x, max_x]
            rand() * 2.0 * max_y - max_y, // [-max_y, max_y]
            0.0,                          // fixed depth
        );

        // 2D rotation around Z only (screen space)
        let rotation = Quat::from_rotation_z(rand() * std::f32::consts::TAU);

        let matrix = Mat4::from_scale_rotation_translation(scale, rotation, translation);
        transforms.push(matrix.to_cols_array());
    }

    let mut idx = 0;
    let transforms_gpu: Vec<IndexedResource> = transforms
        .iter()
        .map(|array| {
            let b = ctx
                .make_buffer(&BufferInfo {
                    debug_name: "t",
                    byte_size: (array.len() * std::mem::size_of::<f32>()) as u32,
                    initial_data: Some(bytemuck::bytes_of(array)),
                    usage: BufferUsage::STORAGE,
                    ..Default::default()
                })
                .expect("Unable to make transform buffer");
            idx += 1;
            IndexedResource {
                resource: ShaderResource::StorageBuffer(BufferView::new(b)),
                slot: idx - 1,
            }
        })
        .collect();

    let sampler = ctx.make_sampler(&Default::default()).unwrap();
    let samplers_gpu: Vec<IndexedResource> = vec![
        IndexedResource {
            resource: ShaderResource::SampledImage(
                ImageView {
                    img: r,
                    ..Default::default()
                },
                sampler,
            ),
            slot: 0,
        },
        IndexedResource {
            resource: ShaderResource::SampledImage(
                ImageView {
                    img: g,
                    ..Default::default()
                },
                sampler,
            ),
            slot: 1,
        },
        IndexedResource {
            resource: ShaderResource::SampledImage(
                ImageView {
                    img: b,
                    ..Default::default()
                },
                sampler,
            ),
            slot: 2,
        },
    ];

    let table = BindTableBuilder::new("bindless_table")
        .layout(layout)
        .set(1)
        .binding(0, &transforms_gpu)
        .binding(1, &samplers_gpu)
        .build(&mut ctx)?;

    // Bind table layout for the per-draw instance index.
    let instance_shader_info = ShaderInfo {
        shader_type: ShaderType::All,
        variables: &[BindTableVariable {
            var_type: BindTableVariableType::DynamicUniform,
            binding: 0,
            count: 1,
        }],
    };
    let instance_layout = ctx.make_bind_table_layout(&BindTableLayoutInfo {
        debug_name: "instance_index_layout",
        shaders: &[instance_shader_info],
    })?;

    let instance_table = BindTableBuilder::new("instance_index")
        .layout(instance_layout)
        .set(0)
        .binding(
            0,
            &[IndexedResource {
                resource: ShaderResource::Dynamic(allocator.state().clone()),
                slot: 0,
            }],
        )
        .build(&mut ctx)?;

    // Geometry buffers.
    const VERTICES: [Vertex; 3] = [
        Vertex {
            position: [0.0, -0.5],
        },
        Vertex {
            position: [0.5, 0.5],
        },
        Vertex {
            position: [-0.5, 0.5],
        },
    ];
    const INDICES: [u32; 3] = [0, 1, 2];

    let vertices = ctx.make_buffer(&BufferInfo {
        debug_name: "bindless_triangle_vertices",
        byte_size: (VERTICES.len() * std::mem::size_of::<Vertex>()) as u32,
        visibility: MemoryVisibility::Gpu,
        usage: BufferUsage::VERTEX,
        initial_data: Some(bytemuck::cast_slice(&VERTICES)),
    })?;

    let indices = ctx.make_buffer(&BufferInfo {
        debug_name: "bindless_triangle_indices",
        byte_size: (INDICES.len() * std::mem::size_of::<u32>()) as u32,
        visibility: MemoryVisibility::Gpu,
        usage: BufferUsage::INDEX,
        initial_data: Some(bytemuck::cast_slice(&INDICES)),
    })?;

    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 720;

    let color_image = ctx.make_image(&ImageInfo {
        debug_name: "bindless_triangle_color",
        dim: [WIDTH, HEIGHT, 1],
        format: Format::RGBA8,
        mip_levels: 1,
        initial_data: None,
        samples: SampleCount::S4,
        ..Default::default()
    })?;

    let color_view = ImageView {
        img: color_image,
        ..Default::default()
    };

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

    let render_pass = RenderPassBuilder::new("bindless_triangle_rp", viewport)
        .add_subpass(
            &[AttachmentDescription {
                samples: SampleCount::S4,
                ..Default::default()
            }],
            None,
            &[],
        )
        .build(&mut ctx)?;

    let vertex_entries = [VertexEntryInfo {
        format: ShaderPrimitiveType::Vec2,
        location: 0,
        offset: 0,
    }];

    let vertex_shader = PipelineShaderInfo {
        stage: ShaderType::Vertex,
        spirv: inline_spirv::inline_spirv!(
            r#"
#version 450
#extension GL_EXT_nonuniform_qualifier : require
layout(location = 0) in vec2 in_position;
layout(location = 0) out vec2 out_coords;

layout(set = 0, binding = 0) uniform InstanceData {
    uint transform_index;
    uint tex_id;
} instance_data;

layout(set = 1, binding = 0) readonly buffer TransformBuffer {
    mat4 transform;
} transforms[1024];

void main() {
    uint index = instance_data.transform_index;
    mat4 model = transforms[nonuniformEXT(index)].transform;
    gl_Position = model * vec4(in_position, 0.0, 1.0);
    out_coords = in_position.xy;
}
"#,
            vert
        ),
        specialization: &[],
    };

    let fragment_shader = PipelineShaderInfo {
        stage: ShaderType::Fragment,
        spirv: inline_spirv::inline_spirv!(
            r#"
#version 450
#extension GL_EXT_nonuniform_qualifier : enable
layout(location = 0) in vec2 in_coords;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform InstanceData {
    uint transform_index;
    uint tex_id;
};

layout(set = 1, binding = 1) uniform sampler2D in_tex[1024];

void main() {
    out_color = texture(in_tex[tex_id], in_coords);
}
"#,
            frag
        ),
        specialization: &[],
    };

    let pipeline_layout = GraphicsPipelineLayoutBuilder::new("bindless_triangle_layout")
        .vertex_info(VertexDescriptionInfo {
            entries: &vertex_entries,
            stride: std::mem::size_of::<Vertex>(),
            rate: VertexRate::Vertex,
        })
        .bind_table_layout(0, instance_layout)
        .bind_table_layout(1, layout)
        .shader(vertex_shader)
        .shader(fragment_shader)
        .sample_count(SampleCount::S4)
        .build(&mut ctx)?;

    let graphics_pipeline = GraphicsPipelineBuilder::new("bindless_triangle_pipeline")
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .build(&mut ctx)?;

    println!("Created bind table: {:?}", table);
    let mut ring = ctx
        .make_command_ring(&CommandQueueInfo2 {
            debug_name: "cmd",
            ..Default::default()
        })
        .unwrap();

    let mut display = ctx
        .make_display(&DisplayInfo {
            window: WindowInfo {
                title: "bindless triangle".to_string(),
                size: [WIDTH, HEIGHT],
                resizable: false,
            },
            ..Default::default()
        })
        .unwrap();
    let sems = ctx.make_semaphores(2).unwrap();
    'running: loop {
        // Reset the allocator
        allocator.reset();

        // Listen to events
        let mut should_exit = false;
        {
            let event_loop = display.winit_event_loop();
            event_loop.run_return(|event, _target, control_flow| {
                *control_flow = ControlFlow::Exit;
                if let Event::WindowEvent { event, .. } = event {
                    match event {
                        WindowEvent::CloseRequested => should_exit = true,
                        WindowEvent::KeyboardInput {
                            input:
                                KeyboardInput {
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    state: ElementState::Pressed,
                                    ..
                                },
                            ..
                        } => should_exit = true,
                        _ => {}
                    }
                }
            });
        }
        if should_exit {
            break 'running;
        }

        // Get the next image from the display.
        let (img, sem, _idx, _good) = ctx.acquire_new_image(&mut display).unwrap();

        allocator.reset();
        ring.record(|list| {
            // Begin render pass & bind pipeline
            let stream = CommandStream::new().begin();

            let mut color_attachments = [None; 8];
            color_attachments[0] = Some(color_view);
            let mut clear_values = [None; 8];
            clear_values[0] = Some(ClearValue::Color([0.0, 0.0, 0.0, 1.0]));

            let mut drawing = stream.begin_drawing(&BeginDrawing {
                viewport,
                render_pass,
                pipeline: graphics_pipeline,
                color_attachments,
                depth_attachment: None,
                clear_values,
                ..Default::default()
            })
            .update_viewport(&viewport);

            for slot in 0..transforms.len() {
                #[repr(C)]
                struct Index {
                    tid: u32,
                    texid: u32,
                }
                let mut buf = allocator.bump().expect("allocator exhausted");
                buf.slice::<Index>()[0] = Index {
                    tid: slot as u32,
                    texid: slot as u32 % 3,
                };
                drawing = drawing.draw_indexed(&DrawIndexed {
                    vertices,
                    indices,
                    index_count: INDICES.len() as u32,
                    bind_tables: [Some(instance_table), Some(table), None, None],
                    dynamic_buffers: [Some(buf), None, None, None],
                    ..Default::default()
                });
            }

            let stream = drawing.stop_drawing().resolve_images(&MSImageResolve {
                src: color_view.img,
                dst: img.img,
                ..Default::default()
            });
            // Transition the display image for presentation
            let stream = stream.prepare_for_presentation(img.img);

            stream.end().append(list);
        })
        .expect("Unable to record drawing commands!");

        // Submit our recorded commands
        ring.submit(&SubmitInfo {
            wait_sems: &[sem],
            signal_sems: &[sems[0]],
            ..Default::default()
        })
        .unwrap();

        // Present the display image, waiting on the semaphore that will signal when our
        // drawing/blitting is done.
        ctx.present_display(&display, &[sems[0]]).unwrap();
    }

    ctx.destroy_dynamic_allocator(allocator);
    ctx.destroy();
    Ok(())
}
