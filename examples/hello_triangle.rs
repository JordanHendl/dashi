use dashi::driver::command::{BeginDrawing, BlitImage, DrawIndexed};
use dashi::*;
use std::time::{Duration, Instant};

pub struct Timer {
    start_time: Option<Instant>,
    elapsed: Duration,
    is_paused: bool,
}

impl Timer {
    // Create a new timer instance
    pub fn new() -> Timer {
        Timer {
            start_time: None,
            elapsed: Duration::new(0, 0),
            is_paused: false,
        }
    }

    // Start the timer
    pub fn start(&mut self) {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        } else if self.is_paused {
            // Resume from where it was paused
            self.start_time = Some(Instant::now() - self.elapsed);
            self.is_paused = false;
        }
    }

    // Stop the timer
    pub fn stop(&mut self) {
        if let Some(start_time) = self.start_time {
            self.elapsed = start_time.elapsed();
            self.start_time = None;
            self.is_paused = false;
        }
    }

    // Pause the timer
    pub fn pause(&mut self) {
        if let Some(start_time) = self.start_time {
            self.elapsed = start_time.elapsed();
            self.is_paused = true;
            self.start_time = None;
        }
    }

    // Reset the timer
    pub fn reset(&mut self) {
        self.start_time = None;
        self.elapsed = Duration::new(0, 0);
        self.is_paused = false;
    }

    // Get the current elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> u128 {
        if let Some(start_time) = self.start_time {
            if self.is_paused {
                self.elapsed.as_millis()
            } else {
                start_time.elapsed().as_millis()
            }
        } else {
            self.elapsed.as_millis()
        }
    }
}

fn viewport_for(size: [u32; 2]) -> Viewport {
    Viewport {
        area: FRect2D {
            w: size[0] as f32,
            h: size[1] as f32,
            ..Default::default()
        },
        scissor: Rect2D {
            w: size[0],
            h: size[1],
            ..Default::default()
        },
        ..Default::default()
    }
}

fn rebuild_framebuffer(
    ctx: &mut gpu::Context,
    size: [u32; 2],
) -> Result<(Handle<Image>, ImageView, Viewport), GPUError> {
    let fb = ctx.make_image(&ImageInfo {
        debug_name: "color_attachment",
        dim: [size[0], size[1], 1],
        format: Format::RGBA8,
        mip_levels: 1,
        initial_data: None,
        ..Default::default()
    })?;

    Ok((
        fb,
        ImageView {
            img: fb,
            ..Default::default()
        },
        viewport_for(size),
    ))
}

fn main() {
    let device = DeviceSelector::new()
        .unwrap()
        .select(DeviceFilter::default().add_required_type(DeviceType::Dedicated))
        .expect("Unable to select device!");
    println!("Using device {}", device);

    // The GPU context that holds all the data.
    let mut ctx = gpu::Context::new(&ContextInfo {
        device,
        ..Default::default()
    })
    .unwrap();

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

    // Allocate the vertices & indices.
    let vertices = ctx
        .make_buffer(&BufferInfo {
            debug_name: "vertices",
            byte_size: (VERTICES.len() * std::mem::size_of::<f32>() * 2) as u32,
            visibility: MemoryVisibility::Gpu,
            usage: BufferUsage::VERTEX,
            initial_data: unsafe { Some(VERTICES.align_to::<u8>().1) },
        })
        .unwrap();

    let indices = ctx
        .make_buffer(&BufferInfo {
            debug_name: "indices",
            byte_size: (INDICES.len() * std::mem::size_of::<u32>()) as u32,
            visibility: MemoryVisibility::Gpu,
            usage: BufferUsage::INDEX,
            initial_data: unsafe { Some(INDICES.align_to::<u8>().1) },
        })
        .unwrap();

    // Make the bind table layout. This describes the bindings into a shader.
    let bt_layout = ctx
        .make_bind_table_layout(&BindTableLayoutInfo {
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Vertex,
                variables: &[BindTableVariable {
                    var_type: BindTableVariableType::DynamicUniform,
                    binding: 0,
                    ..Default::default()
                }],
            }],
            debug_name: "Hello Triangle",
        })
        .unwrap();

    // Make a pipeline layout. This describes a graphics pipeline's state.
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
            bt_layouts: [Some(bt_layout), None, None, None],
            shaders: &[
                PipelineShaderInfo {
                    stage: ShaderType::Vertex,
                    spirv: inline_spirv::inline_spirv!(
                        r#"
#version 450
layout(location = 0) in vec2 inPosition;
layout(location = 0) out vec2 frag_color;

layout(binding = 0) uniform position_offset {
    vec2 pos;
};
void main() {
    frag_color = inPosition;
    gl_Position = vec4(inPosition + pos, 0.0, 1.0);
}
"#,
                        vert
                    ),
                    entry_point: "main",
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
                    entry_point: "main",
                    specialization: &[],
                },
            ],
            details: Default::default(),
            debug_name: "Hello Compute",
        })
        .expect("Unable to create GFX Pipeline Layout!");

    // Make a render pass. This describes the targets we wish to draw to.
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
            subpasses: &[SubpassDescription {
                color_attachments: &[AttachmentDescription {
                    ..Default::default()
                }],
                depth_stencil_attachment: None,
                subpass_dependencies: &[],
            }],
            debug_name: "renderpass",
        })
        .unwrap();

    // Make a graphics pipeline. This matches a pipeline layout to a render pass.
    let subpass_info = ctx
        .render_pass_subpass_info(render_pass, 0)
        .expect("render pass subpass info");
    let graphics_pipeline = ctx
        .make_graphics_pipeline(&GraphicsPipelineInfo {
            layout: pipeline_layout,
            attachment_formats: subpass_info.color_formats,
            depth_format: subpass_info.depth_format,
            subpass_samples: subpass_info.samples,
            debug_name: "Pipeline",
            ..Default::default()
        })
        .unwrap();

    // Make dynamic allocator to use for dynamic buffers.
    let mut allocator = ctx.make_dynamic_allocator(&Default::default()).unwrap();

    // Make bind table to match the bind table layout.
    let bind_table = ctx
        .make_bind_table(&BindTableInfo {
            debug_name: "Hello Triangle",
            layout: bt_layout,
            bindings: &[IndexedBindingInfo {
                resources: &[IndexedResource {
                    resource: ShaderResource::Dynamic(allocator.state().clone()),
                    slot: 0,
                }],
                binding: 0,
            }],
            ..Default::default()
        })
        .unwrap();

    // Display for windowing
    let mut display = DisplayBuilder::new()
        .title("Hello Triangle")
        .size(WIDTH, HEIGHT)
        .resizable(true)
        .build(&mut ctx)
        .unwrap();
    let mut display_size = match ctx.prepare_display(&mut display).unwrap() {
        DisplayStatus::Closed => return,
        DisplayStatus::Ready { size } | DisplayStatus::Resized { size } => size,
    };
    let (mut fb, mut fb_view, mut viewport) = rebuild_framebuffer(&mut ctx, display_size).unwrap();
    // Timer to move the triangle
    let mut timer = Timer::new();

    timer.start();
    let mut ring = ctx
        .make_command_ring(&CommandQueueInfo2 {
            debug_name: "cmd",
            ..Default::default()
        })
        .unwrap();
    let render_sems = ctx.make_semaphores(3).unwrap();
    'running: loop {
        let frame_slot = ring.current_index();
        // Reset the allocator
        allocator.reset();

        match ctx.prepare_display(&mut display).unwrap() {
            DisplayStatus::Closed => break 'running,
            DisplayStatus::Ready { size } => {
                display_size = size;
            }
            DisplayStatus::Resized { size } => {
                display_size = size;
                ctx.destroy_image(fb);
                let rebuilt = rebuild_framebuffer(&mut ctx, size).unwrap();
                fb = rebuilt.0;
                fb_view = rebuilt.1;
                viewport = rebuilt.2;
            }
        }

        // Get the next image from the display.
        let (img, sem, _idx, _good) = match ctx.acquire_new_image(&mut display) {
            Ok(frame) => frame,
            Err(GPUError::DisplayNeedsRebuild) => continue,
            Err(err) => panic!("acquire_new_image failed: {:?}", err),
        };

        ring.record(|list| {
            // Begin render pass & bind pipeline
            let stream = CommandStream::new().begin();

            // Begin render pass & bind pipeline
            let draw = stream.begin_drawing(&BeginDrawing {
                viewport,
                render_pass,
                pipeline: graphics_pipeline,
                color_attachments: [Some(fb_view), None, None, None, None, None, None, None],
                depth_attachment: None,
                clear_values: [
                    Some(ClearValue::Color([0.0, 0.0, 0.0, 1.0])),
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

            // Bump alloc some data to write the triangle position to.
            let mut buf = allocator.bump().unwrap();
            let pos = &mut buf.slice::<[f32; 2]>()[0];
            pos[0] = (timer.elapsed_ms() as f32 / 1000.0).sin();
            pos[1] = (timer.elapsed_ms() as f32 / 1000.0).cos();

            // Append a draw call using our vertices & indices & dynamic buffers
            let draw = draw.draw_indexed(&DrawIndexed {
                vertices,
                indices,
                index_count: INDICES.len() as u32,
                bind_tables: [Some(bind_table), None, None, None],
                dynamic_buffers: [Some(buf), None, None, None],
                ..Default::default()
            });

            // End drawing and blit the framebuffer to the display's image.
            let stream = draw.stop_drawing().blit_images(&BlitImage {
                src: fb,
                dst: img.img,
                filter: Filter::Nearest,
                ..Default::default()
            });

            // Transition the display image for presentation
            let stream = stream.prepare_for_presentation(img.img);

            stream.end().append(list).unwrap();
        })
        .unwrap();
        // Submit our recorded commands
        ring.submit(&SubmitInfo {
            wait_sems: &[sem],
            signal_sems: &[render_sems[frame_slot]],
            ..Default::default()
        })
        .unwrap();

        // Present the display image, waiting on the semaphore that will signal when our
        // drawing/blitting is done.
        match ctx.present_display(&display, &[render_sems[frame_slot]]) {
            Ok(()) => {}
            Err(GPUError::DisplayNeedsRebuild) => continue,
            Err(err) => panic!("present_display failed: {:?}", err),
        }
    }

    let _ = display_size;
    ctx.destroy_image(fb);
}
