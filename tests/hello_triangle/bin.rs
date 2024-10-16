use dashi::*;
use sdl2::{event::Event, keyboard::Keycode};
use std::time::{Duration, Instant};

#[cfg(feature = "dashi-tests")]
pub struct Timer {
    start_time: Option<Instant>,
    elapsed: Duration,
    is_paused: bool,
}

#[cfg(feature = "dashi-tests")]
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

#[cfg(feature = "dashi-tests")]
fn main() {

    // The GPU context that holds all the data.
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


    // Allocate the framebuffer image & view
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
    
    // Make the bind group layout. This describes the bindings into a shader.
    let bg_layout = ctx
        .make_bind_group_layout(&BindGroupLayoutInfo {
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Vertex,
                variables: &[BindGroupVariable {
                    var_type: BindGroupVariableType::DynamicUniform,
                    binding: 0,
                }],
            }],
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
            bg_layout,
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
            color_attachments: &[Attachment {
                view: fb_view,
                clear_color: [0.0, 0.0, 0.0, 1.0],
                ..Default::default()
            }],
            depth_stencil_attachment: None,
        })
        .unwrap();
    
    // Make a graphics pipeline. This matches a pipeline layout to a render pass.
    let graphics_pipeline = ctx
        .make_graphics_pipeline(&GraphicsPipelineInfo {
            layout: pipeline_layout,
            render_pass,
        })
        .unwrap();
    
    // Make dynamic allocator to use for dynamic buffers.
    let mut allocator = ctx.make_dynamic_allocator(&Default::default()).unwrap();
    
    // Make bind group what we want to bind to what was described in the Bind Group Layout.
    let bind_group = ctx
        .make_bind_group(&BindGroupInfo {
            layout: bg_layout,
            bindings: &[BindingInfo {
                resource: ShaderResource::Dynamic(&allocator),
                binding: 0,
            }],
            ..Default::default()
        })
        .unwrap();
    
    // Event pump for events
    let mut event_pump = ctx.get_sdl_event();
    // Display for windowing
    let mut display = ctx.make_display(&Default::default()).unwrap();
    // Timer to move the triangle
    let mut timer = Timer::new();

    timer.start();
    'running: loop {
        // Reset the allocator
        allocator.reset();
        
        // Listen to events
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

        // Get the next image from the display.
        let (img, sem, _idx, _good) = ctx.acquire_new_image(&mut display).unwrap();

        // Begin recording commands
        let mut list = ctx.begin_command_list(&Default::default()).unwrap();

        // Begin render pass & bind pipeline
        list.begin_drawing(&DrawBegin {
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
            pipeline: graphics_pipeline,
        })
        .unwrap();


        // Bump alloc some data to write the triangle position to.
        let mut buf = allocator.bump().unwrap();
        let pos = &mut buf.slice::<[f32; 2]>()[0];
        pos[0] = (timer.elapsed_ms() as f32 / 1000.0).sin();
        pos[1] = (timer.elapsed_ms() as f32 / 1000.0).cos();

        // Append a draw call using our vertices & indices & dynamic buffers
        list.append(Command::DrawIndexedCommand(DrawIndexed {
            vertices,
            indices,
            index_count: INDICES.len() as u32,
            bind_groups: [Some(bind_group), None, None, None],
            dynamic_buffers: [Some(buf), None, None, None],
            ..Default::default()
        }));
        
        // End drawing.
        list.end_drawing().expect("Error ending drawing!");
        
        // Blit the framebuffer to the display's image
        list.blit(ImageBlit {
            src: fb_view,
            dst: img,
            filter: Filter::Nearest,
            ..Default::default()
        });
        
        // Submit our recorded commands
        let (sem, fence) = ctx.submit(&mut list, Some(&[sem])).unwrap();

        // Present the display image, waiting on the semaphore that will signal when our
        // drawing/blitting is done.
        ctx.present_display(&display, &[sem]).unwrap();

        // Signal the context to free our command list on the next submit call. This is nice so
        // that we don't have to manually manage it.
        ctx.release_list_on_next_submit(fence, list);
    }
}

#[cfg(not(feature = "dashi-tests"))]
fn main() { //none
}
