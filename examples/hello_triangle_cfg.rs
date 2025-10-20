// examples/hello_triangle_rpass_cfg.rs
// (or src/bin/hello_triangle_rpass_cfg.rs)

use dashi::driver::command::{BeginDrawing, BlitImage, DrawIndexed};
use dashi::*;
use std::time::{Duration, Instant};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::platform::run_return::EventLoopExtRunReturn;

// Only using the RenderPass config for now:
use dashi::gpu::vulkan::structs::config as cfg;

pub struct Timer {
    start_time: Option<Instant>,
    elapsed: Duration,
    is_paused: bool,
}
impl Timer {
    pub fn new() -> Self {
        Self { start_time: None, elapsed: Duration::new(0, 0), is_paused: false }
    }
    pub fn start(&mut self) {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        } else if self.is_paused {
            self.start_time = Some(Instant::now() - self.elapsed);
            self.is_paused = false;
        }
    }
    pub fn elapsed_ms(&self) -> u128 {
        if let Some(t0) = self.start_time {
            if self.is_paused { self.elapsed.as_millis() } else { t0.elapsed().as_millis() }
        } else {
            self.elapsed.as_millis()
        }
    }
}

// Minimal RenderPass YAML (only what we need here).
// Swap this for `std::fs::read_to_string("configs/gbuffer.rpass.yaml")?` later.
const RENDERPASS_YAML: &str = r#"
debug_name: "gbuffer"
viewport:
  area:    { x: 0.0, y: 0.0, w: 1280.0, h: 1024.0 }
  scissor: { x: 0,   y: 0,   w: 1280,   h: 1024   }
  min_depth: 0.0
  max_depth: 1.0
subpasses:
  - color_attachments:
      - { format: RGBA8, samples: S1, load_op: Clear, store_op: Store,
          stencil_load_op: DontCare, stencil_store_op: DontCare }
    depth_stencil_attachment: ~
    subpass_dependencies: []
"#;

fn main() {
    // GPU context
    let device = SelectedDevice::default();
    println!("Using device {}", device);
    let mut ctx = gpu::Context::new(&ContextInfo { device }).expect("ctx");

    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 1024;

    // Geometry
    const VERTICES: [[f32; 2]; 3] = [[0.0, -0.5], [0.5, 0.5], [-0.5, 0.5]];
    const INDICES:  [u32; 3]     = [0, 1, 2];

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

    // Offscreen framebuffer (blitted to the swapchain image)
    let fb = ctx
        .make_image(&ImageInfo {
            debug_name: "color_attachment",
            dim: [WIDTH, HEIGHT, 1],
            format: Format::RGBA8,
            mip_levels: 1,
            initial_data: None,
            ..Default::default()
        })
        .unwrap();
    let fb_view = ImageView { img: fb, ..Default::default() };

    // BindGroupLayout in code (pipelines will keep being in code for now).
    let bgl = ctx
        .make_bind_group_layout(&BindGroupLayoutInfo {
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Vertex,
                variables: &[BindGroupVariable {
                    var_type: BindGroupVariableType::DynamicUniform,
                    binding: 0,
                    ..Default::default()
                }],
            }],
            debug_name: "Hello Triangle BGL",
        })
        .unwrap();

    // Pipeline layout in code
    let shaders = [
        PipelineShaderInfo {
            stage: ShaderType::Vertex,
            spirv: inline_spirv::inline_spirv!(r#"
                #version 450
                layout(location = 0) in vec2 inPosition;
                layout(location = 0) out vec2 frag_color;
                layout(binding = 0) uniform position_offset { vec2 pos; };
                void main() {
                    frag_color = inPosition;
                    gl_Position = vec4(inPosition + pos, 0.0, 1.0);
                }
            "#, vert),
            specialization: &[],
        },
        PipelineShaderInfo {
            stage: ShaderType::Fragment,
            spirv: inline_spirv::inline_spirv!(r#"
                #version 450 core
                layout(location = 0) in  vec2 frag_color;
                layout(location = 0) out vec4 out_color;
                void main() { out_color = vec4(frag_color.xy, 0, 1); }
            "#, frag),
            specialization: &[],
        },
    ];

    let gp_layout = ctx
        .make_graphics_pipeline_layout(&GraphicsPipelineLayoutInfo {
            debug_name: "Hello Triangle Layout",
            vertex_info: VertexDescriptionInfo {
                entries: &[VertexEntryInfo { format: ShaderPrimitiveType::Vec2, location: 0, offset: 0 }],
                stride: 8,
                rate: VertexRate::Vertex,
            },
            bg_layouts: [Some(bgl), None, None, None],
            bt_layouts: [None, None, None, None],
            shaders: &shaders,
            details: GraphicsPipelineDetails {
                subpass: 0,
                color_blend_states: vec![Default::default()],
                topology: Topology::TriangleList,
                culling: CullMode::Back,
                front_face: VertexOrdering::Clockwise,
                depth_test: None,
                dynamic_states: vec![DynamicState::Viewport, DynamicState::Scissor],
            },
        })
        .expect("gfx layout");

    // ───────────────────────────────────────────────────────────────
    // RenderPass is parsed from YAML → borrowed → make_render_pass
    // ───────────────────────────────────────────────────────────────
    let rp_cfg = cfg::RenderPassCfg::from_yaml(RENDERPASS_YAML).expect("parse renderpass yaml");
    let rp_view = rp_cfg.borrow();                 // holds temporary Vec for subpass slices
    let rp_info: RenderPassInfo = rp_view.info;    // borrowed view into rp_cfg
    let render_pass = ctx.make_render_pass(&rp_info).expect("make render pass");

    // Graphics pipeline in code (still referencing the parsed render pass)
    let graphics_pipeline = ctx
        .make_graphics_pipeline(&GraphicsPipelineInfo {
            debug_name: "Hello Triangle Pipeline",
            layout: gp_layout,
            render_pass,
            subpass_id: 0,
        })
        .expect("pipeline");

    // Dynamic allocator + BindGroup
    let mut allocator = ctx.make_dynamic_allocator(&Default::default()).unwrap();
    let bind_group = ctx
        .make_bind_group(&BindGroupInfo {
            debug_name: "Hello Triangle BG",
            layout: bgl,
            bindings: &[BindingInfo { resource: ShaderResource::Dynamic(&allocator), binding: 0 }],
            ..Default::default()
        })
        .unwrap();

    // Display / window
    let mut display = ctx.make_display(&Default::default()).unwrap();
    let mut timer = Timer::new();
    timer.start();

    // Command ring & semaphores
    let mut ring = ctx
        .make_command_ring(&CommandQueueInfo2 { debug_name: "cmd", ..Default::default() })
        .unwrap();
    let sems = ctx.make_semaphores(2).unwrap();

    // Main loop
    'running: loop {
        allocator.reset();

        // Events (close / Esc)
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
        if should_exit { break 'running; }

        // Acquire next image from swapchain
        let (img, sem, _idx, _ok) = ctx.acquire_new_image(&mut display).unwrap();

        ring.record(|list| {
            let mut stream = CommandStream::new().begin();

            // Begin render pass & bind pipeline
            let mut draw = stream.begin_drawing(&BeginDrawing {
                viewport: Viewport {
                    area: FRect2D { w: WIDTH as f32, h: HEIGHT as f32, ..Default::default() },
                    scissor: Rect2D { w: WIDTH, h: HEIGHT, ..Default::default() },
                    ..Default::default()
                },
                pipeline: graphics_pipeline,
                color_attachments: [Some(fb_view), None, None, None],
                depth_attachment: None,
                clear_values: [Some(ClearValue::Color([0.0, 0.0, 0.0, 1.0])), None, None, None],
            });

            // Dynamic offset buffer (moves triangle)
            let mut buf = allocator.bump().unwrap();
            let pos = &mut buf.slice::<[f32; 2]>()[0];
            let t = timer.elapsed_ms() as f32 / 1000.0;
            pos[0] = t.sin();
            pos[1] = t.cos();

            // Draw
            draw.draw_indexed(&DrawIndexed {
                vertices,
                indices,
                index_count: 3,
                bind_groups: [Some(bind_group), None, None, None],
                dynamic_buffers: [Some(buf), None, None, None],
                ..Default::default()
            });

            stream = draw.stop_drawing();

            // Blit to presentable image & transition
            stream.blit_images(&BlitImage {
                src: fb, dst: img.img, filter: Filter::Nearest, ..Default::default()
            });
            stream.prepare_for_presentation(img.img);
            stream.end().append(list);
        }).unwrap();

        // Submit & present
        ring.submit(&SubmitInfo {
            wait_sems: &[sem],
            signal_sems: &[sems[0], sems[1]],
            ..Default::default()
        }).unwrap();
        ctx.present_display(&display, &[sems[0], sems[1]]).unwrap();
    }
}

