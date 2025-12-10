// examples/hello_triangle_rpass_cfg.rs
// (or src/bin/hello_triangle_rpass_cfg.rs)

use dashi::driver::command::{BeginDrawing, DrawIndexed};
use dashi::gpu::execution::{BindingLayoutManager, BindingManager, PipelineManager};
use dashi::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::platform::run_return::EventLoopExtRunReturn;

pub struct Timer {
    start_time: Option<Instant>,
    elapsed: Duration,
    is_paused: bool,
}
impl Timer {
    pub fn new() -> Self {
        Self {
            start_time: None,
            elapsed: Duration::new(0, 0),
            is_paused: false,
        }
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
            if self.is_paused {
                self.elapsed.as_millis()
            } else {
                t0.elapsed().as_millis()
            }
        } else {
            self.elapsed.as_millis()
        }
    }
}

// Minimal YAML snippets (only what we need here).
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
      - debug_name: "hello_triangle.swapchain"
        format: BGRA8
        samples: S1
        load_op: Clear
        store_op: Store
        stencil_load_op: DontCare
        stencil_store_op: DontCare
        clear_value:
          kind: color
          value: [0.0, 0.0, 0.0, 1.0]
    depth_stencil_attachment: ~
    subpass_dependencies: []
"#;

const BINDING_LAYOUTS_YAML: &str = r#"
bind_group_layouts:
  - name: "hello_triangle.layouts.main"
    layout:
      debug_name: "Hello Triangle"
      shaders:
        - stage: Vertex
          variables:
            - var_type: DynamicUniform
              binding: 0
              count: 1
"#;

const BIND_GROUP_LAYOUT_NAME: &str = "hello_triangle.layouts.main";
const PIPELINE_NAME: &str = "hello_triangle.pipeline.main";
const RENDER_PASS_NAME: &str = "hello_triangle.render_pass.main";
const SWAPCHAIN_ATTACHMENT_NAME: &str = "hello_triangle.swapchain";
const HELLO_TRIANGLE_VERT_SPV: &str = env!("HELLO_TRIANGLE_VERT_SPV");
const HELLO_TRIANGLE_FRAG_SPV: &str = env!("HELLO_TRIANGLE_FRAG_SPV");

fn pipelines_yaml() -> String {
    format!(
        r#"graphics_pipeline_layouts:
  - name: "hello_triangle.layouts.pipeline"
    layout:
      debug_name: "Hello Triangle Pipeline Layout"
      vertex_info:
        entries:
          - format: Vec2
            location: 0
            offset: 0
        stride: 8
        rate: Vertex
      layouts:
        bg_layouts:
          - "hello_triangle.layouts.main"
          - ~
          - ~
          - ~
        bt_layouts:
          - ~
          - ~
          - ~
          - ~
      shaders:
        - stage: Vertex
          spirv_path: "{vert}"
        - stage: Fragment
          spirv_path: "{frag}"
graphics_pipelines:
  - name: "hello_triangle.pipeline.main"
    pipeline:
      debug_name: "Hello Triangle Pipeline"
      layout: "hello_triangle.layouts.pipeline"
      render_pass: "hello_triangle.render_pass.main"
      subpass_id: 0
"#,
        vert = HELLO_TRIANGLE_VERT_SPV,
        frag = HELLO_TRIANGLE_FRAG_SPV,
    )
}

fn main() {
    let device = DeviceSelector::new()
        .expect("Unable to make device selector")
        .select(DeviceFilter::default().add_required_type(DeviceType::Dedicated))
        .expect("Unable to find dedicated device!");

    println!("Using device {}", device);

    // The GPU context that holds all the data.
    let mut ctx = gpu::Context::new(&ContextInfo { device }).unwrap();

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

    // Use the layout manager to author and resolve bind layouts via string keys.
    let binding_layouts = BindingLayoutManager::new(&mut ctx as *mut _, 2, 1);
    let binding_manager: Arc<BindingManager> = binding_layouts.binding_manager();

    binding_layouts
        .load_from_yaml(BINDING_LAYOUTS_YAML)
        .expect("binding layouts loaded");

    let pipeline_manager = PipelineManager::new(&mut ctx as *mut _, &binding_layouts);

    let bg_layout = binding_layouts
        .bind_group_layout(BIND_GROUP_LAYOUT_NAME)
        .expect("bind group layout registered");

    // Make a render pass. This describes the targets we wish to draw to.
    let mut render_pass_with_images = ctx
        .make_render_pass_from_yaml(RENDERPASS_YAML)
        .expect("render_pass");

    let render_pass = render_pass_with_images.render_pass;
    let subpass_count = render_pass_with_images.subpasses.len();

    let mut render_passes = HashMap::new();
    render_passes.insert(RENDER_PASS_NAME.to_string(), render_pass);

    let pipelines_yaml = pipelines_yaml();
    pipeline_manager
        .load_from_yaml(&pipelines_yaml, &render_passes)
        .expect("pipelines loaded");

    let graphics_pipeline = pipeline_manager
        .graphics_pipeline(PIPELINE_NAME)
        .expect("graphics pipeline registered");

    let mut subpass_pipelines = vec![None; subpass_count];
    if let Some(slot) = subpass_pipelines.get_mut(0) {
        *slot = Some(graphics_pipeline);
    }

    // Make dynamic allocator to use for dynamic buffers.
    let mut allocator = ctx.make_dynamic_allocator(&Default::default()).unwrap();

    // Make bind group what we want to bind to what was described in the Bind Group Layout.
    let bind_group = binding_manager.alloc_bind_group(
        0,
        0,
        |ctx| {
            let bindings = [BindingInfo {
                resource: ShaderResource::Dynamic(allocator.state().clone()),
                binding: 0,
            }];
            ctx.make_bind_group(&BindGroupInfo {
                debug_name: "Hello Triangle",
                layout: bg_layout,
                bindings: &bindings,
                ..Default::default()
            })
            .expect("make bind group")
        },
        |_ctx, _| {},
    );

    // Display for windowing
    let mut display = ctx.make_display(&Default::default()).unwrap();
    // Timer to move the triangle
    let mut timer = Timer::new();

    timer.start();
    let mut ring = ctx
        .make_command_ring(&CommandQueueInfo2 {
            debug_name: "cmd",
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

        ring.record(|list| {
            let mut stream = CommandStream::new().begin();

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

            for (subpass_idx, subpass_targets) in
                render_pass_with_images.subpasses.iter_mut().enumerate()
            {
                if let Some(attachment) =
                    subpass_targets.find_color_attachment_mut(SWAPCHAIN_ATTACHMENT_NAME)
                {
                    attachment.view = img;
                }

                let Some(pipeline) = subpass_pipelines
                    .get(subpass_idx)
                    .and_then(|pipeline| *pipeline)
                else {
                    continue;
                };

                let mut draw = stream.begin_drawing(&BeginDrawing {
                    viewport,
                    render_pass,
                    pipeline,
                    color_attachments: subpass_targets.color_views(),
                    depth_attachment: subpass_targets.depth_view(),
                    clear_values: subpass_targets.color_clear_values(),
                    ..Default::default()
                });

                if subpass_idx == 0 {
                    let mut buf = allocator.bump().unwrap();
                    let pos = &mut buf.slice::<[f32; 2]>()[0];
                    pos[0] = (timer.elapsed_ms() as f32 / 1000.0).sin();
                    pos[1] = (timer.elapsed_ms() as f32 / 1000.0).cos();

                    draw.draw_indexed(&DrawIndexed {
                        vertices,
                        indices,
                        index_count: INDICES.len() as u32,
                        bind_groups: [Some(bind_group), None, None, None],
                        dynamic_buffers: [Some(buf), None, None, None],
                        ..Default::default()
                    });
                }

                stream = draw.stop_drawing();
            }

            stream.prepare_for_presentation(img.img);

            stream.end().append(list);
        })
        .unwrap();
        // Submit our recorded commands
        ring.submit(&SubmitInfo {
            wait_sems: &[sem],
            signal_sems: &[sems[0], sems[1]],
            ..Default::default()
        })
        .unwrap();

        // Present the display image, waiting on the semaphore that will signal when our
        // drawing/blitting is done.
        ctx.present_display(&display, &[sems[0], sems[1]]).unwrap();
    }
}
