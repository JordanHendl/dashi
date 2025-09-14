//use dashi::*;
//use dashi::framegraph::{Graph, Node, PassDecl, BufferUse, ImageUse};
//use dashi::sync::state::{Access, Stage, ResState, ImageLayout};
//use ::image::{ColorType, save_buffer};
//use std::fs;
//
//#[test]
//fn compute_then_graphics() {
//    const WIDTH: u32 = 32;
//    const HEIGHT: u32 = 32;
//
//    let mut ctx = Context::headless(&Default::default()).unwrap();
//    let ctx_ptr = &mut ctx as *mut _ as usize;
//
//    // buffer that compute pass writes color into and graphics pass reads from
//    let color_buf = ctx
//        .make_buffer(&BufferInfo {
//            debug_name: "color_buf",
//            byte_size: 16,
//            visibility: MemoryVisibility::Gpu,
//            usage: BufferUsage::STORAGE,
//            initial_data: None,
//        })
//        .unwrap();
//
//    // output image
//    let img = ctx
//        .make_image(&ImageInfo {
//            debug_name: "out_img",
//            dim: [WIDTH, HEIGHT, 1],
//            format: Format::RGBA8,
//            mip_levels: 1,
//            ..Default::default()
//        })
//        .unwrap();
//    let view = ImageView { img, ..Default::default() };
//
//    let rp = ctx
//        .make_render_pass(&RenderPassInfo {
//            debug_name: "rp",
//            viewport: Viewport {
//                area: FRect2D { w: WIDTH as f32, h: HEIGHT as f32, ..Default::default() },
//                scissor: Rect2D { w: WIDTH, h: HEIGHT, ..Default::default() },
//                ..Default::default()
//            },
//            subpasses: &[SubpassDescription {
//                color_attachments: &[AttachmentDescription::default()],
//                depth_stencil_attachment: None,
//                subpass_dependencies: &[],
//            }],
//        })
//        .unwrap();
//
//    let rt = ctx
//        .make_render_target(&RenderTargetInfo { debug_name: "rt", render_pass: rp, attachments: &[view] })
//        .unwrap();
//
//    // shaders
//    let comp_spv = inline_spirv::inline_spirv!(r"#version 450
//layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
//layout(binding = 0) buffer Color { vec4 color; } buf;
//void main() { buf.color = vec4(0.0, 1.0, 0.0, 1.0); }
//", comp);
//
//    let vert_spv = inline_spirv::inline_spirv!(r"#version 450
//layout(location=0) out vec2 uv;
//vec2 positions[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
//void main(){
//    uv = (positions[gl_VertexIndex] + 1.0) * 0.5;
//    gl_Position = vec4(positions[gl_VertexIndex],0.0,1.0);
//}
//", vert);
//
//    let frag_spv = inline_spirv::inline_spirv!(r"#version 450
//layout(location=0) in vec2 uv;
//layout(location=0) out vec4 out_color;
//layout(binding=0) readonly buffer Color { vec4 color; } buf;
//void main(){ out_color = buf.color; }
//", frag);
//
//    // bind group layout for storage buffer used in compute and fragment stages
//    let bg_layout = ctx
//        .make_bind_group_layout(&BindGroupLayoutInfo {
//            debug_name: "bg_layout",
//            shaders: &[
//                ShaderInfo {
//                    shader_type: ShaderType::Compute,
//                    variables: &[BindGroupVariable { var_type: BindGroupVariableType::Storage, binding: 0, count: 1 }],
//                },
//                ShaderInfo {
//                    shader_type: ShaderType::Fragment,
//                    variables: &[BindGroupVariable { var_type: BindGroupVariableType::Storage, binding: 0, count: 1 }],
//                },
//            ],
//        })
//        .unwrap();
//
//    let bind_group = ctx
//        .make_bind_group(&BindGroupInfo {
//            debug_name: "bg",
//            layout: bg_layout,
//            bindings: &[BindingInfo { resource: ShaderResource::StorageBuffer(color_buf), binding: 0 }],
//            ..Default::default()
//        })
//        .unwrap();
//
//    let comp_layout = ctx
//        .make_compute_pipeline_layout(&ComputePipelineLayoutInfo {
//            bg_layouts: [Some(bg_layout), None, None, None],
//            bt_layouts: [None, None, None, None],
//            shader: &PipelineShaderInfo { stage: ShaderType::Compute, spirv: comp_spv, specialization: &[] },
//        })
//        .unwrap();
//
//    let comp_pipe = ctx
//        .make_compute_pipeline(&ComputePipelineInfo { debug_name: "comp", layout: comp_layout })
//        .unwrap();
//
//    let gfx_layout = ctx
//        .make_graphics_pipeline_layout(&GraphicsPipelineLayoutInfo {
//            debug_name: "gfx_layout",
//            vertex_info: VertexDescriptionInfo { entries: &[], stride: 0, rate: VertexRate::Vertex },
//            bg_layouts: [Some(bg_layout), None, None, None],
//            bt_layouts: [None, None, None, None],
//            shaders: &[
//                PipelineShaderInfo { stage: ShaderType::Vertex, spirv: vert_spv, specialization: &[] },
//                PipelineShaderInfo { stage: ShaderType::Fragment, spirv: frag_spv, specialization: &[] },
//            ],
//            details: Default::default(),
//        })
//        .unwrap();
//
//    let gfx_pipe = ctx
//        .make_graphics_pipeline(&GraphicsPipelineInfo { debug_name: "gfx", layout: gfx_layout, render_pass: rp, ..Default::default() })
//        .unwrap();
//
//    // vertex buffer for fullscreen triangle (not actually used but required)
//    let vb = ctx
//        .make_buffer(&BufferInfo {
//            debug_name: "vb",
//            byte_size: 4,
//            visibility: MemoryVisibility::Gpu,
//            usage: BufferUsage::VERTEX,
//            initial_data: Some(&[0u8;4]),
//        })
//        .unwrap();
//
//    // framegraph setup
//    let mut graph = Graph::new();
//    let buf_handle = color_buf; // Handle<Buffer>
//    let img_handle = img; // Handle<Image>
//
//    let mut compute_decl = PassDecl::new();
//    compute_decl.buffers.push(BufferUse {
//        handle: buf_handle,
//        state: ResState {
//            access: Access::SHADER_WRITE,
//            stages: Stage::COMPUTE_SHADER,
//            layout: ImageLayout::UNDEFINED,
//            _pad: 0,
//        },
//    });
//    let compute = graph.add_node(Node::new(compute_decl, move || {
//        let ctx = unsafe { &mut *(ctx_ptr as *mut Context) };
//        let mut list = ctx.begin_command_list(&Default::default()).unwrap();
//        list.dispatch_compute(Dispatch {
//            compute: comp_pipe,
//            workgroup_size: [1,1,1],
//            bindings: Bindings {
//                bind_groups: [Some(bind_group), None, None, None],
//                ..Default::default()
//            },
//            ..Default::default()
//        });
//        let fence = ctx.submit(&mut list, &Default::default()).unwrap();
//        ctx.wait(fence).unwrap();
//        ctx.destroy_cmd_list(list);
//    }));
//
//    let mut graphics_decl = PassDecl::new();
//    graphics_decl.buffers.push(BufferUse {
//        handle: buf_handle,
//        state: ResState {
//            access: Access::SHADER_READ,
//            stages: Stage::FRAGMENT_SHADER,
//            layout: ImageLayout::UNDEFINED,
//            _pad: 0,
//        },
//    });
//    graphics_decl.images.push(ImageUse {
//        handle: img_handle,
//        state: ResState {
//            access: Access::COLOR_ATTACHMENT_READ | Access::COLOR_ATTACHMENT_WRITE,
//            stages: Stage::COLOR_ATTACHMENT_OUTPUT,
//            layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
//            _pad: 0,
//        },
//    });
//
//    let graphics = graph.add_node(Node::new(graphics_decl, move || {
//        let ctx = unsafe { &mut *(ctx_ptr as *mut Context) };
//        let mut list = ctx.begin_command_list(&CommandListInfo { debug_name: "draw", ..Default::default() }).unwrap();
//        list.begin_drawing(&DrawBegin {
//            viewport: Viewport {
//                area: FRect2D { w: WIDTH as f32, h: HEIGHT as f32, ..Default::default() },
//                scissor: Rect2D { w: WIDTH, h: HEIGHT, ..Default::default() },
//                ..Default::default()
//            },
//            pipeline: gfx_pipe,
//            render_target: rt,
//            clear_values: &[ClearValue::Color([0.0,0.0,0.0,1.0])],
//        }).unwrap();
//        list.append(Command::Draw(Draw {
//            vertices: vb,
//            count: 3,
//            bindings: Bindings {
//                bind_groups: [Some(bind_group), None, None, None],
//                ..Default::default()
//            },
//            ..Default::default()
//        }));
//        list.end_drawing().unwrap();
//        let fence = ctx.submit(&mut list, &Default::default()).unwrap();
//        ctx.wait(fence).unwrap();
//        ctx.destroy_cmd_list(list);
//    }));
//
//    graph.add_dependency(graphics, compute);
//
//    // execute in order
//    graph.execute();
//
//    // read back image and save
////    let readback = ctx
////        .make_buffer(&BufferInfo { debug_name: "readback", byte_size: (WIDTH*HEIGHT*4) as u32, visibility: MemoryVisibility::CpuAndGpu, ..Default::default() })
////        .unwrap();
////    let mut list = ctx.begin_command_list(&Default::default()).unwrap();
////    list.copy_image_to_buffer(ImageBufferCopy { src: view, dst: readback, dst_offset: 0 });
////    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
////    ctx.wait(fence).unwrap();
////    let data = ctx.map_buffer::<u8>(readback).unwrap().to_vec();
////    ctx.unmap_buffer(readback).unwrap();
////    fs::create_dir_all("tests/output").unwrap();
////    save_buffer("tests/output/compute_then_graphics.png", &data, WIDTH, HEIGHT, ColorType::Rgba8).unwrap();
//
//    ctx.destroy_cmd_list(list);
//    ctx.destroy_buffer(readback);
//    ctx.destroy_buffer(color_buf);
//    ctx.destroy_buffer(vb);
//    ctx.destroy_render_target(rt);
//    ctx.destroy_render_pass(rp);
//    ctx.destroy_image(img);
//    ctx.destroy();
//}
