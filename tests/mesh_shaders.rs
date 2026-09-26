mod common;
use common::ValidationContext;
use dashi::{driver::command::*, *};

fn compile(source: &str, kind: shaderc::ShaderKind) -> Vec<u32> {
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_2 as u32,
    );
    options.set_target_spirv(shaderc::SpirvVersion::V1_5);
    shaderc::Compiler::new()
        .unwrap()
        .compile_into_spirv(source, kind, "mesh-test", "main", Some(&options))
        .unwrap()
        .as_binary()
        .to_vec()
}

#[test]
#[serial_test::serial]
fn task_mesh_direct_indirect_and_count_draws_match() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();
    let caps = ctx.mesh_shader_capabilities();
    if !caps.mesh_shader || !caps.task_shader {
        eprintln!("mesh/task unsupported");
        return;
    }
    assert!(caps.max_mesh_output_vertices >= 3);
    let task = compile("#version 460\n#extension GL_EXT_mesh_shader : require\nlayout(local_size_x=1) in; void main(){EmitMeshTasksEXT(1,1,1);}", shaderc::ShaderKind::Task);
    let mesh = compile("#version 460\n#extension GL_EXT_mesh_shader : require\nlayout(local_size_x=1) in; layout(triangles,max_vertices=3,max_primitives=1) out; void main(){SetMeshOutputsEXT(3,1); gl_MeshVerticesEXT[0].gl_Position=vec4(-1,-1,0,1);gl_MeshVerticesEXT[1].gl_Position=vec4(3,-1,0,1);gl_MeshVerticesEXT[2].gl_Position=vec4(-1,3,0,1);gl_PrimitiveTriangleIndicesEXT[0]=uvec3(0,1,2);}", shaderc::ShaderKind::Mesh);
    let frag = compile(
        "#version 450\nlayout(location=0) out vec4 color; void main(){color=vec4(0,1,0,1);}",
        shaderc::ShaderKind::Fragment,
    );
    let stages = [ShaderType::Task, ShaderType::Mesh, ShaderType::Fragment]
        .into_iter()
        .zip([&task, &mesh, &frag])
        .map(|(stage, words)| PipelineShaderInfo {
            stage,
            spirv: words,
            entry_point: "main",
            specialization: &[],
        })
        .collect::<Vec<_>>();
    let layout = ctx
        .make_graphics_pipeline_layout(&GraphicsPipelineLayoutInfo {
            shaders: &stages,
            details: GraphicsPipelineDetails {
                culling: CullMode::None,
                ..Default::default()
            },
            debug_name: "mesh test",
            vertex_info: VertexDescriptionInfo {
                entries: &[],
                stride: 0,
                rate: VertexRate::Vertex,
            },
            bt_layouts: [None; 4],
        })
        .unwrap();
    let viewport = Viewport {
        area: FRect2D {
            w: 8.0,
            h: 8.0,
            ..Default::default()
        },
        scissor: Rect2D {
            w: 8,
            h: 8,
            ..Default::default()
        },
        ..Default::default()
    };
    let image = ctx
        .make_image(&ImageInfo {
            dim: [8, 8, 1],
            format: Format::RGBA8,
            ..Default::default()
        })
        .unwrap();
    let pass = ctx
        .make_render_pass(&RenderPassInfo {
            debug_name: "mesh test",
            viewport,
            subpasses: &[SubpassDescription {
                color_attachments: &[AttachmentDescription::default()],
                depth_stencil_attachment: None,
                subpass_dependencies: &[],
            }],
        })
        .unwrap();
    let subpass = ctx.render_pass_subpass_info(pass, 0).unwrap();
    let pipeline = ctx
        .make_graphics_pipeline(&GraphicsPipelineInfo {
            layout,
            attachment_formats: subpass.color_formats,
            subpass_samples: subpass.samples,
            ..Default::default()
        })
        .unwrap();
    let indirect = ctx
        .make_buffer(&BufferInfo {
            byte_size: 32,
            usage: BufferUsage::INDIRECT,
            initial_data: Some(bytemuck::cast_slice(&[0u32, 1, 1, 1, 0, 0, 1, 0])),
            ..Default::default()
        })
        .unwrap();
    let readback = ctx
        .make_buffer(&BufferInfo {
            byte_size: 256,
            visibility: MemoryVisibility::CpuAndGpu,
            ..Default::default()
        })
        .unwrap();
    for mode in 0..if caps.draw_indirect_count { 3 } else { 2 } {
        let draw = DrawMeshTasksIndirect {
            indirect: BufferView {
                handle: indirect,
                offset: 4,
                size: 12,
            },
            ..Default::default()
        };
        let stream = CommandStream::new()
            .begin()
            .begin_render_pass(&BeginRenderPass {
                viewport,
                render_pass: pass,
                color_attachments: [
                    Some(ImageView {
                        img: image,
                        ..Default::default()
                    }),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ],
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
            .bind_graphics_pipeline(pipeline)
            .update_viewport(&viewport);
        let stream = match mode {
            0 => stream.draw_mesh_tasks(&DrawMeshTasks {
                group_count: [1, 1, 1],
                ..Default::default()
            }),
            1 => stream.draw_mesh_tasks_indirect(&draw),
            _ => stream.draw_mesh_tasks_indirect_count(&DrawMeshTasksIndirectCount {
                draws: draw,
                count: BufferView {
                    handle: indirect,
                    offset: 24,
                    size: 4,
                },
            }),
        };
        let stream = stream
            .unbind_graphics_pipeline()
            .stop_drawing()
            .copy_image_to_buffer(&CopyImageBuffer {
                src: image,
                dst: readback,
                ..Default::default()
            })
            .end();
        let mut queue = ctx
            .begin_command_queue(QueueType::Graphics, "mesh readback", false)
            .unwrap();
        stream.append(&mut queue).unwrap();
        let fence = ctx.submit(&mut queue, &Default::default()).unwrap();
        ctx.wait(fence).unwrap();
        let pixels = ctx.map_buffer::<u8>(BufferView::new(readback)).unwrap();
        assert!(
            pixels.chunks_exact(4).all(|p| p == [0, 255, 0, 255]),
            "mode {mode}"
        );
        ctx.unmap_buffer(readback).unwrap();
        ctx.destroy_command_queue(queue);
    }
    ctx.destroy_buffer(readback);
    ctx.destroy_buffer(indirect);
    ctx.destroy_render_pass(pass);
    ctx.destroy_image(image);
}
