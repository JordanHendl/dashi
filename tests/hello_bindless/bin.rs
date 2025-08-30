use dashi::builders::{BindTableBuilder, BindTableLayoutBuilder};
use dashi::*;

#[cfg(feature = "dashi-tests")]
fn main() {
    // Create a headless context.
    let mut ctx = gpu::Context::headless(&ContextInfo::default()).unwrap();

    // One dynamic uniform at binding 0.
    let shader_info = ShaderInfo {
        shader_type: ShaderType::Compute,
        variables: &[BindGroupVariable {
            var_type: BindGroupVariableType::DynamicUniform,
            binding: 0,
            count: 1,
        }],
    };

    // Layouts for bind table and pipeline.
    let bt_layout = BindTableLayoutBuilder::new("bindless_layout")
        .shader(shader_info)
        .build(&mut ctx)
        .unwrap();
    let bg_layout = ctx
        .make_bind_group_layout(&BindGroupLayoutInfo {
            debug_name: "bindless_bg_layout",
            shaders: &[shader_info],
        })
        .unwrap();

    // Compute pipeline using the layout above.
    let pipeline_layout = ctx
        .make_compute_pipeline_layout(&ComputePipelineLayoutInfo {
            bg_layouts: [Some(bg_layout), None, None, None],
            shader: &PipelineShaderInfo {
                stage: ShaderType::Compute,
                spirv: inline_spirv::inline_spirv!(
                    r#"
                    #version 450
                    layout(local_size_x = 1) in;
                    layout(binding = 0) uniform Data { float value; } data;
                    void main() {}
                    "#,
                    comp
                ),
                specialization: &[],
            },
        })
        .unwrap();
    let pipeline = ctx
        .make_compute_pipeline(&ComputePipelineInfo {
            debug_name: "noop",
            layout: pipeline_layout,
        })
        .unwrap();

    // Build table with a dynamic allocator binding.
    let mut allocator = ctx.make_dynamic_allocator(&Default::default()).unwrap();
    let table = BindTableBuilder::new("bindless_table")
        .layout(bt_layout)
        .binding(
            0,
            &[IndexedResource {
                slot: 0,
                resource: ShaderResource::Dynamic(&allocator),
            }],
        )
        .build(&mut ctx)
        .unwrap();

    // Update the table after creation.
    ctx.update_bind_table(&BindTableUpdateInfo {
        table,
        bindings: &[IndexedBindingInfo {
            binding: 0,
            resources: &[IndexedResource {
                slot: 0,
                resource: ShaderResource::Dynamic(&allocator),
            }],
        }],
    })
    .unwrap();

    // Bind the table in a command list and dispatch.
    let mut list = ctx.begin_command_list(&Default::default()).unwrap();
    let buf = allocator.bump().unwrap();
    list.dispatch_compute(Dispatch {
        compute: pipeline,
        workgroup_size: [1, 1, 1],
        bindings: Bindings {
            bind_tables: [Some(table), None, None, None],
            dynamic_buffers: [Some(buf), None, None, None],
            ..Default::default()
        },
        ..Default::default()
    });
    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
    ctx.wait(fence).unwrap();

    ctx.destroy_dynamic_allocator(allocator);
    ctx.destroy();
}

#[cfg(not(feature = "dashi-tests"))]
fn main() {}
