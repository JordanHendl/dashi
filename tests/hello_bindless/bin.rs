use dashi::*;

#[cfg(feature = "dashi-tests")]
fn main() {
    // The GPU context that holds all the data.
    let mut ctx = gpu::Context::new(&Default::default()).unwrap();

    // Make the bind group layout. This describes the bindless bindings.
    let bg_layout = ctx
        .make_bindless_bind_group_layout(&BindGroupLayoutInfo {
            debug_name: "Hello Compute Bindless",
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Compute,
                variables: &[
                    BindGroupVariable {
                        var_type: BindGroupVariableType::Storage,
                        binding: 0,
                    },
                    BindGroupVariable {
                        var_type: BindGroupVariableType::Storage,
                        binding: 1,
                    },
                ],
            }],
        })
        .unwrap();

    // Make the bind group layout. This describes the dynamic bindings
    let dyn_bg_layout = ctx
        .make_bind_group_layout(&BindGroupLayoutInfo {
            debug_name: "Hello Compute Dynamic Binding",
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Compute,
                variables: &[BindGroupVariable {
                    var_type: BindGroupVariableType::DynamicUniform,
                    binding: 0,
                }],
            }],
        })
        .unwrap();

    // Make a pipeline layout. This describes a graphics pipeline's state.
    let pipeline_layout = ctx
        .make_compute_pipeline_layout(&ComputePipelineLayoutInfo {
            bg_layouts: [Some(bg_layout), Some(dyn_bg_layout), None, None],
            shader: &PipelineShaderInfo {
                stage: ShaderType::Compute,
                spirv: inline_spirv::inline_spirv!(
                    r#"
#version 450
#extension GL_EXT_nonuniform_qualifier : enable 

layout(local_size_x = 1) in;

layout(set = 0,binding = 0) buffer InputBuffer {
    float data;
} c_input[];

layout(set = 0, binding = 1) buffer OutputBuffer {
    float data;
} c_output[];


layout(set = 1, binding = 0) uniform OutputBuffer {
    float num_to_add;
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    c_output[index].data = c_input[index].data + num_to_add;
}
"#,
                    comp
                ),
                specialization: &[],
            },
        })
        .expect("Unable to create Compute Pipeline Layout!");

    // Make a compute pipeline. This describes a compute pass.
    let pipeline = ctx
        .make_compute_pipeline(&ComputePipelineInfo {
            debug_name: "Compute",
            layout: pipeline_layout,
        })
        .unwrap();

    // Make dynamic allocator to use for dynamic buffers.
    let mut allocator = ctx.make_dynamic_allocator(&Default::default()).unwrap();
    const BUFF_SIZE: u32 = 2048 * std::mem::size_of::<f32>() as u32;
    let initial_data = vec![0; BUFF_SIZE as usize];

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for i in 0..256 {
        inputs.push(ShaderResource::StorageBuffer(
            ctx.make_buffer(&BufferInfo {
                debug_name: "input_test",
                byte_size: BUFF_SIZE,
                visibility: MemoryVisibility::Gpu,
                usage: BufferUsage::STORAGE,
                initial_data: Some(&initial_data),
            })
            .unwrap(),
        ));

        outputs.push(ShaderResource::StorageBuffer(
            ctx.make_buffer(&BufferInfo {
                debug_name: "output_test",
                byte_size: BUFF_SIZE,
                visibility: MemoryVisibility::CpuAndGpu,
                usage: BufferUsage::STORAGE,
                initial_data: Some(&initial_data),
            })
            .unwrap(),
        ));
    }

    // Make bind group what we want to bind to what was described in the Bind Group Layout.
    let bind_group = ctx
        .make_indexed_bind_group(&IndexedBindGroupInfo {
            debug_name: "Hello Compute Bindless",
            layout: bg_layout,
            bindings: &[
                IndexedBindingInfo {
                    resources: &inputs,
                    binding: 0,
                },
                IndexedBindingInfo {
                    resources: &outputs,
                    binding: 1,
                },
            ],
            ..Default::default()
        })
        .unwrap();

    let dynamic_bind_group = ctx
        .make_bind_group(&BindGroupInfo {
            debug_name: "Hello Compute Dynamic",
            layout: dyn_bg_layout,
            set: 1,
            bindings: &[BindingInfo {
                resource: ShaderResource::Dynamic(&allocator),
                binding: 0,
            }],
            ..Default::default()
        })
        .unwrap();

    // Reset the allocator
    allocator.reset();

    // Begin recording commands
    let mut list = ctx.begin_command_list(&Default::default()).unwrap();

    // Bump alloc some data to write the triangle position to.
    let mut buf = allocator.bump().unwrap();
    buf.slice::<f32>()[0] = 5.0;
    list.dispatch(&Dispatch {
        compute: pipeline,
        workgroup_size: [BUFF_SIZE / std::mem::size_of::<f32>() as u32, 1, 1],
        bind_groups: [Some(bind_group), Some(dynamic_bind_group), None, None],
        dynamic_buffers: [None, Some(buf), None, None],
        ..Default::default()
    });

    // Submit our recorded commands
    let fence = ctx.submit(&mut list, &Default::default()).unwrap();

    ctx.wait(fence).unwrap();

    //    let data = ctx.map_buffer::<f32>(output).unwrap();
    //    for entry in data {
    //        assert!(*entry == 5.0);
    //    }
    //
//    ctx.unmap_buffer(output).unwrap();
    ctx.clean_up();
}

#[cfg(not(feature = "dashi-tests"))]
fn main() { //none
}
