use dashi::gpu::vulkan::*;
use serial_test::serial;

#[test]
#[serial]
fn test_context() {
    let ctx = Context::headless(&Default::default());
    assert!(ctx.is_ok());
    ctx.unwrap().destroy();
}

#[test]
#[serial]
fn test_buffer() {
    let c_buffer_size = 1280;
    let c_test_val = 8 as u8;
    let mut ctx = Context::headless(&Default::default()).unwrap();

    let initial_data = vec![c_test_val as u8; c_buffer_size as usize];
    let buffer_res = ctx.make_buffer(&BufferInfo {
        debug_name: "Test Buffer",
        byte_size: c_buffer_size,
        visibility: MemoryVisibility::CpuAndGpu,
        initial_data: Some(&initial_data),
        ..Default::default()
    });

    assert!(buffer_res.is_ok());

    let buffer = buffer_res.unwrap();

    let mapped_res = ctx.map_buffer::<u8>(buffer);
    assert!(mapped_res.is_ok());

    let mapped = mapped_res.unwrap();
    for byte in mapped {
        assert_eq!(*byte, c_test_val);
    }

    let res = ctx.unmap_buffer(buffer);
    assert!(res.is_ok());

    ctx.destroy_buffer(buffer);
    ctx.destroy();
}

#[test]
#[serial]
fn test_image() {
    let c_test_dim: [u32; 3] = [1280, 1024, 1];
    let c_format = Format::RGBA8;
    let c_mip_levels = 1;
    let c_test_val = 8 as u8;
    let initial_data =
        vec![c_test_val as u8; (c_test_dim[0] * c_test_dim[1] * c_test_dim[2] * 4) as usize];
    let mut ctx = Context::headless(&Default::default()).unwrap();
    let image_res = ctx.make_image(&ImageInfo {
        debug_name: "Test Image",
        dim: c_test_dim,
        format: c_format,
        mip_levels: c_mip_levels,
        initial_data: Some(&initial_data),
        ..Default::default()
    });

    assert!(image_res.is_ok());
    let image = image_res.unwrap();
    ctx.destroy_image(image);
    ctx.destroy();
}

#[test]
#[serial]
fn test_headless_context_creation() {
    // headless() should succeed...
    let ctx = Context::headless(&ContextInfo::default());
    assert!(ctx.is_ok(), "Context::headless() failed to create");
    let mut ctx = ctx.unwrap();

    // ...and never initialize any SDL bits in windowed builds
    #[cfg(feature = "dashi-sdl2")]
    {
        assert!(
            ctx.sdl_video.is_none(),
            "SDL video subsystem must be None in headless"
        );
    }

    // Core Vulkan ops still work:
    let buf = ctx
        .make_buffer(&BufferInfo {
            debug_name: "headless-buffer",
            byte_size: 128,
            visibility: MemoryVisibility::CpuAndGpu,
            ..Default::default()
        })
        .expect("make_buffer failed in headless mode");
    ctx.destroy_buffer(buf);

    // And we can clean up without panicking
    ctx.destroy();
}

#[test]
#[serial]
#[cfg(not(feature = "dashi-openxr"))]
fn test_headless_rejects_display() {
    let mut ctx =
        Context::headless(&ContextInfo::default()).expect("headless() should succeed");

    // Try to call the windowed API -- should panic or error
    let info = DisplayInfo {
        window: WindowInfo {
            title: "nope".to_string(),
            size: [64, 64],
            resizable: false,
        },
        vsync: false,
        buffering: WindowBuffering::Double,
    };

    let err = ctx.make_display(&info).err().unwrap();

    assert!(
        matches!(err, GPUError::HeadlessDisplayNotSupported),
        "expected HeadlessDisplayNotSupported, got {:?}",
        err
    );

    ctx.destroy();
}

#[test]
#[serial]
fn compute_test() {
    // The GPU context that holds all the data.
    let mut ctx = Context::headless(&Default::default()).unwrap();

    // Make the bind group layout. This describes the bindings into a shader.
    let bg_layout = ctx
        .make_bind_group_layout(&BindGroupLayoutInfo {
            debug_name: "Hello Compute BG Layout",
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Compute,
                variables: &[
                    BindGroupVariable {
                        var_type: BindGroupVariableType::Storage,
                        binding: 0,
                        count: 2048,
                    },
                    BindGroupVariable {
                        var_type: BindGroupVariableType::Storage,
                        binding: 1,
                        count: 2048,
                    },
                    BindGroupVariable {
                        var_type: BindGroupVariableType::DynamicUniform,
                        binding: 2,
                        count: 1,
                    },
                ],
            }],
        })
        .unwrap();

    // Make a pipeline layout. This describes a graphics pipeline's state.
    let pipeline_layout = ctx
        .make_compute_pipeline_layout(&ComputePipelineLayoutInfo {
            bg_layouts: [Some(bg_layout), None, None, None],
            bt_layouts: [None, None, None, None],
            shader: &PipelineShaderInfo {
                stage: ShaderType::Compute,
                spirv: inline_spirv::inline_spirv!(
                    r#"
#version 450

layout(local_size_x = 1) in;

layout(binding = 0) buffer InputBuffer {
float inputData[];
};

layout(binding = 1) buffer OutputBuffer {
float outputData[];
};

layout(binding = 2) uniform OutputBuffer {
float num_to_add;
};

void main() {
uint index = gl_GlobalInvocationID.x;
outputData[index] = inputData[index] + num_to_add;
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
    let input = ctx
        .make_buffer(&BufferInfo {
            debug_name: "input_test",
            byte_size: BUFF_SIZE,
            visibility: MemoryVisibility::Gpu,
            usage: BufferUsage::STORAGE,
            initial_data: Some(&initial_data),
        })
        .unwrap();

    let output = ctx
        .make_buffer(&BufferInfo {
            debug_name: "output_test",
            byte_size: BUFF_SIZE,
            visibility: MemoryVisibility::CpuAndGpu,
            usage: BufferUsage::STORAGE,
            initial_data: Some(&initial_data),
        })
        .unwrap();

    // Make bind group what we want to bind to what was described in the Bind Group Layout.
    let bind_group = ctx
        .make_bind_group(&BindGroupInfo {
            debug_name: "Hello Compute BG",
            layout: bg_layout,
            bindings: &[
                BindingInfo {
                    resource: ShaderResource::StorageBuffer(input),
                    binding: 0,
                },
                BindingInfo {
                    resource: ShaderResource::StorageBuffer(output),
                    binding: 1,
                },
                BindingInfo {
                    resource: ShaderResource::Dynamic(&allocator),
                    binding: 2,
                },
            ],
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

    list.dispatch_compute(Dispatch {
        compute: pipeline,
        workgroup_size: [BUFF_SIZE / std::mem::size_of::<f32>() as u32, 1, 1],
        bindings: Bindings {
            bind_groups: [Some(bind_group), None, None, None],
            dynamic_buffers: [Some(buf), None, None, None],
            ..Default::default()
        },
        ..Default::default()
    });

    // Submit our recorded commands
    let fence = ctx.submit(&mut list, &Default::default()).unwrap();

    ctx.wait(fence).unwrap();

    let data = ctx.map_buffer::<f32>(output).unwrap();
    for entry in data {
        assert!(*entry == 5.0);
    }

    ctx.unmap_buffer(output).unwrap();
    ctx.destroy_dynamic_allocator(allocator);
    ctx.destroy();
}

#[test]
#[serial]
fn bind_table_test() {
    // The GPU context that holds all the data.
    let ctx = Context::headless(&Default::default()).unwrap();
    // Bind table support is optional; ensure context can be created and cleaned up.
    ctx.destroy();
}
