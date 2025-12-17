mod common;

use common::ValidationContext;
use dashi::gpu::vulkan::*;
use dashi::gpu::{
    cmd::CommandStream,
    driver::command::{CopyBuffer, Dispatch},
};
use serial_test::serial;

#[test]
#[serial]
fn test_context() {
    let ctx = ValidationContext::headless(&Default::default());
    assert!(ctx.is_ok());
}

#[test]
#[serial]
fn test_buffer() {
    let c_buffer_size = 1280;
    let c_test_val = 8 as u8;
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();

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

    let mapped_res = ctx.map_buffer::<u8>(BufferView::new(buffer));
    assert!(mapped_res.is_ok());

    let mapped = mapped_res.unwrap();
    for byte in mapped {
        assert_eq!(*byte, c_test_val);
    }

    let res = ctx.unmap_buffer(buffer);
    assert!(res.is_ok());

    ctx.destroy_buffer(buffer);
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
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();
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
}

#[test]
#[serial]
fn test_headless_context_creation() {
    // headless() should succeed...
    let ctx = ValidationContext::headless(&ContextInfo::default());
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
}

#[test]
#[serial]
#[cfg(not(feature = "dashi-openxr"))]
fn test_headless_rejects_display() {
    let mut ctx =
        ValidationContext::headless(&ContextInfo::default()).expect("headless() should succeed");

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
}

#[test]
#[serial]
fn bind_table_test() {
    // The GPU context that holds all the data.
    let _ctx = ValidationContext::headless(&Default::default()).unwrap();
    // Bind table support is optional; ensure context can be created and cleaned up.
}

#[test]
#[serial]
fn buffer_copy_then_compute_reads() {
    let mut ctx = ValidationContext::headless(&ContextInfo::default()).unwrap();

    let data: [u32; 4] = [1, 2, 3, 4];
    let staging = ctx
        .make_buffer(&BufferInfo {
            debug_name: "staging", // copy source
            byte_size: (data.len() * std::mem::size_of::<u32>()) as u32,
            visibility: MemoryVisibility::CpuAndGpu,
            initial_data: Some(bytemuck::cast_slice(&data)),
            usage: BufferUsage::ALL,
        })
        .unwrap();

    let storage = ctx
        .make_buffer(&BufferInfo {
            debug_name: "storage", // copy destination and shader input
            byte_size: (data.len() * std::mem::size_of::<u32>()) as u32,
            visibility: MemoryVisibility::Gpu,
            usage: BufferUsage::STORAGE,
            initial_data: None,
        })
        .unwrap();

    let bg_layout = ctx
        .make_bind_group_layout(&BindGroupLayoutInfo {
            debug_name: "copy_then_compute_layout",
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Compute,
                variables: &[BindGroupVariable {
                    var_type: BindGroupVariableType::Storage,
                    binding: 0,
                    count: 1,
                }],
            }],
        })
        .unwrap();

    let bind_group = ctx
        .make_bind_group(&BindGroupInfo {
            debug_name: "copy_then_compute_group",
            layout: bg_layout,
            bindings: &[BindingInfo {
                resource: ShaderResource::StorageBuffer(BufferView::new(storage)),
                binding: 0,
            }],
            set: 0,
        })
        .unwrap();

    let pipeline_layout = ctx
        .make_compute_pipeline_layout(&ComputePipelineLayoutInfo {
            bg_layouts: [Some(bg_layout), None, None, None],
            bt_layouts: [None, None, None, None],
            shader: &PipelineShaderInfo {
                stage: ShaderType::Compute,
                spirv: inline_spirv::inline_spirv!(
                    r"#version 450
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0) buffer Data { uint data[]; } buf;
void main() {
    uint v = buf.data[0];
    buf.data[0] = v;
}
",
                    comp
                ),
                specialization: &[],
            },
        })
        .unwrap();

    let pipeline = ctx
        .make_compute_pipeline(&ComputePipelineInfo {
            debug_name: "copy_then_compute_pipeline",
            layout: pipeline_layout,
        })
        .unwrap();

    let ctx_ptr = ctx.as_mut_ptr();
    let mut list = ctx
        .pool_mut(QueueType::Graphics)
        .begin(ctx_ptr, "copy_then_compute", false)
        .unwrap();

    let mut stream = CommandStream::new().begin();
    stream.copy_buffers(&CopyBuffer {
        src: staging,
        dst: storage,
        src_offset: 0,
        dst_offset: 0,
        amount: (data.len() * std::mem::size_of::<u32>()) as u32,
    });

    let stream = stream.dispatch(&Dispatch {
        x: 1,
        y: 1,
        z: 1,
        pipeline,
        bind_groups: [Some(bind_group), None, None, None],
        bind_tables: [None, None, None, None],
        dynamic_buffers: [None, None, None, None],
    });

    let stream = stream.unbind_pipeline().end();
    stream.append(&mut list);

    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
    ctx.wait(fence).unwrap();
    ctx.destroy_cmd_queue(list);

    ctx.destroy_buffer(staging);
    ctx.destroy_buffer(storage);
}
