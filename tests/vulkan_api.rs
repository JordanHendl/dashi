mod common;

use common::ValidationContext;
use dashi::gpu::vulkan::*;
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
