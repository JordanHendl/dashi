use std::path::Path;
use dashi::*;
use image_utils::{load_png, compare_rgba};

#[test]
fn framebuffer_compare() {
    let ref_path = Path::new("tests/reference/red.png");
    if !ref_path.exists() {
        eprintln!("reference image missing, skipping test");
        return;
    }
    let (expected, width, height) = load_png(ref_path).expect("load reference");

    let mut ctx = Context::headless(&Default::default()).unwrap();
    let image = ctx
        .make_image(&ImageInfo {
            debug_name: "ref_image",
            dim: [width, height, 1],
            format: Format::RGBA8,
            mip_levels: 1,
            initial_data: Some(&expected),
            ..Default::default()
        })
        .unwrap();

    let view = ImageView { img: image, ..Default::default() };

    let buffer = ctx
        .make_buffer(&BufferInfo {
            debug_name: "readback",
            byte_size: (width * height * 4) as u32,
            visibility: MemoryVisibility::CpuAndGpu,
            ..Default::default()
        })
        .unwrap();

    let mut list = ctx
        .begin_command_list(&CommandListInfo { debug_name: "copy", ..Default::default() })
        .unwrap();
    list.copy_image_to_buffer(ImageBufferCopy { src: view, dst: buffer, dst_offset: 0 });
    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
    ctx.wait(fence).unwrap();

    let actual = ctx.map_buffer::<u8>(buffer).unwrap().to_vec();
    ctx.unmap_buffer(buffer).unwrap();

    assert!(compare_rgba(&actual, &expected, width, height, 0));

    ctx.destroy_cmd_list(list);
    ctx.destroy_buffer(buffer);
    ctx.destroy_image(image);
    ctx.destroy();
}
