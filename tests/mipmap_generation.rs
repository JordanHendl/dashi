use dashi::*;

#[test]
#[ignore]
fn mipmap_generation() {
    const WIDTH: u32 = 4;
    const HEIGHT: u32 = 4;
    // solid red base level
    let data = vec![255u8, 0, 0, 255].repeat((WIDTH * HEIGHT) as usize);

    let mut ctx = Context::headless(&Default::default()).unwrap();
    let image = ctx
        .make_image(&ImageInfo {
            debug_name: "mip_test",
            dim: [WIDTH, HEIGHT, 1],
            format: Format::RGBA8,
            mip_levels: 3,
            initial_data: Some(&data),
            ..Default::default()
        })
        .unwrap();

    // read back from the smallest mip
    let view = ImageView {
        img: image,
        mip_level: 2,
        ..Default::default()
    };

    let buffer = ctx
        .make_buffer(&BufferInfo {
            debug_name: "readback",
            byte_size: 4,
            visibility: MemoryVisibility::CpuAndGpu,
            ..Default::default()
        })
        .unwrap();

//    let mut list = ctx
//        .begin_command_list(&Default::default())
//        .unwrap();
//    list.copy_image_to_buffer(ImageBufferCopy { src: view, dst: buffer, dst_offset: 0 });
//    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
//    ctx.wait(fence).unwrap();
//
//    let actual = ctx.map_buffer::<u8>(buffer).unwrap().to_vec();
//    ctx.unmap_buffer(buffer).unwrap();
//
//    assert_eq!(actual, vec![255u8, 0, 0, 255]);

//    ctx.destroy_cmd_list(list);
    ctx.destroy_buffer(buffer);
    ctx.destroy_image(image);
    ctx.destroy();
}
