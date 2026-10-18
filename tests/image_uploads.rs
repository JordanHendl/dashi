#![cfg(feature = "vulkan")]
mod common;
use common::ValidationContext;
use dashi::driver::command::CopyImageBuffer;
use dashi::*;

fn read(ctx: &mut Context, image: Handle<Image>, mip: u32, layers: u32, size: u32) -> Vec<u8> {
    let buffer = ctx
        .make_buffer(&BufferInfo {
            byte_size: size,
            visibility: MemoryVisibility::CpuAndGpu,
            ..Default::default()
        })
        .unwrap();
    let mut commands = ctx
        .begin_command_queue(QueueType::Graphics, "upload readback", false)
        .unwrap();
    CommandStream::new()
        .begin()
        .copy_image_to_buffer(&CopyImageBuffer {
            src: image,
            dst: buffer,
            range: SubresourceRange::new(mip, 1, 0, layers),
            dst_offset: 0,
        })
        .end()
        .append(&mut commands)
        .unwrap();
    let fence = ctx
        .submit_command_queue(&mut commands, &Default::default())
        .unwrap();
    ctx.wait_fence(fence).unwrap();
    ctx.invalidate_buffer(BufferView::new(buffer)).unwrap();
    let result = ctx
        .map_buffer::<u8>(BufferView::new(buffer))
        .unwrap()
        .to_vec();
    ctx.unmap_buffer(buffer).unwrap();
    ctx.destroy_command_queue(commands);
    ctx.destroy_buffer(buffer);
    result
}

#[test]
#[serial_test::serial]
fn one_batch_uploads_images_layers_volumes_and_generated_mips() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();
    let mut batch = UploadBatch::new();
    let mut images = Vec::new();
    for (dim, layers, cube) in [
        ([4, 4, 1], 1, false),
        ([4, 4, 1], 2, false),
        ([4, 4, 1], 6, true),
        ([4, 4, 4], 1, false),
    ] {
        let image = ctx
            .make_image(&ImageInfo {
                dim,
                layers,
                cube_compatible: cube,
                format: Format::RGBA8Unorm,
                mip_levels: 3,
                ..Default::default()
            })
            .unwrap();
        let bytes = [255, 0, 0, 255].repeat((dim[0] * dim[1] * dim[2] * layers) as usize);
        ctx.stage_image_initial_data(&mut batch, image, &bytes)
            .unwrap();
        images.push((image, layers));
    }
    assert!(!batch.is_empty());
    let mut ticket = ctx.submit_upload_batch(batch).unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !ctx.poll_upload(&mut ticket).unwrap() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert!(ticket.completed());
    assert!(ctx.poll_upload(&mut ticket).unwrap());
    for (image, layers) in images {
        assert_eq!(
            read(&mut ctx, image, 2, layers, layers * 4),
            [255, 0, 0, 255].repeat(layers as usize)
        );
        ctx.destroy_image(image);
    }
}

#[test]
#[serial_test::serial]
fn region_uploads_span_batches_and_reject_invalid_ranges_before_staging() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();
    let image = ctx
        .make_image(&ImageInfo {
            dim: [2, 2, 1],
            format: Format::RGBA8Unorm,
            ..Default::default()
        })
        .unwrap();
    let region = ImageUploadRegion {
        image,
        range: SubresourceRange::new(0, 1, 0, 1),
        offset: [0; 3],
        extent: [2, 1, 1],
    };
    let mut invalid = UploadBatch::new();
    assert!(ctx
        .stage_image_upload(&mut invalid, region, &[0; 4])
        .is_err());
    assert!(ctx
        .stage_image_upload(
            &mut invalid,
            ImageUploadRegion {
                offset: [2, 0, 0],
                ..region
            },
            &[0; 8]
        )
        .is_err());
    assert!(invalid.is_empty());
    ctx.discard_upload_batch(invalid).unwrap();
    let mut first = UploadBatch::new();
    ctx.stage_image_upload(&mut first, region, &[255, 0, 0, 255, 255, 0, 0, 255])
        .unwrap();
    let mut first = ctx.submit_upload_batch(first).unwrap();
    let mut second = UploadBatch::new();
    ctx.stage_image_upload(
        &mut second,
        ImageUploadRegion {
            offset: [0, 1, 0],
            ..region
        },
        &[0, 255, 0, 255, 0, 255, 0, 255],
    )
    .unwrap();
    ctx.finish_image_upload(&mut second, image, 1).unwrap();
    let mut second = ctx.submit_upload_batch(second).unwrap();
    ctx.wait_upload(&mut second).unwrap();
    assert!(ctx.poll_upload(&mut first).unwrap());
    assert_eq!(
        read(&mut ctx, image, 0, 1, 16),
        [255, 0, 0, 255, 255, 0, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255]
    );
    ctx.destroy_image(image);
}

#[test]
#[serial_test::serial]
fn supplied_mips_are_preserved_and_unsubmitted_staging_can_be_discarded() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();
    let image = ctx
        .make_image(&ImageInfo {
            dim: [2, 2, 1],
            mip_levels: 2,
            format: Format::RGBA8Unorm,
            ..Default::default()
        })
        .unwrap();
    let mut bytes = [255, 0, 0, 255].repeat(4);
    bytes.extend([0, 0, 255, 255]);
    let mut batch = UploadBatch::new();
    ctx.stage_image_initial_data(&mut batch, image, &bytes)
        .unwrap();
    assert_eq!(batch.bytes(), 20);
    ctx.discard_upload_batch(batch).unwrap();
    let mut batch = UploadBatch::new();
    ctx.stage_image_initial_data(&mut batch, image, &bytes)
        .unwrap();
    let mut ticket = ctx.submit_upload_batch(batch).unwrap();
    ctx.wait_upload(&mut ticket).unwrap();
    assert_eq!(read(&mut ctx, image, 1, 1, 4), [0, 0, 255, 255]);
    ctx.destroy_image(image);
}
