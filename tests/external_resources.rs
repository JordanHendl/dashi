#![cfg(all(feature = "vulkan", any(windows, unix)))]
mod common;
use common::ValidationContext;
use dashi::driver::command::{CopyBuffer, CopyBufferImage, CopyImageBuffer};
use dashi::*;

#[test]
#[serial_test::serial]
fn opaque_memory_roundtrip_reuses_image_buffer_and_binary_semaphores() {
    let selector = DeviceSelector::new().unwrap();
    let selected = selector.select(DeviceFilter::default()).unwrap();
    let identity = selected.info.identity;
    let mut producer = ValidationContext::headless(&ContextInfo {
        device: selected,
        ..Default::default()
    })
    .unwrap();
    let mut consumer = ValidationContext::headless(&ContextInfo {
        device: selector
            .select(DeviceFilter::default().add_required_identity(identity))
            .unwrap(),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        producer.device_identity().unwrap(),
        consumer.device_identity().unwrap()
    );
    let capabilities = producer.external_resource_capabilities().unwrap();
    assert!(
        capabilities.memory && capabilities.binary_semaphore,
        "GPU sharing test requires opaque external memory and semaphore support"
    );
    let image_info = ImageInfo {
        dim: [8, 8, 1],
        format: Format::RGBA8,
        storage: false,
        ..Default::default()
    };
    let buffer_info = BufferInfo {
        byte_size: 256,
        visibility: MemoryVisibility::Gpu,
        ..Default::default()
    };
    let image = producer.make_external_image(&image_info).unwrap();
    let buffer = producer.make_external_buffer(&buffer_info).unwrap();
    let ready = producer.make_external_semaphore().unwrap();
    let released = producer.make_external_semaphore().unwrap();
    let exported = producer.export_image(image).unwrap();
    let cloned = exported.handle.try_clone().unwrap();
    drop(cloned);
    let imported_image = unsafe { consumer.import_image(&image_info, exported) }.unwrap();
    let imported_buffer =
        unsafe { consumer.import_buffer(&buffer_info, producer.export_buffer(buffer).unwrap()) }
            .unwrap();
    let imported_ready =
        unsafe { consumer.import_semaphore(producer.export_semaphore(ready).unwrap()) }.unwrap();
    let imported_released =
        unsafe { consumer.import_semaphore(producer.export_semaphore(released).unwrap()) }.unwrap();
    let upload = producer
        .make_buffer(&BufferInfo {
            byte_size: 256,
            ..Default::default()
        })
        .unwrap();
    let image_readback = consumer
        .make_buffer(&BufferInfo {
            byte_size: 256,
            ..Default::default()
        })
        .unwrap();
    let buffer_readback = consumer
        .make_buffer(&BufferInfo {
            byte_size: 256,
            ..Default::default()
        })
        .unwrap();
    assert!(producer.export_buffer(upload).is_err());
    assert!(consumer.export_image(imported_image).is_err());
    assert!(producer.map_buffer::<u8>(BufferView::new(buffer)).is_err());
    for frame in 0..12u8 {
        let expected: Vec<u8> = (0..256).map(|i| (i as u8).wrapping_add(frame)).collect();
        producer
            .map_buffer_mut::<u8>(BufferView::new(upload))
            .unwrap()
            .copy_from_slice(&expected);
        producer.flush_buffer(BufferView::new(upload)).unwrap();
        producer.unmap_buffer(upload).unwrap();
        let mut write = producer
            .begin_command_queue(QueueType::Graphics, "external write", false)
            .unwrap();
        if frame != 0 {
            producer.acquire_external_image(&mut write, image).unwrap();
            producer
                .acquire_external_buffer(&mut write, buffer)
                .unwrap();
        }
        CommandStream::new()
            .begin()
            .copy_buffers(&CopyBuffer {
                src: upload,
                dst: buffer,
                amount: 256,
                ..Default::default()
            })
            .copy_buffer_to_image(&CopyBufferImage {
                src: upload,
                dst: image,
                range: SubresourceRange::default(),
                amount: 256,
                ..Default::default()
            })
            .end()
            .append(&mut write)
            .unwrap();
        producer.release_external_image(&mut write, image).unwrap();
        producer
            .release_external_buffer(&mut write, buffer)
            .unwrap();
        assert!(producer.release_external_image(&mut write, image).is_err());
        let wait = if frame == 0 { vec![] } else { vec![released] };
        let written = producer
            .submit_command_queue(
                &mut write,
                &SubmitInfo {
                    wait_sems: &wait,
                    signal_sems: &[ready],
                },
            )
            .unwrap();
        let mut read = consumer
            .begin_command_queue(QueueType::Graphics, "external read", false)
            .unwrap();
        consumer
            .acquire_external_image(&mut read, imported_image)
            .unwrap();
        consumer
            .acquire_external_buffer(&mut read, imported_buffer)
            .unwrap();
        CommandStream::new()
            .begin()
            .copy_buffers(&CopyBuffer {
                src: imported_buffer,
                dst: buffer_readback,
                amount: 256,
                ..Default::default()
            })
            .copy_image_to_buffer(&CopyImageBuffer {
                src: imported_image,
                dst: image_readback,
                range: SubresourceRange::default(),
                ..Default::default()
            })
            .end()
            .append(&mut read)
            .unwrap();
        consumer
            .release_external_image(&mut read, imported_image)
            .unwrap();
        consumer
            .release_external_buffer(&mut read, imported_buffer)
            .unwrap();
        let copied = consumer
            .submit_command_queue(
                &mut read,
                &SubmitInfo {
                    wait_sems: &[imported_ready],
                    signal_sems: &[imported_released],
                },
            )
            .unwrap();
        consumer.wait_fence(copied).unwrap();
        producer.wait_fence(written).unwrap();
        for readback in [image_readback, buffer_readback] {
            assert_eq!(
                consumer
                    .map_buffer::<u8>(BufferView::new(readback))
                    .unwrap(),
                expected
            );
            consumer.unmap_buffer(readback).unwrap();
        }
        producer.destroy_cmd_queue(write);
        consumer.destroy_cmd_queue(read);
    }
    consumer.destroy_image(imported_image);
    consumer.destroy_buffer(imported_buffer);
    consumer.destroy_buffer(image_readback);
    consumer.destroy_buffer(buffer_readback);
    consumer.destroy_semaphore(imported_ready);
    consumer.destroy_semaphore(imported_released);
    producer.destroy_image(image);
    producer.destroy_buffer(buffer);
    producer.destroy_buffer(upload);
    producer.destroy_semaphore(ready);
    producer.destroy_semaphore(released);
}

#[test]
#[serial_test::serial]
fn external_import_rejects_mismatched_metadata_without_leaking_resources() {
    let mut ctx = ValidationContext::headless(&ContextInfo::default()).unwrap();
    let info = ImageInfo {
        dim: [8, 8, 1],
        format: Format::RGBA8,
        ..Default::default()
    };
    let image = ctx.make_external_image(&info).unwrap();
    let mut exported = ctx.export_image(image).unwrap();
    exported.descriptor.memory.device.device_uuid[0] ^= 255;
    assert!(unsafe { ctx.import_image(&info, exported) }.is_err());
    let wrong_size = ImageInfo {
        dim: [16, 8, 1],
        ..info
    };
    let exported = ctx.export_image(image).unwrap();
    assert!(unsafe { ctx.import_image(&wrong_size, exported) }.is_err());
    assert!(ctx
        .make_external_image(&ImageInfo {
            mip_levels: 2,
            ..info
        })
        .is_err());
    ctx.destroy_image(image);
}
