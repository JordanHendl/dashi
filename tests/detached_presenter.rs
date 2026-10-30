#![cfg(all(feature = "vulkan", feature = "dashi-winit", target_os = "windows"))]
mod common;
use common::ValidationContext;
use dashi::*;
use std::{sync::mpsc, time::Duration};

#[test]
#[ignore = "requires a desktop window"]
#[serial_test::serial]
fn retained_frames_present_on_worker_while_producer_submits() {
    let mut ctx = ValidationContext::windowed(&ContextInfo::default()).unwrap();
    let mut window = DisplayBuilder::new()
        .title("Genlocked presentation regression")
        .size(320, 240)
        .build(&mut ctx)
        .unwrap();
    ctx.prepare_display(&mut window).unwrap();
    let format = ctx.display_image_format(&window).unwrap();
    let data = vec![127u8; 320 * 240 * 4];
    let source = ctx
        .make_image(&ImageInfo {
            dim: [320, 240, 1],
            format,
            initial_data: Some(&data),
            ..Default::default()
        })
        .unwrap();
    let held: Vec<_> = (0..3)
        .map(|_| {
            ctx.make_image(&ImageInfo {
                dim: [320, 240, 1],
                format,
                ..Default::default()
            })
            .unwrap()
        })
        .collect();
    let ready = ctx.make_semaphores(3).unwrap();
    let mut ring =
        execution::CommandRing::new(&mut ctx, "Retained frames", 3, QueueType::Graphics).unwrap();
    let mut presenter = unsafe { ctx.detached_presenter(&window, &held).unwrap() };
    let (send, receive) = mpsc::channel::<(usize, PresentationSubmission)>();
    let (done, retired) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        for (slot, submission) in receive {
            while !presenter.stage(slot, submission).unwrap() {
                std::thread::sleep(Duration::from_millis(1));
            }
            std::thread::sleep(Duration::from_millis(8));
            presenter.present().unwrap();
            presenter.finish_copy().unwrap();
            done.send(slot).unwrap();
        }
    });
    for n in 0..9 {
        let slot = n % 3;
        if n >= 3 {
            assert_eq!(retired.recv_timeout(Duration::from_secs(5)).unwrap(), slot);
        }
        ring.record(|queue| {
            CommandStream::new()
                .begin()
                .blit_images(&driver::command::BlitImage {
                    src: source,
                    dst: held[slot],
                    filter: Filter::Nearest,
                    ..Default::default()
                })
                .end()
                .append(queue)
                .unwrap();
        })
        .unwrap();
        let fence = ring
            .submit_with_fence(&SubmitInfo {
                signal_sems: &[ready[slot]],
                ..Default::default()
            })
            .unwrap();
        send.send((slot, unsafe {
            ctx.presentation_submission(fence, ready[slot]).unwrap()
        }))
        .unwrap();
    }
    for _ in 0..3 {
        retired.recv_timeout(Duration::from_secs(5)).unwrap();
    }
    drop(send);
    worker.join().unwrap();
    ring.wait_all().unwrap();
    drop(ring);
    for image in held {
        ctx.destroy_image(image);
    }
    for semaphore in ready {
        ctx.destroy_semaphore(semaphore);
    }
    ctx.destroy_image(source);
    ctx.destroy_display(window);
}
