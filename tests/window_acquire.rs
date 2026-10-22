#![cfg(all(feature = "vulkan", feature = "dashi-winit", target_os = "windows"))]

mod common;
use common::ValidationContext;
use dashi::{
    execution::CommandRing, CommandStream, ContextInfo, DisplayBuilder, QueueType, SubmitInfo,
};

#[test]
#[ignore = "requires a desktop window"]
#[serial_test::serial]
fn swapchain_acquisition_reuses_completed_fences() {
    let mut context = ValidationContext::windowed(&ContextInfo::default()).unwrap();
    let mut display = DisplayBuilder::new()
        .title("Swapchain acquisition fence regression")
        .size(320, 240)
        .build(&mut context)
        .unwrap();
    let ready = context.make_semaphores(1).unwrap()[0];
    let mut ring =
        CommandRing::new(&mut context, "acquisition fence", 2, QueueType::Graphics).unwrap();
    for _ in 0..12 {
        context.prepare_display(&mut display).unwrap();
        let (image, acquired, _, _) = context.acquire_new_image(&mut display).unwrap();
        ring.record(|queue| {
            CommandStream::new()
                .begin()
                .prepare_for_presentation(image.img)
                .end()
                .append(queue)
                .unwrap();
        })
        .unwrap();
        ring.submit_with_fence(&SubmitInfo {
            wait_sems: &[acquired],
            signal_sems: &[ready],
        })
        .unwrap();
        context.present_display(&display, &[ready]).unwrap();
        // Isolate acquire-fence reuse from presentation-semaphore recycling.
        context.sync_current_device();
    }
    ring.wait_all().unwrap();
    drop(ring);
    context.destroy_semaphore(ready);
    context.destroy_display(display);
}
