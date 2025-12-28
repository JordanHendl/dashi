mod common;

use common::ValidationContext;
use dashi::*;

#[test]
fn gpu_timer() {
    let mut ctx = match ValidationContext::headless(&ContextInfo::default()) {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!(
                "Skipping gpu_timer test: Vulkan initialization unavailable: {:?}",
                err
            );
            return;
        }
    };
    // GPU timers must be initialized before use.
    ctx.init_gpu_timers(1).unwrap();

    let mut list = ctx
        .begin_command_queue(QueueType::Graphics, "timer", false)
        .unwrap();
    // Begin and end must bracket commands on the same list.
    ctx.gpu_timer_begin(&mut list, 0);
    // intentionally no operations to measure minimal overhead
    ctx.gpu_timer_end(&mut list, 0);
    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
    // Timing results are valid only after submission and waiting.
    ctx.wait(fence).unwrap();

    let elapsed = ctx.get_elapsed_gpu_time_ms(0).unwrap();
    assert!(elapsed >= 0.0);

    ctx.destroy_cmd_queue(list);
}
