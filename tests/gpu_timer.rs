use dashi::*;

#[test]
fn gpu_timer() {
    let mut ctx = match gpu::Context::headless(&ContextInfo::default()) {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!(
                "Skipping gpu_timer test: Vulkan initialization unavailable: {:?}",
                err
            );
            return;
        }
    };
    ctx.init_gpu_timers(1).unwrap();

    let mut list = ctx
        .begin_command_list(&CommandListInfo { debug_name: "timer", ..Default::default() })
        .unwrap();
    ctx.gpu_timer_begin(&mut list, 0);
    // intentionally no operations to measure minimal overhead
    ctx.gpu_timer_end(&mut list, 0);
    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
    ctx.wait(fence).unwrap();

    let elapsed = ctx.get_elapsed_gpu_time_ms(0).unwrap();
    assert!(elapsed >= 0.0);

    ctx.destroy_cmd_list(list);
    ctx.destroy();
}
