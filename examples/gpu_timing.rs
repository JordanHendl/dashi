use dashi::*;

fn main() {
    let mut ctx = gpu::Context::headless(&ContextInfo::default()).unwrap();
    ctx.init_gpu_timers(1).unwrap();

    let mut list = ctx.begin_command_list(&Default::default()).unwrap();
    ctx.gpu_timer_begin(&mut list, 0);
    // no-op workload
    ctx.gpu_timer_end(&mut list, 0);

    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
    ctx.wait(fence).unwrap();

    if let Some(ms) = ctx.get_elapsed_gpu_time_ms(0) {
        println!("GPU elapsed time: {:.3} ms", ms);
    } else {
        println!("Unable to query GPU time");
    }

    ctx.destroy();
}
