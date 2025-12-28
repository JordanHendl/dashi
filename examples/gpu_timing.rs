use dashi::*;

fn main() {
    let mut ctx = gpu::Context::headless(&ContextInfo::default()).unwrap();
    // GPU timers must be initialized before use.
    ctx.init_gpu_timers(1).unwrap();

    let ctx_ptr = ctx.vulkan_mut_ptr();
    let mut list = ctx.pool_mut(QueueType::Graphics).begin(ctx_ptr, "", false).unwrap();
    // Begin and end must bracket commands on the same list.
    ctx.gpu_timer_begin(&mut list, 0);
    // no-op workload
    ctx.gpu_timer_end(&mut list, 0);

    let fence = ctx.submit(&mut list, &Default::default()).unwrap();
    // Timing results are valid only after submission and waiting.
    ctx.wait(fence).unwrap();

    if let Some(ms) = ctx.get_elapsed_gpu_time_ms(0) {
        println!("GPU elapsed time: {:.3} ms", ms);
    } else {
        println!("Unable to query GPU time");
    }

    ctx.destroy();
}
