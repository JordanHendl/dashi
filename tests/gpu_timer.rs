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

#[test]
fn command_tape_timer_inside_render_pass() {
    use dashi::driver::command::BeginRenderPass;
    let mut ctx = ValidationContext::headless(&ContextInfo::default()).unwrap();
    ctx.init_gpu_timers(1).unwrap();
    let viewport = Viewport {
        area: FRect2D {
            w: 2.0,
            h: 2.0,
            ..Default::default()
        },
        scissor: Rect2D {
            w: 2,
            h: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let image = ctx
        .make_image(&ImageInfo {
            debug_name: "timer depth",
            dim: [2, 2, 1],
            format: Format::D24S8,
            ..Default::default()
        })
        .unwrap();
    let render_pass = ctx
        .make_render_pass(&RenderPassInfo {
            debug_name: "timer pass",
            viewport,
            subpasses: &[SubpassDescription {
                color_attachments: &[],
                depth_stencil_attachment: Some(&AttachmentDescription {
                    format: Format::D24S8,
                    ..Default::default()
                }),
                subpass_dependencies: &[],
            }],
        })
        .unwrap();
    for _ in 0..3 {
        let mut list = ctx
            .begin_command_queue(QueueType::Graphics, "timer pass", false)
            .unwrap();
        CommandStream::new()
            .begin()
            .begin_render_pass(&BeginRenderPass {
                viewport,
                render_pass,
                depth_attachment: Some(ImageView {
                    img: image,
                    aspect: AspectMask::Depth,
                    ..Default::default()
                }),
                depth_clear: Some(ClearValue::DepthStencil {
                    depth: 1.0,
                    stencil: 0,
                }),
                ..Default::default()
            })
            .gpu_timer_begin(0)
            .gpu_timer_end(0)
            .stop_drawing()
            .end()
            .append(&mut list)
            .unwrap();
        let fence = ctx.submit(&mut list, &Default::default()).unwrap();
        ctx.wait(fence).unwrap();
        assert!(ctx.get_elapsed_gpu_time_ms(0).unwrap() >= 0.0);
        ctx.destroy_cmd_queue(list);
    }
    ctx.destroy_render_pass(render_pass);
    ctx.destroy_image(image);
}
