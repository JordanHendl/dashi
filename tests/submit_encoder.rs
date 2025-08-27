use dashi::{
    driver::command::CommandEncoder,
    *,
};

#[test]
fn submit_encoder() {
    let mut ctx = match gpu::Context::headless(&ContextInfo::default()) {
        Ok(ctx) => ctx,
        Err(err) => {
            eprintln!(
                "Skipping submit_encoder test: Vulkan initialization unavailable: {:?}",
                err
            );
            return;
        }
    };

    let mut list = ctx
        .begin_command_list(&CommandListInfo { debug_name: "encoder", ..Default::default() })
        .unwrap();

    let enc = CommandEncoder::new();
    let fence = ctx
        .submit_encoder(&mut list, &enc, &SubmitInfo::default())
        .unwrap();
    ctx.wait(fence).unwrap();

    ctx.destroy_cmd_list(list);
    ctx.destroy();
}

