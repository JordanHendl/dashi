#![cfg(feature = "dashi-openxr")]
use dashi::*;

#[test]
fn session_requires_startup_opt_in() {
    let ctx = Context::headless(&Default::default()).unwrap();
    assert!(ctx.create_xr_session().is_err());
    ctx.destroy();
}

#[test]
#[ignore = "requires an installed OpenXR runtime and connected headset"]
fn runtime_graphics_bootstrap_creates_session() {
    let xr = XrGraphics::load("dashi OpenXR smoke").unwrap();
    let ctx = Context::new(&ContextInfo {
        xr: Some(xr.clone()),
        ..Default::default()
    })
    .unwrap();
    let (session, waiter, stream) = ctx.create_xr_session().unwrap();
    drop(stream);
    drop(waiter);
    drop(session);
    ctx.destroy();
}
