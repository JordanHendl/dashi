#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
use dashi::{webgpu::Context, ContextInfo, WebSurfaceInfo};
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
use wasm_bindgen::prelude::*;

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
#[wasm_bindgen(start)]
pub fn start() -> Result<(), JsValue> {
    let mut info = ContextInfo::default();
    info.web_surface = Some(WebSurfaceInfo::CanvasId("dashi-canvas".to_string()));

    let ctx = Context::new(&info).map_err(|err| JsValue::from_str(&err.to_string()))?;
    let mut display = ctx
        .make_display(&info)
        .map_err(|err| JsValue::from_str(&err.to_string()))?;
    let frame = display
        .acquire_frame(ctx.device())
        .map_err(|err| JsValue::from_str(&err.to_string()))?;

    let mut encoder = ctx
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dashi-webgpu-encoder"),
        });
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dashi-webgpu-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.07,
                        g: 0.12,
                        b: 0.18,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
    }

    ctx.queue().submit(Some(encoder.finish()));
    frame.present();

    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
fn main() {}

#[cfg(not(all(target_arch = "wasm32", feature = "webgpu")))]
fn main() {
    println!("This example is intended to run on wasm32 with the webgpu feature enabled.");
}
