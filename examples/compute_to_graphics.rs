use dashi::driver::command::{
    BeginRenderPass, BindPipeline, BindTableCmd, BufferBarrier, ColorAttachment, CommandEncoder,
    CommandSink, CopyBuffer, CopyImage, DebugMarkerBegin, DebugMarkerEnd, Dispatch, Draw,
    EndRenderPass, ImageBarrier, LoadOp, RenderPassDesc, StoreOp,
};
use dashi::driver::state::SubresourceRange;
use dashi::driver::types::{Handle, Pipeline};
use dashi::Image;

struct PrintSink;

impl CommandSink for PrintSink {
    fn begin_render_pass(&mut self, _: &BeginRenderPass) {
        println!("begin_render_pass");
    }
    fn end_render_pass(&mut self, _: &EndRenderPass) {
        println!("end_render_pass");
    }
    fn bind_pipeline(&mut self, _: &BindPipeline) {
        println!("bind_pipeline");
    }
    fn bind_table(&mut self, _: &BindTableCmd) {
        println!("bind_table");
    }
    fn draw(&mut self, _: &Draw) {
        println!("draw");
    }
    fn dispatch(&mut self, _: &Dispatch) {
        println!("dispatch");
    }
    fn copy_buffer(&mut self, _: &CopyBuffer) {
        println!("copy_buffer");
    }
    fn copy_image(&mut self, _: &CopyImage) {
        println!("copy_image");
    }
    fn texture_barrier(&mut self, _: &ImageBarrier) {
        println!("texture_barrier");
    }
    fn buffer_barrier(&mut self, _: &BufferBarrier) {
        println!("buffer_barrier");
    }
    fn debug_marker_begin(&mut self, _: &DebugMarkerBegin) {
        println!("debug_marker_begin");
    }
    fn debug_marker_end(&mut self, _: &DebugMarkerEnd) {
        println!("debug_marker_end");
    }
}

fn main() {
    let src = Handle::<Image>::new(1, 0);
    let dst = Handle::<Image>::new(2, 0);
    let range = SubresourceRange::new(0, 1, 0, 1);
    let color = ColorAttachment {
        handle: dst,
        range,
        clear: [0.0; 4],
        load: LoadOp::Load,
        store: StoreOp::Store,
        _pad: [0; 2],
    };

    // Record commands using the IR encoder (acts like a secondary command buffer).
    let mut enc = CommandEncoder::new();

    // "Compute" step: copy data into the destination image.
    enc.copy_image(src, dst, range);

    // "Graphics" step: render to the same image.
    enc.begin_render_pass(RenderPassDesc { colors: &[color], depth: None });
    enc.bind_pipeline(Handle::<Pipeline>::new(3, 0));
    enc.draw(3, 1);
    enc.end_render_pass();

    // Replay on a sink to show the automatically inserted barriers.
    let mut sink = PrintSink;
    enc.submit(&mut sink);
}

