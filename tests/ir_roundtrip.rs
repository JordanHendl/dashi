//use dashi::driver::command::{
//    BeginRenderPass, BindPipeline, BindTableCmd, BufferBarrier, ColorAttachment, CommandEncoder,
//    CommandSink, CopyBuffer, CopyImage, DebugMarkerBegin, DebugMarkerEnd, Dispatch,
//    Draw, EndRenderPass, ImageBarrier, LoadOp, Op, RenderPassDesc, StoreOp,
//};
//use dashi::driver::state::SubresourceRange;
//use dashi::driver::types::{BindTable as BindTableRes, Handle, Pipeline};
//use dashi::{Buffer, Image};
//use dashi::ir::{CommandReplayer, Replayer};
//
//#[derive(Debug, PartialEq)]
//enum Recorded {
//    BeginRenderPass(BeginRenderPass),
//    EndRenderPass(EndRenderPass),
//    BindPipeline(BindPipeline),
//    BindTable(BindTableCmd),
//    Draw(Draw),
//    Dispatch(Dispatch),
//    CopyBuffer(CopyBuffer),
//    CopyImage(CopyImage),
//    ImageBarrier(ImageBarrier),
//    BufferBarrier(BufferBarrier),
//    DebugMarkerBegin(DebugMarkerBegin),
//    DebugMarkerEnd(DebugMarkerEnd),
//}
//
//#[derive(Default)]
//struct Recorder {
//    cmds: Vec<Recorded>,
//}
//
//impl CommandSink for Recorder {
//    fn begin_render_pass(&mut self, pass: &BeginRenderPass) {
//        self.cmds.push(Recorded::BeginRenderPass(*pass));
//    }
//    fn end_render_pass(&mut self, pass: &EndRenderPass) {
//        self.cmds.push(Recorded::EndRenderPass(*pass));
//    }
//    fn bind_pipeline(&mut self, cmd: &BindPipeline) {
//        self.cmds.push(Recorded::BindPipeline(*cmd));
//    }
//    fn bind_table(&mut self, cmd: &BindTableCmd) {
//        self.cmds.push(Recorded::BindTable(*cmd));
//    }
//    fn draw(&mut self, cmd: &Draw) {
//        self.cmds.push(Recorded::Draw(*cmd));
//    }
//    fn dispatch(&mut self, cmd: &Dispatch) {
//        self.cmds.push(Recorded::Dispatch(*cmd));
//    }
//    fn copy_buffer(&mut self, cmd: &CopyBuffer) {
//        self.cmds.push(Recorded::CopyBuffer(*cmd));
//    }
//    fn copy_image(&mut self, cmd: &CopyImage) {
//        self.cmds.push(Recorded::CopyImage(*cmd));
//    }
//    fn texture_barrier(&mut self, cmd: &ImageBarrier) {
//        self.cmds.push(Recorded::ImageBarrier(*cmd));
//    }
//    fn buffer_barrier(&mut self, cmd: &BufferBarrier) {
//        self.cmds.push(Recorded::BufferBarrier(*cmd));
//    }
//    fn debug_marker_begin(&mut self, cmd: &DebugMarkerBegin) {
//        self.cmds.push(Recorded::DebugMarkerBegin(*cmd));
//    }
//    fn debug_marker_end(&mut self, cmd: &DebugMarkerEnd) {
//        self.cmds.push(Recorded::DebugMarkerEnd(*cmd));
//    }
//}
//
//#[test]
//fn ir_roundtrip_core_ops() {
//    let img_a = Handle::<Image>::new(1, 0);
//    let img_b = Handle::<Image>::new(2, 0);
//    let buf_a = Handle::<Buffer>::new(3, 0);
//    let buf_b = Handle::<Buffer>::new(4, 0);
//    let pipe = Handle::<Pipeline>::new(5, 0);
//    let table = Handle::<BindTableRes>::new(6, 0);
//    let range = SubresourceRange::new(0, 1, 0, 1);
//
//    let color = ColorAttachment {
//        handle: img_b,
//        range,
//        clear: [0.0; 4],
//        load: LoadOp::Load,
//        store: StoreOp::Store,
//        _pad: [0; 2],
//    };
//
//    let mut enc = CommandEncoder::new();
//    enc.begin_debug_marker();
//    enc.copy_image(img_a, img_b, range);
//    enc.begin_render_pass(RenderPassDesc { colors: &[color], depth: None });
//    enc.bind_pipeline(pipe);
//    enc.bind_table(table);
//    enc.draw(3, 1);
//    enc.end_render_pass();
//    enc.dispatch(1, 2, 3);
//    enc.copy_buffer(buf_a, buf_b);
//    enc.end_debug_marker();
//
//    let expected: Vec<Recorded> = enc
//        .iter()
//        .map(|c| match c.op {
//            Op::BeginRenderPass => Recorded::BeginRenderPass(*c.payload()),
//            Op::EndRenderPass => Recorded::EndRenderPass(*c.payload()),
//            Op::BindPipeline => Recorded::BindPipeline(*c.payload()),
//            Op::BindTable => Recorded::BindTable(*c.payload()),
//            Op::Draw => Recorded::Draw(*c.payload()),
//            Op::Dispatch => Recorded::Dispatch(*c.payload()),
//            Op::CopyBuffer => Recorded::CopyBuffer(*c.payload()),
//            Op::CopyImage => Recorded::CopyImage(*c.payload()),
//            Op::ImageBarrier => Recorded::ImageBarrier(*c.payload()),
//            Op::BufferBarrier => Recorded::BufferBarrier(*c.payload()),
//            Op::DebugMarkerBegin => Recorded::DebugMarkerBegin(*c.payload()),
//            Op::DebugMarkerEnd => Recorded::DebugMarkerEnd(*c.payload()),
//        })
//        .collect();
//
//    let mut recorder = Recorder::default();
//    CommandReplayer::new(&mut recorder).replay(&enc);
//    assert_eq!(expected, recorder.cmds);
//}
//
