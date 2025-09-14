//use dashi::framegraph::{Graph, Node, PassDecl, ImageUse};
//use dashi::sync::state::{Access, ImageLayout, ResState, Stage};
//use dashi::Handle;
//use dashi::gpu::vulkan::Image;
//
//fn main() {
//    let mut graph = Graph::new();
//
//    let image = Handle::<Image>::default();
//
//    let mut compute_decl = PassDecl::new();
//    compute_decl.images.push(ImageUse {
//        handle: image,
//        state: ResState {
//            access: Access::SHADER_WRITE,
//            stages: Stage::COMPUTE_SHADER,
//            layout: ImageLayout::GENERAL,
//            _pad: 0,
//        },
//    });
//
//    let compute = graph.add_node(Node::new(compute_decl, || {
//        println!("dispatch compute");
//    }));
//
//    let mut graphics_decl = PassDecl::new();
//    graphics_decl.images.push(ImageUse {
//        handle: image,
//        state: ResState {
//            access: Access::COLOR_ATTACHMENT_READ | Access::COLOR_ATTACHMENT_WRITE,
//            stages: Stage::COLOR_ATTACHMENT_OUTPUT,
//            layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
//            _pad: 0,
//        },
//    });
//
//    let graphics = graph.add_node(Node::new(graphics_decl, || {
//        println!("draw graphics");
//    }));
//    graph.add_dependency(graphics, compute);
//
//    graph.execute();
//    println!("execution order: {:?}", graph.execution_order());
//}
//
//

fn main() {

}
