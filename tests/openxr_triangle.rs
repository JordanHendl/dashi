#[cfg(feature = "dashi-openxr")]
use dashi::*;
#[cfg(feature = "dashi-openxr")]
use std::time::{Duration, Instant};

#[cfg(feature = "dashi-openxr")]
pub struct Timer {
    start_time: Option<Instant>,
    elapsed: Duration,
    is_paused: bool,
}

#[cfg(feature = "dashi-openxr")]
impl Timer {
    pub fn new() -> Self {
        Self { start_time: None, elapsed: Duration::new(0, 0), is_paused: false }
    }
    pub fn start(&mut self) {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        } else if self.is_paused {
            self.start_time = Some(Instant::now() - self.elapsed);
            self.is_paused = false;
        }
    }
    pub fn elapsed_ms(&self) -> u128 {
        if let Some(start_time) = self.start_time {
            if self.is_paused { self.elapsed.as_millis() } else { start_time.elapsed().as_millis() }
        } else { self.elapsed.as_millis() }
    }
}

#[cfg(feature = "dashi-openxr")]
#[test]
#[ignore]
fn openxr_triangle() {
    let device = SelectedDevice::default();
    println!("Using device {}", device);
    let mut ctx = gpu::Context::new(&ContextInfo { device }).unwrap();
    let mut display = ctx.make_xr_display(&XrDisplayInfo::default()).unwrap();
    let views = display.xr_view_configuration();
    let width = views[0].recommended_image_rect_width;
    let height = views[0].recommended_image_rect_height;

    const VERTICES: [[f32; 2]; 3] = [[0.0, -0.5],[0.5, 0.5],[-0.5, 0.5]];
    const INDICES: [u32; 3] = [0,1,2];
    let vertices = ctx.make_buffer(&BufferInfo{
        debug_name:"vertices",
        byte_size:(VERTICES.len()*std::mem::size_of::<f32>()*2) as u32,
        visibility:MemoryVisibility::Gpu,
        usage:BufferUsage::VERTEX,
        initial_data:unsafe{Some(VERTICES.align_to::<u8>().1)}
    }).unwrap();
    let indices = ctx.make_buffer(&BufferInfo{
        debug_name:"indices",
        byte_size:(INDICES.len()*std::mem::size_of::<u32>()) as u32,
        visibility:MemoryVisibility::Gpu,
        usage:BufferUsage::INDEX,
        initial_data:unsafe{Some(INDICES.align_to::<u8>().1)}
    }).unwrap();
    let fb = ctx.make_image(&ImageInfo{
        debug_name:"color_attachment",
        dim:[width,height,1],
        format:Format::RGBA8,
        mip_levels:1,
        initial_data:None,
        ..Default::default()
    }).unwrap();
    let fb_view = ctx.make_image_view(&ImageViewInfo{img:fb,..Default::default()}).unwrap();
    let bg_layout = ctx.make_bind_group_layout(&BindGroupLayoutInfo{
        shaders:&[ShaderInfo{
            shader_type:ShaderType::Vertex,
            variables:&[BindGroupVariable{var_type:BindGroupVariableType::DynamicUniform,binding:0,..Default::default()}]
        }],
        debug_name:"OpenXR Triangle"
    }).unwrap();
    let pipeline_layout = ctx.make_graphics_pipeline_layout(&GraphicsPipelineLayoutInfo{
        vertex_info:VertexDescriptionInfo{
            entries:&[VertexEntryInfo{format:ShaderPrimitiveType::Vec2,location:0,offset:0}],
            stride:8,
            rate:VertexRate::Vertex
        },
        bg_layouts:[Some(bg_layout),None,None,None],
        bt_layouts:[None,None,None,None],
        shaders:&[
            PipelineShaderInfo{
                stage:ShaderType::Vertex,
                spirv:inline_spirv::inline_spirv!(r"#version 450
layout(location = 0) in vec2 inPosition;
layout(location = 0) out vec2 frag_color;
layout(binding = 0) uniform position_offset { vec2 pos; };
void main(){frag_color=inPosition;gl_Position=vec4(inPosition+pos,0.0,1.0);}",vert),
                specialization:&[]
            },
            PipelineShaderInfo{
                stage:ShaderType::Fragment,
                spirv:inline_spirv::inline_spirv!(r"#version 450
layout(location = 0) in vec2 frag_color;
layout(location = 0) out vec4 out_color;
void main(){ out_color=vec4(frag_color.xy,0,1); }",frag),
                specialization:&[]
            }
        ],
        details:Default::default(),
        debug_name:"OpenXR Pipeline"
    }).expect("pipeline");
    let render_pass = ctx.make_render_pass(&RenderPassInfo{
        viewport:Viewport{area:FRect2D{w:width as f32,h:height as f32,..Default::default()},scissor:Rect2D{w:width,h:height,..Default::default()},..Default::default()},
        subpasses:&[SubpassDescription{color_attachments:&[AttachmentDescription{..Default::default()}],depth_stencil_attachment:None,subpass_dependencies:&[]}],
        debug_name:"renderpass"
    }).unwrap();
    let render_target = ctx.make_render_target(&RenderTargetInfo{debug_name:"rt",render_pass,attachments:&[fb_view]}).unwrap();
    let graphics_pipeline = ctx.make_graphics_pipeline(&GraphicsPipelineInfo{layout:pipeline_layout,render_pass,debug_name:"Pipeline",..Default::default()}).unwrap();
    let mut allocator = ctx.make_dynamic_allocator(&Default::default()).unwrap();
    let bind_group = ctx.make_bind_group(&BindGroupInfo{debug_name:"OpenXR Triangle",layout:bg_layout,bindings:&[BindingInfo{resource:ShaderResource::Dynamic(&allocator),binding:0}],..Default::default()}).unwrap();
    let mut timer = Timer::new();
    timer.start();
    let mut framed_list = FramedCommandList::new(&mut ctx,"Default",2);
    allocator.reset();
    let (_idx,state) = ctx.acquire_xr_image(&mut display).unwrap();
    framed_list.record(|list|{
        list.begin_drawing(&DrawBegin{
            viewport:Viewport{area:FRect2D{w:width as f32,h:height as f32,..Default::default()},scissor:Rect2D{w:width,h:height,..Default::default()},..Default::default()},
            pipeline:graphics_pipeline,
            render_target,
            clear_values:&[ClearValue::Color([0.0,0.0,0.0,1.0])]
        }).unwrap();
        let mut buf=allocator.bump().unwrap();
        let pos=&mut buf.slice::<[f32;2]>()[0];
        pos[0]=(timer.elapsed_ms() as f32/1000.0).sin();
        pos[1]=(timer.elapsed_ms() as f32/1000.0).cos();
        list.append(Command::DrawIndexed(DrawIndexed{vertices,indices,index_count:INDICES.len() as u32,bind_groups:[Some(bind_group),None,None,None],dynamic_buffers:[Some(buf),None,None,None],..Default::default()}));
        list.end_drawing().unwrap();
    });
    framed_list.submit(&SubmitInfo::default());
    ctx.present_xr_display(&mut display,state).unwrap();
}

#[cfg(not(feature = "dashi-openxr"))]
#[test]
#[ignore]
fn openxr_triangle() {}
