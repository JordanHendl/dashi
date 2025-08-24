//! Builder pattern wrappers for Dashi Context resource creation.

use crate::utils::Handle;
use crate::{
    AttachmentDescription, BindGroupLayout, ComputePipeline, ComputePipelineInfo,
    ComputePipelineLayout, ComputePipelineLayoutInfo, Display, DisplayInfo, GraphicsPipeline,
    GraphicsPipelineDetails, GraphicsPipelineInfo, GraphicsPipelineLayout,
    GraphicsPipelineLayoutInfo, PipelineShaderInfo, RenderPass, RenderPassInfo, SubpassDependency,
    SubpassDescription, VertexDescriptionInfo, Viewport, WindowBuffering, DynamicState,
};
use crate::{Context, GPUError};
#[cfg(feature = "dashi-openxr")]
use crate::XrDisplayInfo;

/// Builds a RenderPass via the builder pattern.
pub struct RenderPassBuilder<'a> {
    debug_name: &'a str,
    viewport: Viewport,
    subpasses: Vec<SubpassDescription<'a>>,
}

impl<'a> RenderPassBuilder<'a> {
    /// Create a new builder with the given debug name and viewport.
    pub fn new(debug_name: &'a str, viewport: Viewport) -> Self {
        Self {
            debug_name,
            viewport,
            subpasses: Vec::new(),
        }
    }

    /// Add a subpass with color attachments, optional depth, and dependencies.
    pub fn add_subpass(
        mut self,
        color_attachments: &'a [AttachmentDescription],
        depth_stencil: Option<&'a AttachmentDescription>,
        dependencies: &'a [SubpassDependency],
    ) -> Self {
        let desc = SubpassDescription {
            color_attachments,
            depth_stencil_attachment: depth_stencil,
            subpass_dependencies: dependencies,
        };
        self.subpasses.push(desc);
        self
    }

    /// Finalize and create the RenderPass in the given context.
    pub fn build(self, ctx: &mut Context) -> Result<Handle<RenderPass>, GPUError> {
        let info = RenderPassInfo {
            debug_name: self.debug_name,
            viewport: self.viewport,
            subpasses: &self.subpasses,
        };
        ctx.make_render_pass(&info)
    }
}

/// Builds a Display (window + swapchain) via the builder pattern.
pub struct DisplayBuilder {
    info: DisplayInfo,
}

impl DisplayBuilder {
    /// Start with default DisplayInfo.
    pub fn new() -> Self {
        Self {
            info: DisplayInfo::default(),
        }
    }

    /// Set the window title.
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.info.window.title = title.into();
        self
    }

    /// Set the window size.
    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.info.window.size = [width, height];
        self
    }

    /// Enable or disable window resizing.
    pub fn resizable(mut self, resizable: bool) -> Self {
        self.info.window.resizable = resizable;
        self
    }

    /// Enable or disable vsync.
    pub fn vsync(mut self, vsync: bool) -> Self {
        self.info.vsync = vsync;
        self
    }

    /// Choose double or triple buffering.
    pub fn buffering(mut self, buffering: WindowBuffering) -> Self {
        self.info.buffering = buffering;
        self
    }

    /// Finalize and create the Display.
    #[cfg(not(feature = "dashi-openxr"))]
    pub fn build(self, ctx: &mut Context) -> Result<Display, GPUError> {
        ctx.make_display(&self.info)
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn build(self, ctx: &mut Context) -> Result<Display, GPUError> {
        ctx.make_xr_display(&XrDisplayInfo::default())
    }

    /// Create an OpenXR display when the `dashi-openxr` feature is enabled.
    #[cfg(feature = "dashi-openxr")]
    pub fn build_xr(&self, ctx: &mut Context) -> Result<Display, GPUError> {
        ctx.make_xr_display(&XrDisplayInfo::default())
    }
}

/// Builds a GraphicsPipelineLayout via the builder pattern.
pub struct GraphicsPipelineLayoutBuilder<'a> {
    debug_name: &'a str,
    vertex_info: Option<VertexDescriptionInfo<'a>>,
    bg_layouts: [Option<Handle<BindGroupLayout>>; 4],
    shaders: Vec<PipelineShaderInfo<'a>>,
    details: GraphicsPipelineDetails,
}

impl<'a> GraphicsPipelineLayoutBuilder<'a> {
    /// Start a new builder for a graphics pipeline layout.
    pub fn new(debug_name: &'a str) -> Self {
        Self {
            debug_name,
            vertex_info: None,
            bg_layouts: [None, None, None, None],
            shaders: Vec::new(),
            details: GraphicsPipelineDetails::default(),
        }
    }

    /// Specify the vertex input description.
    pub fn vertex_info(mut self, info: VertexDescriptionInfo<'a>) -> Self {
        self.vertex_info = Some(info);
        self
    }

    /// Attach a bind group layout at a given index (0..3).
    pub fn bind_group_layout(mut self, index: usize, layout: Handle<BindGroupLayout>) -> Self {
        if index < 4 {
            self.bg_layouts[index] = Some(layout);
        }
        self
    }

    /// Add a shader stage (SPIR-V module info).
    pub fn shader(mut self, shader_info: PipelineShaderInfo<'a>) -> Self {
        self.shaders.push(shader_info);
        self
    }

    /// Configure blend, culling, topology, etc.
    pub fn details(mut self, details: GraphicsPipelineDetails) -> Self {
        self.details = details;
        self
    }

    /// Specify which pipeline states will be set dynamically.
    pub fn dynamic_states(mut self, states: Vec<DynamicState>) -> Self {
        self.details.dynamic_states = states;
        self
    }

    /// Finalize and create the GraphicsPipelineLayout.
    pub fn build(self, ctx: &mut Context) -> Result<Handle<GraphicsPipelineLayout>, GPUError> {
        let info = GraphicsPipelineLayoutInfo {
            debug_name: self.debug_name,
            vertex_info: self.vertex_info.expect("vertex_info is required"),
            bg_layouts: self.bg_layouts,
            shaders: &self.shaders,
            details: self.details,
        };
        ctx.make_graphics_pipeline_layout(&info)
    }
}

/// Builds a GraphicsPipeline via the builder pattern.
pub struct GraphicsPipelineBuilder {
    debug_name: String,
    layout: Handle<GraphicsPipelineLayout>,
    render_pass: Handle<RenderPass>,
    subpass_id: u8,
}

impl GraphicsPipelineBuilder {
    /// Start a new pipeline builder with a name.
    pub fn new(debug_name: impl Into<String>) -> Self {
        Self {
            debug_name: debug_name.into(),
            layout: Handle::default(),
            render_pass: Handle::default(),
            subpass_id: 0,
        }
    }

    /// Specify the pipeline layout.
    pub fn layout(mut self, layout: Handle<GraphicsPipelineLayout>) -> Self {
        self.layout = layout;
        self
    }

    /// Specify the render pass.
    pub fn render_pass(mut self, rp: Handle<RenderPass>) -> Self {
        self.render_pass = rp;
        self
    }

    /// Specify which subpass.
    pub fn subpass(mut self, id: u8) -> Self {
        self.subpass_id = id;
        self
    }

    /// Finalize and create the GraphicsPipeline.
    pub fn build(self, ctx: &mut Context) -> Result<Handle<GraphicsPipeline>, GPUError> {
        let info = GraphicsPipelineInfo {
            debug_name: &self.debug_name,
            layout: self.layout,
            render_pass: self.render_pass,
            subpass_id: self.subpass_id,
        };
        ctx.make_graphics_pipeline(&info)
    }
}

/// Builds a ComputePipelineLayout via the builder pattern.
pub struct ComputePipelineLayoutBuilder<'a> {
    bg_layouts: [Option<Handle<BindGroupLayout>>; 4],
    shader: Option<PipelineShaderInfo<'a>>,
}

impl<'a> ComputePipelineLayoutBuilder<'a> {
    /// Start a new builder for compute pipeline layout.
    pub fn new() -> Self {
        Self {
            bg_layouts: [None, None, None, None],
            shader: None,
        }
    }

    /// Attach a bind group layout at a given index.
    pub fn bind_group_layout(mut self, index: usize, layout: Handle<BindGroupLayout>) -> Self {
        if index < 4 {
            self.bg_layouts[index] = Some(layout);
        }
        self
    }

    /// Specify the compute shader module info.
    pub fn shader(mut self, shader_info: PipelineShaderInfo<'a>) -> Self {
        self.shader = Some(shader_info);
        self
    }

    /// Finalize and create the ComputePipelineLayout.
    pub fn build(self, ctx: &mut Context) -> Result<Handle<ComputePipelineLayout>, GPUError> {
        let info = ComputePipelineLayoutInfo {
            bg_layouts: self.bg_layouts,
            shader: &self.shader.expect("shader is required"),
        };
        ctx.make_compute_pipeline_layout(&info)
    }
}

/// Builds a ComputePipeline via the builder pattern.
pub struct ComputePipelineBuilder {
    debug_name: String,
    layout: Handle<ComputePipelineLayout>,
}

impl ComputePipelineBuilder {
    /// Start a new compute pipeline builder with a name.
    pub fn new(debug_name: impl Into<String>) -> Self {
        Self {
            debug_name: debug_name.into(),
            layout: Handle::default(),
        }
    }

    /// Specify which layout to use.
    pub fn layout(mut self, layout: Handle<ComputePipelineLayout>) -> Self {
        self.layout = layout;
        self
    }

    /// Finalize and create the ComputePipeline.
    pub fn build(self, ctx: &mut Context) -> Result<Handle<ComputePipeline>, GPUError> {
        let info = ComputePipelineInfo {
            debug_name: &self.debug_name,
            layout: self.layout,
        };
        ctx.make_compute_pipeline(&info)
    }
}
/// Builds a BindGroupLayout via the builder pattern.
pub struct BindGroupLayoutBuilder<'a> {
    debug_name: &'a str,
    shaders: Vec<crate::ShaderInfo<'a>>,
}

impl<'a> BindGroupLayoutBuilder<'a> {
    /// Start a new builder with a debug name.
    pub fn new(debug_name: &'a str) -> Self {
        Self {
            debug_name,
            shaders: Vec::new(),
        }
    }

    /// Add a shader stage with its variable descriptors.
    pub fn shader(mut self, shader_info: crate::ShaderInfo<'a>) -> Self {
        self.shaders.push(shader_info);
        self
    }

    /// Finalize and create the BindGroupLayout.
    pub fn build(self, ctx: &mut Context) -> Result<Handle<BindGroupLayout>, GPUError> {
        let info = crate::BindGroupLayoutInfo {
            debug_name: self.debug_name,
            shaders: &self.shaders,
        };
        ctx.make_bind_group_layout(&info)
    }
}

/// Builds a BindGroup via the builder pattern.
pub struct BindGroupBuilder<'a> {
    debug_name: &'a str,
    layout: Handle<BindGroupLayout>,
    bindings: Vec<crate::BindingInfo<'a>>,
    set: u32,
}

impl<'a> BindGroupBuilder<'a> {
    /// Start a new builder for a bind group.
    pub fn new(debug_name: &'a str) -> Self {
        Self {
            debug_name,
            layout: Handle::default(),
            bindings: Vec::new(),
            set: 0,
        }
    }

    /// Specify the layout handle to allocate from.
    pub fn layout(mut self, layout: Handle<BindGroupLayout>) -> Self {
        self.layout = layout;
        self
    }

    /// Add a binding with its resource at a binding slot.
    pub fn binding(mut self, binding: u32, resource: crate::ShaderResource<'a>) -> Self {
        self.bindings.push(crate::BindingInfo { binding, resource });
        self
    }

    /// Set the descriptor set index.
    pub fn set(mut self, set: u32) -> Self {
        self.set = set;
        self
    }

    /// Finalize and create the BindGroup.
    pub fn build(self, ctx: &mut Context) -> Result<Handle<crate::BindGroup>, GPUError> {
        let info = crate::BindGroupInfo {
            debug_name: self.debug_name,
            layout: self.layout,
            bindings: &self.bindings,
            set: self.set,
        };
        ctx.make_bind_group(&info)
    }
}
// Unit tests for each builder
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ContextInfo;
    use crate::*;
    use serial_test::serial;
    use std::panic;

    #[test]
    #[serial]
    fn test_render_pass_builder() {
        let mut ctx = Context::headless(&ContextInfo::default()).unwrap();
        let viewport: Viewport = Default::default();
        let desc = AttachmentDescription::default();
        let dep = SubpassDependency {
            subpass_id: 0,
            attachment_id: 0,
            depth_id: 0,
        };
        let rp = RenderPassBuilder::new("rp", viewport)
            .add_subpass(&[desc], None, &[dep])
            .build(&mut ctx)
            .unwrap();
        ctx.destroy_render_pass(rp);
        ctx.destroy();
    }

    #[test]
    #[serial]
    fn test_display_builder() {
        let mut ctx = Context::headless(&ContextInfo::default()).unwrap();
        let result = DisplayBuilder::new()
            .title("test")
            .size(300, 200)
            .resizable(true)
            .vsync(false)
            .buffering(WindowBuffering::Triple)
            .build(&mut ctx);

        match result {
            Ok(dsp) => {
                ctx.destroy_display(dsp);
                ctx.destroy();
            }
            Err(GPUError::HeadlessDisplayNotSupported) => {
                ctx.destroy();
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    #[serial]
    fn test_graphics_pipeline_layout_builder_missing_vertex_info() {
        let mut ctx = Context::headless(&ContextInfo::default()).unwrap();
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            GraphicsPipelineLayoutBuilder::new("gpl")
                .build(&mut ctx)
                .unwrap();
        }));
        ctx.destroy();
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_graphics_pipeline_builder_missing_fields() {
        let mut ctx = Context::headless(&ContextInfo::default()).unwrap();
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            GraphicsPipelineBuilder::new("gp").build(&mut ctx).unwrap();
        }));
        ctx.destroy();
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_compute_pipeline_layout_builder_missing_shader() {
        let mut ctx = Context::headless(&ContextInfo::default()).unwrap();
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            ComputePipelineLayoutBuilder::new().build(&mut ctx).unwrap();
        }));
        ctx.destroy();
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_compute_pipeline_builder_missing_layout() {
        let mut ctx = Context::headless(&ContextInfo::default()).unwrap();
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            ComputePipelineBuilder::new("cp").build(&mut ctx).unwrap();
        }));
        ctx.destroy();
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_bind_group_layout_builder() {
        let mut ctx = Context::headless(&ContextInfo::default()).unwrap();
        let shader_info = crate::ShaderInfo {
            shader_type: crate::ShaderType::All,
            variables: &[],
        };
        let _bgl = BindGroupLayoutBuilder::new("bgl")
            .shader(shader_info)
            .build(&mut ctx)
            .unwrap();
        //        ctx.destroy_bind_group_layout(_bgl);
        ctx.destroy();
    }

    #[test]
    #[serial]
    fn test_bind_group_builder() {
        let mut ctx = Context::headless(&ContextInfo::default()).unwrap();
        let shader_info = crate::ShaderInfo {
            shader_type: crate::ShaderType::All,
            variables: &[],
        };
        let bgl = BindGroupLayoutBuilder::new("bgl")
            .shader(shader_info)
            .build(&mut ctx)
            .unwrap();
        let _bg = BindGroupBuilder::new("bg")
            .layout(bgl)
            .build(&mut ctx)
            .unwrap();
        //    ctx.destroy_bind_group(_bg);
        ctx.destroy();
    }

    #[test]
    #[serial]
    fn render_pass_builder_creates_pass_with_and_without_depth() {
        let mut ctx = Context::headless(&Default::default()).unwrap();
        let vp = Viewport::default();

        // 1) single subpass, only color
        let color_desc = AttachmentDescription {
            format: Format::RGBA8,
            samples: SampleCount::S1,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            stencil_load_op: LoadOp::DontCare,
            stencil_store_op: StoreOp::DontCare,
        };
        let dep = SubpassDependency {
            subpass_id: 0,
            attachment_id: 0,
            depth_id: 0,
        };

        let rp = RenderPassBuilder::new("rp_color_only", vp)
            .add_subpass(&[color_desc], None, &[])
            .build(&mut ctx)
            .expect("color-only pass");

        // tearing down
        ctx.destroy_render_pass(rp);

        // 2) with depth+stencil attachment
        let ds_desc = AttachmentDescription {
            format: Format::D24S8,
            samples: SampleCount::S1,
            load_op: LoadOp::Clear,
            store_op: StoreOp::DontCare,
            stencil_load_op: LoadOp::Clear,
            stencil_store_op: StoreOp::DontCare,
        };
        let rp2 = RenderPassBuilder::new("rp_with_depth", vp)
            .add_subpass(&[color_desc], Some(&ds_desc), &[dep])
            .build(&mut ctx)
            .expect("with-depth pass");

        ctx.destroy_render_pass(rp2);
        ctx.destroy();
    }

    // Happy‐path smoke‐tests (still destroy resources cleanly):
    #[test]
    #[serial]
    fn test_render_pass_builder_happy_path() {
        let mut ctx = Context::headless(&Default::default()).unwrap();
        let vp = Viewport::default();
        let desc = AttachmentDescription::default();
        let rp = RenderPassBuilder::new("rp", vp)
            .add_subpass(&[desc], None, &[])
            .build(&mut ctx)
            .unwrap();
        ctx.destroy_render_pass(rp);
        ctx.destroy();
    }

    #[test]
    #[serial]
    fn test_graphics_pipeline_layout_builder_happy_path() {
        let mut ctx = Context::headless(&Default::default()).unwrap();
        // minimal vertex_info
        let vert_info = VertexDescriptionInfo {
            stride: 16,
            rate: VertexRate::Vertex,
            entries: &[],
        };
        // dummy bind group layout
        let bgl = ctx
            .make_bind_group_layout(&BindGroupLayoutInfo {
                debug_name: "bgl",
                shaders: &[],
            })
            .unwrap();

        let spirv_vert = inline_spirv::inline_spirv!(
            r#"
            #version 450
            void main() { gl_Position = vec4(0.0); }
            "#,
            vert
        );
        let _layout = GraphicsPipelineLayoutBuilder::new("gpl")
            .vertex_info(vert_info)
            .bind_group_layout(0, bgl)
            .shader(PipelineShaderInfo {
                stage: ShaderType::Vertex,
                spirv: spirv_vert,
                specialization: &[],
            })
            .build(&mut ctx)
            .unwrap();
        ctx.destroy();
    }
}
