use super::*;

#[derive(Clone, Default, Debug)]
pub struct ComputePipelineLayout {
    pub(super) shader_stage: vk::PipelineShaderStageCreateInfo,
    pub(super) layout: vk::PipelineLayout,
}

#[derive(Clone, Default, Debug)]
pub struct GraphicsPipelineLayout {
    pub(super) input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    pub(super) shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    pub(super) depth_stencil: Option<vk::PipelineDepthStencilStateCreateInfo>,
    pub(super) multisample: vk::PipelineMultisampleStateCreateInfo,
    pub(super) rasterizer: vk::PipelineRasterizationStateCreateInfo,
    pub(super) vertex_input: vk::VertexInputBindingDescription,
    pub(super) color_blend_states: Vec<vk::PipelineColorBlendAttachmentState>,
    pub(super) vertex_attribs: Vec<vk::VertexInputAttributeDescription>,
    pub(super) dynamic_states: Vec<vk::DynamicState>,
    pub(super) layout: vk::PipelineLayout,
    pub(super) sample_count: SampleCount,
    pub(super) min_sample_shading: f32,
}

#[derive(Clone, Default)]
pub struct ComputePipeline {
    pub(super) raw: vk::Pipeline,
    pub(super) layout: Handle<ComputePipelineLayout>,
}

#[derive(Clone, Default)]
pub struct GraphicsPipeline {
    pub(super) raw: vk::Pipeline,
    pub(super) render_pass: Handle<RenderPass>,
    pub(super) layout: Handle<GraphicsPipelineLayout>,
    pub(super) subpass: u8,
}
