use crate::gpu::vulkan::ContextLimits;

#[derive(Debug, Clone, Copy)]
pub struct WebGpuLimits {
    pub max_bind_groups: u32,
    pub max_samplers_per_shader_stage: u32,
    pub max_sampled_textures_per_shader_stage: u32,
    pub max_storage_textures_per_shader_stage: u32,
    pub max_uniform_buffer_binding_size: u32,
    pub max_storage_buffer_binding_size: u32,
    pub max_push_constant_size: u32,
    pub max_color_attachments: u32,
}

impl Default for WebGpuLimits {
    fn default() -> Self {
        Self {
            max_bind_groups: 0,
            max_samplers_per_shader_stage: 0,
            max_sampled_textures_per_shader_stage: 0,
            max_storage_textures_per_shader_stage: 0,
            max_uniform_buffer_binding_size: 0,
            max_storage_buffer_binding_size: 0,
            max_push_constant_size: 0,
            max_color_attachments: 0,
        }
    }
}

impl From<WebGpuLimits> for ContextLimits {
    fn from(limits: WebGpuLimits) -> Self {
        Self {
            max_sampler_array_len: limits.max_samplers_per_shader_stage,
            max_sampled_texture_array_len: limits.max_sampled_textures_per_shader_stage,
            max_storage_texture_array_len: limits.max_storage_textures_per_shader_stage,
            max_uniform_buffer_range: limits.max_uniform_buffer_binding_size,
            max_storage_buffer_range: limits.max_storage_buffer_binding_size,
            max_push_constant_size: limits.max_push_constant_size,
            max_color_attachments: limits.max_color_attachments,
            max_bound_bind_tables: limits.max_bind_groups,
        }
    }
}
