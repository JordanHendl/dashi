use super::*;

pub(super) fn capabilities(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
    features: &vk::PhysicalDeviceMeshShaderFeaturesEXT,
    draw: &vk::PhysicalDeviceShaderDrawParametersFeatures,
    indirect_count: bool,
) -> crate::MeshShaderCapabilities {
    if features.mesh_shader != vk::TRUE {
        return Default::default();
    }
    let mut mesh = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
    let mut props = vk::PhysicalDeviceProperties2::builder().push_next(&mut mesh);
    unsafe { instance.get_physical_device_properties2(device, &mut props) };
    let max_draw_indirect_count = props.properties.limits.max_draw_indirect_count;
    crate::MeshShaderCapabilities {
        mesh_shader: true,
        task_shader: features.task_shader == vk::TRUE,
        shader_draw_parameters: draw.shader_draw_parameters == vk::TRUE,
        draw_indirect_count: indirect_count,
        max_draw_indirect_count,
        max_task_work_group_total_count: mesh.max_task_work_group_total_count,
        max_task_work_group_count: mesh.max_task_work_group_count,
        max_task_work_group_invocations: mesh.max_task_work_group_invocations,
        max_task_work_group_size: mesh.max_task_work_group_size,
        max_task_payload_size: mesh.max_task_payload_size,
        max_task_shared_memory_size: mesh.max_task_shared_memory_size,
        max_task_payload_and_shared_memory_size: mesh.max_task_payload_and_shared_memory_size,
        max_mesh_work_group_total_count: mesh.max_mesh_work_group_total_count,
        max_mesh_work_group_count: mesh.max_mesh_work_group_count,
        max_mesh_work_group_invocations: mesh.max_mesh_work_group_invocations,
        max_mesh_work_group_size: mesh.max_mesh_work_group_size,
        max_mesh_shared_memory_size: mesh.max_mesh_shared_memory_size,
        max_mesh_payload_and_shared_memory_size: mesh.max_mesh_payload_and_shared_memory_size,
        max_mesh_output_memory_size: mesh.max_mesh_output_memory_size,
        max_mesh_payload_and_output_memory_size: mesh.max_mesh_payload_and_output_memory_size,
        max_mesh_output_components: mesh.max_mesh_output_components,
        max_mesh_output_vertices: mesh.max_mesh_output_vertices,
        max_mesh_output_primitives: mesh.max_mesh_output_primitives,
        mesh_output_per_vertex_granularity: mesh.mesh_output_per_vertex_granularity,
        mesh_output_per_primitive_granularity: mesh.mesh_output_per_primitive_granularity,
    }
}

impl VulkanContext {
    pub fn mesh_shader_capabilities(&self) -> crate::MeshShaderCapabilities {
        self.mesh_caps
    }
}

pub(super) fn validate_stages(
    info: &GraphicsPipelineLayoutInfo,
    caps: crate::MeshShaderCapabilities,
) -> Result<()> {
    let mut seen = ShaderStageMask::NONE;
    for shader in info.shaders {
        let stage = shader.stage.stage_mask();
        if seen.intersects(stage) {
            return Err(GPUError::LibraryError(
                "Duplicate graphics shader stage".into(),
            ));
        }
        seen |= stage;
    }
    let mesh = seen.contains(ShaderStageMask::MESH);
    if seen.contains(ShaderStageMask::TASK) && (!mesh || !caps.task_shader) {
        return Err(GPUError::UnsupportedShaderStage(ShaderType::Task));
    }
    if mesh && !caps.mesh_shader {
        return Err(GPUError::UnsupportedShaderStage(ShaderType::Mesh));
    }
    let traditional = ShaderStageMask::VERTEX
        | ShaderStageMask::GEOMETRY
        | ShaderStageMask::TESSELLATION_CONTROL
        | ShaderStageMask::TESSELLATION_EVALUATION;
    if mesh && (seen.intersects(traditional) || !info.vertex_info.entries.is_empty()) {
        return Err(GPUError::LibraryError(
            "Mesh pipelines cannot use vertex stages or vertex input".into(),
        ));
    }
    if !mesh && !seen.contains(ShaderStageMask::VERTEX) {
        return Err(GPUError::LibraryError(
            "Graphics pipelines require vertex or mesh stage".into(),
        ));
    }
    Ok(())
}
