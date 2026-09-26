//! Backend-neutral mesh shading capabilities and indirect command layout.
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshShaderCapabilities {
    /// These flags describe enabled device features, not merely physical-device support.
    pub mesh_shader: bool,
    pub task_shader: bool,
    pub shader_draw_parameters: bool,
    pub draw_indirect_count: bool,
    pub max_draw_indirect_count: u32,
    pub max_task_work_group_total_count: u32,
    pub max_task_work_group_count: [u32; 3],
    pub max_task_work_group_invocations: u32,
    pub max_task_work_group_size: [u32; 3],
    pub max_task_payload_size: u32,
    pub max_task_shared_memory_size: u32,
    pub max_task_payload_and_shared_memory_size: u32,
    pub max_mesh_work_group_total_count: u32,
    pub max_mesh_work_group_count: [u32; 3],
    pub max_mesh_work_group_invocations: u32,
    pub max_mesh_work_group_size: [u32; 3],
    pub max_mesh_shared_memory_size: u32,
    pub max_mesh_payload_and_shared_memory_size: u32,
    pub max_mesh_output_memory_size: u32,
    pub max_mesh_payload_and_output_memory_size: u32,
    pub max_mesh_output_components: u32,
    pub max_mesh_output_vertices: u32,
    pub max_mesh_output_primitives: u32,
    pub mesh_output_per_vertex_granularity: u32,
    pub mesh_output_per_primitive_granularity: u32,
}

/// Matches VkDrawMeshTasksIndirectCommandEXT; no instance or vertex-buffer fields.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct MeshTasksIndirectCommand {
    pub group_count_x: u32,
    pub group_count_y: u32,
    pub group_count_z: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::command::{
        CommandEncoder, DrawMeshTasks, DrawMeshTasksIndirect, DrawMeshTasksIndirectCount, Op,
    };
    #[test]
    fn mesh_command_layout_and_stream_preserve_all_draw_modes() {
        assert_eq!(std::mem::size_of::<MeshTasksIndirectCommand>(), 12);
        let mut encoder = CommandEncoder::new(crate::QueueType::Graphics);
        encoder.draw_mesh_tasks(&DrawMeshTasks {
            group_count: [3, 2, 1],
            ..Default::default()
        });
        encoder.draw_mesh_tasks_indirect(&DrawMeshTasksIndirect::default());
        encoder.draw_mesh_tasks_indirect_count(&DrawMeshTasksIndirectCount {
            draws: DrawMeshTasksIndirect::default(),
            count: crate::BufferView::new(Default::default()),
        });
        let ops: Vec<_> = encoder.iter().map(|cmd| cmd.op).collect();
        assert_eq!(
            ops,
            [
                Op::DrawMeshTasks,
                Op::DrawMeshTasksIndirect,
                Op::DrawMeshTasksIndirectCount
            ]
        );
    }
}
