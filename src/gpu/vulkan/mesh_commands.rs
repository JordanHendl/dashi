use super::*;
use crate::driver::command::{DrawMeshTasks, DrawMeshTasksIndirect};
use crate::{BufferView, ShaderType};

fn invalid(message: &str) -> GPUError {
    GPUError::LibraryError(message.into())
}

impl CommandQueue {
    fn prepare_mesh(
        &mut self,
        tables: &[Option<Handle<BindTable>>; 4],
        dynamic: &[Option<DynamicBuffer>; 4],
    ) -> Result<bool> {
        if self.ctx_ref().mesh_loader.is_none() {
            return Err(GPUError::UnsupportedShaderStage(ShaderType::Mesh));
        }
        self.ensure_active_render_pass()?;
        let pipeline = self
            .curr_pipeline
            .ok_or_else(|| invalid("Mesh draw requires a graphics pipeline"))?;
        let ctx = self.ctx_ref();
        let pipeline = ctx
            .gfx_pipelines
            .get_ref(pipeline)
            .ok_or(GPUError::SlotError())?;
        let layout = ctx
            .gfx_pipeline_layouts
            .get_ref(pipeline.layout)
            .ok_or(GPUError::SlotError())?;
        if !layout
            .shader_stages
            .iter()
            .any(|s| s.stage == vk::ShaderStageFlags::MESH_EXT)
        {
            return Err(invalid("Mesh draw requires a mesh pipeline"));
        }
        let task = layout
            .shader_stages
            .iter()
            .any(|s| s.stage == vk::ShaderStageFlags::TASK_EXT);
        let raw_layout = layout.layout;
        self.ensure_binding_states(
            tables,
            ShaderStageMask::TASK | ShaderStageMask::MESH | ShaderStageMask::FRAGMENT,
        )?;
        self.bind_descriptor_tables(vk::PipelineBindPoint::GRAPHICS, raw_layout, tables, dynamic)?;
        Ok(task)
    }

    pub(super) fn record_mesh_tasks(&mut self, cmd: &DrawMeshTasks) -> Result<()> {
        let task = self.prepare_mesh(&cmd.bind_tables, &cmd.dynamic_buffers)?;
        let caps = self.ctx_ref().mesh_caps;
        let (limits, total) = if task {
            (
                caps.max_task_work_group_count,
                caps.max_task_work_group_total_count,
            )
        } else {
            (
                caps.max_mesh_work_group_count,
                caps.max_mesh_work_group_total_count,
            )
        };
        let count = cmd.group_count;
        let product = count.iter().try_fold(1u64, |a, &b| a.checked_mul(b as u64));
        if count.iter().zip(limits).any(|(&n, max)| n > max)
            || product.is_none_or(|n| n > total as u64)
        {
            return Err(invalid("Mesh draw workgroup count exceeds device limits"));
        }
        unsafe {
            self.ctx_ref()
                .mesh_loader
                .as_ref()
                .unwrap()
                .cmd_draw_mesh_tasks(self.cmd_buf, count[0], count[1], count[2]);
        }
        self.update_last_access(
            vk::PipelineStageFlags::ALL_GRAPHICS,
            vk::AccessFlags::SHADER_READ,
        );
        Ok(())
    }

    pub(super) fn record_mesh_tasks_indirect(
        &mut self,
        cmd: &DrawMeshTasksIndirect,
        count: Option<BufferView>,
    ) -> Result<()> {
        let caps = self.ctx_ref().mesh_caps;
        if count.is_some() && !caps.draw_indirect_count {
            return Err(invalid("Indirect draw count is not enabled"));
        }
        if cmd.draw_count > caps.max_draw_indirect_count || cmd.stride < 12 || cmd.stride % 4 != 0 {
            return Err(invalid("Invalid mesh indirect draw count or stride"));
        }
        let bytes = if cmd.draw_count == 0 {
            0
        } else {
            (cmd.draw_count as u64 - 1) * cmd.stride as u64 + 12
        };
        let check_view = |view: BufferView, bytes: u64| -> Result<()> {
            let buffer = self
                .ctx_ref()
                .buffers
                .get_ref(view.handle)
                .ok_or(GPUError::SlotError())?;
            if (view.size != 0 && bytes > view.size)
                || view.offset % 4 != 0
                || view
                    .offset
                    .checked_add(bytes)
                    .is_none_or(|end| end > buffer.size as u64)
            {
                return Err(invalid(
                    "Mesh indirect buffer range is out of bounds or unaligned",
                ));
            }
            Ok(())
        };
        check_view(cmd.indirect, bytes)?;
        if let Some(count) = count {
            check_view(count, 4)?;
        }
        self.ensure_buffer_state(cmd.indirect.handle, UsageBits::INDIRECT_READ)?;
        if let Some(count) = count {
            self.ensure_buffer_state(count.handle, UsageBits::INDIRECT_READ)?;
        }
        self.prepare_mesh(&cmd.bind_tables, &cmd.dynamic_buffers)?;
        let ctx = self.ctx_ref();
        let indirect = ctx
            .buffers
            .get_ref(cmd.indirect.handle)
            .ok_or(GPUError::SlotError())?;
        let loader = ctx.mesh_loader.as_ref().unwrap();
        unsafe {
            if let Some(view) = count {
                let count = ctx
                    .buffers
                    .get_ref(view.handle)
                    .ok_or(GPUError::SlotError())?;
                loader.cmd_draw_mesh_tasks_indirect_count(
                    self.cmd_buf,
                    indirect.buf,
                    indirect.offset as u64 + cmd.indirect.offset,
                    count.buf,
                    count.offset as u64 + view.offset,
                    cmd.draw_count,
                    cmd.stride,
                );
            } else {
                loader.cmd_draw_mesh_tasks_indirect(
                    self.cmd_buf,
                    indirect.buf,
                    indirect.offset as u64 + cmd.indirect.offset,
                    cmd.draw_count,
                    cmd.stride,
                );
            }
        }
        self.update_last_access(
            vk::PipelineStageFlags::ALL_GRAPHICS,
            vk::AccessFlags::SHADER_READ,
        );
        Ok(())
    }
}
