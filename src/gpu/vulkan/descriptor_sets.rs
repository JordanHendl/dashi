use super::*;

#[allow(dead_code)]
#[derive(Debug)]
pub struct BindGroupLayout {
    pub(super) pool: vk::DescriptorPool,
    pub(super) layout: vk::DescriptorSetLayout,
    pub(super) variables: Vec<BindGroupVariable>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct BindGroup {
    pub(super) set: vk::DescriptorSet,
    pub(super) set_id: u32,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct BindTableLayout {
    pub(super) pool: vk::DescriptorPool,
    pub(super) layout: vk::DescriptorSetLayout,
    pub(super) variables: Vec<BindGroupVariable>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct BindTable {
    pub(super) set: vk::DescriptorSet,
    pub(super) set_id: u32,
    pub(super) layout: Handle<BindTableLayout>,
}

impl CommandQueue {
    pub(crate) fn bind_descriptor_set(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        table: Option<Handle<BindTable>>,
        group: Option<Handle<BindGroup>>,
        offsets: &[u32],
    ) {
        unsafe {
            if let Some(bt) = table {
                let bt_data = self.ctx_ref().bind_tables.get_ref(bt).unwrap();
                self.ctx_ref().device.cmd_bind_descriptor_sets(
                    self.cmd_buf,
                    bind_point,
                    layout,
                    bt_data.set_id,
                    &[bt_data.set],
                    &[],
                );
            } else if let Some(bg) = group {
                let bg_data = self.ctx_ref().bind_groups.get_ref(bg).unwrap();
                self.ctx_ref().device.cmd_bind_descriptor_sets(
                    self.cmd_buf,
                    bind_point,
                    layout,
                    bg_data.set_id,
                    &[bg_data.set],
                    offsets,
                );
            }
        }
    }
}
