use super::*;

#[allow(dead_code)]
#[derive(Debug)]
pub struct BindTableLayout {
    pub(super) pool: vk::DescriptorPool,
    pub(super) layout: vk::DescriptorSetLayout,
    pub(super) variables: Vec<BindTableVariable>,
    pub(super) requirements: Vec<NormalizedBinding>,
    pub(super) update_after_bind: bool,
    pub(super) partially_bound: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct BoundBufferRequirement {
    pub(super) buffer: Handle<Buffer>,
    pub(super) read_usage: UsageBits,
    pub(super) write_usage: UsageBits,
    pub(super) read_stages: ShaderStageMask,
    pub(super) write_stages: ShaderStageMask,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct BindTable {
    pub(super) set: vk::DescriptorSet,
    pub(super) set_id: u32,
    pub(super) layout: Handle<BindTableLayout>,
    pub(super) bound_buffers: HashMap<(u32, u32), BoundBufferRequirement>,
    pub(super) buffer_states: Vec<BoundBufferRequirement>,
}

impl CommandQueue {
    #[allow(dead_code)]
    pub(crate) fn bind_descriptor_set(
        &mut self,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        table: Option<Handle<BindTable>>,
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
                    offsets,
                );
            }
        }
    }
}
