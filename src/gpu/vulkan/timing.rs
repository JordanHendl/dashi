use ash::vk;
use crate::GPUError;

pub struct GpuTimer {
    pub(super) pool: vk::QueryPool,
}

impl GpuTimer {
    pub(super) fn new(device: &ash::Device) -> Result<Self, GPUError> {
        let info = vk::QueryPoolCreateInfo::builder()
            .query_count(2)
            .query_type(vk::QueryType::TIMESTAMP)
            .build();
        let pool = unsafe { device.create_query_pool(&info, None)? };
        Ok(Self { pool })
    }

    pub(super) unsafe fn destroy(&self, device: &ash::Device) {
        device.destroy_query_pool(self.pool, None);
    }

    pub(super) unsafe fn begin(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        device.cmd_reset_query_pool(cmd, self.pool, 0, 2);
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, self.pool, 0);
    }

    pub(super) unsafe fn end(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.pool, 1);
    }

    pub(super) fn resolve(&self, device: &ash::Device, period: f32) -> Result<f32, GPUError> {
        let mut data = [0u64; 2];
        unsafe {
            device.get_query_pool_results(
                self.pool,
                0,
                2,
                &mut data,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
        }
        let diff = data[1].saturating_sub(data[0]);
        Ok(diff as f32 * period / 1_000_000.0)
    }
}

