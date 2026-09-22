use crate::GPUError;
use ash::vk;

pub struct GpuTimer {
    pub(super) pool: vk::QueryPool,
    state: TimerState,
}

impl GpuTimer {
    pub(super) fn new(
        device: &ash::Device,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) -> Result<Self, GPUError> {
        let info = vk::QueryPoolCreateInfo::builder()
            .query_count(2)
            .query_type(vk::QueryType::TIMESTAMP)
            .build();
        let pool = unsafe { device.create_query_pool(&info, allocation_callbacks)? };
        Ok(Self {
            pool,
            state: TimerState::Uninitialized,
        })
    }

    pub(super) unsafe fn destroy(
        &self,
        device: &ash::Device,
        allocation_callbacks: Option<&vk::AllocationCallbacks>,
    ) {
        device.destroy_query_pool(self.pool, allocation_callbacks);
    }

    pub(super) unsafe fn begin(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, self.pool, 0);
        self.state = TimerState::Begun;
    }

    pub(super) unsafe fn end(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.pool, 1);
        if self.state == TimerState::Begun {
            self.state = TimerState::Ended;
        }
    }

    pub(super) fn resolve(&mut self, device: &ash::Device, period: f32) -> Result<f32, GPUError> {
        if self.state != TimerState::Ended {
            return Err(GPUError::LibraryError(
                "GPU timer queries have not been initialized or ended yet.".to_string(),
            ));
        }
        let mut data = [0u64; 2];
        unsafe {
            device.get_query_pool_results(
                self.pool,
                0,
                2,
                &mut data,
                vk::QueryResultFlags::TYPE_64,
            )?;
        }
        let diff = data[1].saturating_sub(data[0]);
        Ok(diff as f32 * period / 1_000_000.0)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum TimerState {
    Uninitialized,
    Begun,
    Ended,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::vulkan::{
        DebugMessageSeverity, DebugMessageType, DebugMessengerCreateInfo, VulkanContext,
    };
    use crate::{ContextInfo, QueueType};
    use std::sync::atomic::{AtomicBool, Ordering};

    unsafe extern "system" fn validation_error(
        severity: DebugMessageSeverity,
        kind: DebugMessageType,
        _message: &std::ffi::CStr,
        data: *mut std::ffi::c_void,
    ) -> bool {
        if severity.contains(DebugMessageSeverity::ERROR)
            && kind.contains(DebugMessageType::VALIDATION)
        {
            (*(data as *const AtomicBool)).store(true, Ordering::SeqCst);
        }
        false
    }

    #[test]
    #[serial_test::serial]
    fn unavailable_timestamp_pair_does_not_wait_for_gpu_work() {
        let mut ctx = VulkanContext::headless(&ContextInfo::default()).unwrap();
        let error = AtomicBool::new(false);
        let messenger = ctx
            .create_debug_messenger(&DebugMessengerCreateInfo {
                message_severity: DebugMessageSeverity::ERROR,
                message_type: DebugMessageType::VALIDATION,
                user_callback: validation_error,
                user_data: &error as *const AtomicBool as *mut std::ffi::c_void,
            })
            .unwrap();
        ctx.init_gpu_timers(1).unwrap();
        let mut reset = ctx
            .pool_mut(QueueType::Graphics)
            .begin("unavailable timer", false)
            .unwrap();
        unsafe {
            ctx.device
                .cmd_reset_query_pool(reset.cmd_buf, ctx.gpu_timers[0].pool, 0, 2);
        }
        let fence = ctx.submit(&mut reset, &Default::default()).unwrap();
        ctx.wait(fence).unwrap();
        // Recording an end changes CPU state before its timestamps execute on the GPU.
        ctx.gpu_timers[0].state = TimerState::Ended;
        let started = std::time::Instant::now();
        assert!(ctx.get_elapsed_gpu_time_ms(0).is_none());
        assert!(started.elapsed() < std::time::Duration::from_secs(1));
        let mut measured = ctx
            .pool_mut(QueueType::Graphics)
            .begin("completed timer", false)
            .unwrap();
        ctx.gpu_timer_begin(&mut measured, 0);
        ctx.gpu_timer_end(&mut measured, 0);
        let fence = ctx.submit(&mut measured, &Default::default()).unwrap();
        ctx.wait(fence).unwrap();
        assert!(ctx.get_elapsed_gpu_time_ms(0).unwrap() >= 0.0);
        ctx.destroy_cmd_queue(reset);
        ctx.destroy_cmd_queue(measured);
        ctx.destroy_debug_messenger(messenger);
        ctx.destroy();
        assert!(
            !error.load(Ordering::SeqCst),
            "Vulkan validation reported an error"
        );
    }
}
