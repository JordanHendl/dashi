//! A narrow presentation endpoint with no window object or mutable context.
use super::*;
use std::sync::{Arc, Mutex};

/// Captured completion objects. The producer retains them until the consumer retires the frame.
#[derive(Clone, Copy)]
pub struct PresentationSubmission {
    fence: vk::Fence,
    semaphore: vk::Semaphore,
}

pub struct DetachedPresenter {
    device: ash::Device,
    loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    queue: vk::Queue,
    lock: Arc<Mutex<()>>,
    sources: Vec<vk::Image>,
    targets: Vec<vk::Image>,
    extent: vk::Extent3D,
    pool: vk::CommandPool,
    commands: vk::CommandBuffer,
    fence: vk::Fence,
    acquired: vk::Semaphore,
    present: Vec<vk::Semaphore>,
    active: Option<u32>,
    submitted: bool,
}

impl VulkanContext {
    /// Capture the producer's completion objects without transferring ownership.
    ///
    /// # Safety
    /// The fence and semaphore must not be reset, reused or destroyed until the
    /// detached consumer has completed or drained this submission.
    pub unsafe fn presentation_submission(
        &self,
        fence: Handle<Fence>,
        semaphore: Handle<Semaphore>,
    ) -> Result<PresentationSubmission> {
        Ok(PresentationSubmission {
            fence: self.fences.get_ref(fence).ok_or(GPUError::SlotError())?.raw,
            semaphore: self
                .semaphores
                .get_ref(semaphore)
                .ok_or(GPUError::SlotError())?
                .raw,
        })
    }

    pub fn display_image_format(&self, display: &Display) -> Result<Format> {
        Ok(self
            .image_info(*display.images.first().ok_or(GPUError::SlotError())?)
            .format)
    }

    /// Create an independently owned presenter, suitable for a worker thread.
    ///
    /// # Safety
    /// Stop and drop the endpoint before rebuilding/destroying the display,
    /// source images or device. No other code may acquire/present this display
    /// while the endpoint exists. Sources must be single-sampled, match the
    /// swapchain extent/format, and be in TRANSFER_DST_OPTIMAL when submitted.
    /// A source is reusable only after its presentation copy/drain completes.
    pub unsafe fn detached_presenter(
        &self,
        display: &Display,
        sources: &[Handle<Image>],
    ) -> Result<DetachedPresenter> {
        let format = self.display_image_format(display)?;
        let mut raw_sources = Vec::new();
        for source in sources {
            let info = self.image_info(*source);
            if info.format != format
                || info.samples != SampleCount::S1
                || info.dim[..2] != [display.extent.width, display.extent.height]
            {
                return Err(GPUError::LibraryError(
                    "Detached presentation source format/extent mismatch".into(),
                ));
            }
            raw_sources.push(
                self.images
                    .get_ref(*source)
                    .ok_or(GPUError::SlotError())?
                    .img,
            );
        }
        let pool = self.device.create_command_pool(
            &vk::CommandPoolCreateInfo::builder()
                .queue_family_index(self.gfx_queue.family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        )?;
        let mut presenter = DetachedPresenter {
            device: self.device.clone(),
            loader: ash::extensions::khr::Swapchain::new(&self.instance, &self.device),
            swapchain: display.swapchain,
            queue: self.gfx_queue.queue,
            lock: self.submission_lock.clone(),
            sources: raw_sources,
            targets: display
                .images
                .iter()
                .map(|h| self.images.get_ref(*h).unwrap().img)
                .collect(),
            extent: vk::Extent3D {
                width: display.extent.width,
                height: display.extent.height,
                depth: 1,
            },
            pool,
            commands: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            acquired: vk::Semaphore::null(),
            present: Vec::new(),
            active: None,
            submitted: false,
        };
        presenter.commands = self.device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::builder()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        presenter.fence = self
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)?;
        presenter.acquired = self
            .device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
        for _ in &display.images {
            presenter.present.push(
                self.device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?,
            );
        }
        Ok(presenter)
    }
}

impl DetachedPresenter {
    pub fn gpu_complete(&self, submission: PresentationSubmission) -> Result<bool> {
        Ok(unsafe { self.device.get_fence_status(submission.fence)? })
    }

    /// Stage a completed source for presentation. Returns false on bounded acquisition timeout.
    pub fn stage(&mut self, slot: usize, submission: PresentationSubmission) -> Result<bool> {
        if self.active.is_some() {
            return Err(GPUError::LibraryError("Presentation already staged".into()));
        }
        if !self.gpu_complete(submission)? {
            return Ok(false);
        }
        let source = *self.sources.get(slot).ok_or(GPUError::SlotError())?;
        self.finish_copy()?;
        let (index, suboptimal) = match unsafe {
            self.loader.acquire_next_image(
                self.swapchain,
                1_000_000,
                self.acquired,
                vk::Fence::null(),
            )
        } {
            Ok(v) => v,
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => return Ok(false),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Err(GPUError::DisplayNeedsRebuild),
            Err(e) => return Err(e.into()),
        };
        // A suboptimal acquisition still owns an image and a signaled semaphore.
        // Complete that acquisition normally; resize is handled by the window owner.
        let _ = suboptimal;
        self.active = Some(index);
        let target = self.targets[index as usize];
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let barrier = |image, old_layout, new_layout, src_access_mask, dst_access_mask| {
            vk::ImageMemoryBarrier {
                image,
                old_layout,
                new_layout,
                src_access_mask,
                dst_access_mask,
                subresource_range: range,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                ..Default::default()
            }
        };
        unsafe {
            self.device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            self.device.begin_command_buffer(
                self.commands,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.device.cmd_pipeline_barrier(
                self.commands,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    barrier(
                        source,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::TRANSFER_READ,
                    ),
                    barrier(
                        target,
                        vk::ImageLayout::UNDEFINED,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::AccessFlags::empty(),
                        vk::AccessFlags::TRANSFER_WRITE,
                    ),
                ],
            );
            let layers = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            };
            self.device.cmd_copy_image(
                self.commands,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                target,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageCopy {
                    src_subresource: layers,
                    dst_subresource: layers,
                    extent: self.extent,
                    ..Default::default()
                }],
            );
            self.device.cmd_pipeline_barrier(
                self.commands,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    barrier(
                        source,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::AccessFlags::TRANSFER_WRITE,
                    ),
                    barrier(
                        target,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::PRESENT_SRC_KHR,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::AccessFlags::empty(),
                    ),
                ],
            );
            self.device.end_command_buffer(self.commands)?;
            self.device.reset_fences(&[self.fence])?;
            let _guard = self.lock.lock().unwrap();
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[self.commands])
                    .wait_semaphores(&[self.acquired, submission.semaphore])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER; 2])
                    .signal_semaphores(&[self.present[index as usize]])
                    .build()],
                self.fence,
            )?;
            self.submitted = true;
        }
        Ok(true)
    }

    pub fn copy_complete(&self) -> Result<bool> {
        Ok(!self.submitted || unsafe { self.device.get_fence_status(self.fence)? })
    }

    pub fn present(&mut self) -> Result<()> {
        let index = self.active.take().ok_or(GPUError::SlotError())?;
        let _guard = self.lock.lock().unwrap();
        match unsafe {
            self.loader.queue_present(
                self.queue,
                &vk::PresentInfoKHR::builder()
                    .swapchains(&[self.swapchain])
                    .image_indices(&[index])
                    .wait_semaphores(&[self.present[index as usize]]),
            )
        } {
            Ok(_) => Ok(()),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                Err(GPUError::DisplayNeedsRebuild)
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Consume a cancelled producer semaphore without acquiring or presenting an image.
    pub fn drain(&mut self, submission: PresentationSubmission) -> Result<()> {
        self.finish_copy()?;
        unsafe {
            self.device.reset_fences(&[self.fence])?;
            {
                let _guard = self.lock.lock().unwrap();
                self.device.queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::builder()
                        .wait_semaphores(&[submission.semaphore])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_COMMANDS])
                        .build()],
                    self.fence,
                )?;
            }
            self.submitted = true;
        }
        self.finish_copy()
    }

    pub fn finish_copy(&mut self) -> Result<()> {
        if self.submitted {
            unsafe {
                self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
            }
            self.submitted = false;
        }
        Ok(())
    }
}

impl Drop for DetachedPresenter {
    fn drop(&mut self) {
        let _ = self.finish_copy();
        // The caller retires the swapchain after this endpoint. Waiting here also
        // ensures present semaphore waits have completed before destroying them.
        let _guard = self.lock.lock().unwrap();
        unsafe {
            let _ = self.device.queue_wait_idle(self.queue);
            for sem in self.present.drain(..) {
                self.device.destroy_semaphore(sem, None);
            }
            self.device.destroy_semaphore(self.acquired, None);
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn presentation_endpoint_is_send_without_context_or_window() {
        fn assert_send<T: Send>() {}
        assert_send::<DetachedPresenter>();
        assert_send::<PresentationSubmission>();
    }
}
