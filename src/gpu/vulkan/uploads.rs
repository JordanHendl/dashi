//! Explicit asynchronous image transfers. A ticket owns staging until GPU completion.
use super::*;
use crate::driver::command::{CopyBufferImage, CopyBufferImageRegion};

#[derive(Clone, Copy, Debug)]
pub struct ImageUploadRegion {
    pub image: Handle<Image>,
    pub range: SubresourceRange,
    pub offset: [u32; 3],
    pub extent: [u32; 3],
}

#[must_use = "submit or discard the batch through its context"]
#[derive(Default)]
pub struct UploadBatch {
    owner: Option<u64>,
    commands: Option<CommandStream<crate::cmd::Executable>>,
    staging: Vec<Handle<Buffer>>,
    bytes: u64,
}

impl UploadBatch {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
    pub fn is_empty(&self) -> bool {
        self.commands.is_none()
    }
    fn append(&mut self, commands: CommandStream<crate::cmd::Executable>) {
        self.commands = Some(match self.commands.take() {
            Some(previous) => previous.combine(commands),
            None => commands,
        });
    }
}

/// Images referenced by a ticket must remain alive until it completes. Dropping an
/// unfinished ticket does not wait or free GPU resources; the owner must poll it.
#[must_use = "poll the ticket to reclaim staging and command resources"]
pub struct UploadTicket {
    owner: u64,
    fence: Option<Handle<Fence>>,
    commands: Option<CommandQueue>,
    staging: Vec<Handle<Buffer>>,
    bytes: u64,
}

impl UploadTicket {
    pub fn completed(&self) -> bool {
        self.fence.is_none()
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

fn invalid(message: &str) -> GPUError {
    GPUError::LibraryError(message.into())
}

impl VulkanContext {
    /// Records a complete supplied mip prefix and generates any remaining levels.
    pub fn stage_image_initial_data(
        &mut self,
        batch: &mut UploadBatch,
        image: Handle<Image>,
        data: &[u8],
    ) -> Result<()> {
        let info = self.image_info(image);
        let (dim, layers, levels, format) = (info.dim, info.layers, info.mip_levels, info.format);
        let mut offset = 0usize;
        let mut regions = Vec::new();
        for level in 0..levels {
            let extent = crate::gpu::mip_dimensions(dim, level);
            let bytes = extent
                .into_iter()
                .chain([layers, channel_count(&format), bytes_per_channel(&format)])
                .try_fold(1usize, |n, size| n.checked_mul(size as usize))
                .ok_or_else(|| invalid("Image byte count overflow"))?;
            let end = offset
                .checked_add(bytes)
                .ok_or_else(|| invalid("Image byte count overflow"))?;
            if end > data.len() {
                return Err(invalid("Initial data is not a complete mip prefix"));
            }
            regions.push((
                ImageUploadRegion {
                    image,
                    range: SubresourceRange::new(level, 1, 0, layers),
                    offset: [0; 3],
                    extent,
                },
                offset..end,
            ));
            offset = end;
            if offset == data.len() {
                break;
            }
        }
        if regions.is_empty() || offset != data.len() {
            return Err(invalid("Initial data is not a complete mip prefix"));
        }
        let provided = regions.len() as u32;
        for (region, bytes) in regions {
            self.stage_image_upload(batch, region, &data[bytes])?;
        }
        self.finish_image_upload(batch, image, provided)
    }

    fn own_upload_batch(&self, batch: &mut UploadBatch) -> Result<()> {
        let owner = self.device.handle().as_raw();
        if batch.owner.is_some_and(|previous| previous != owner) {
            return Err(invalid("Upload batch belongs to another context"));
        }
        batch.owner = Some(owner);
        Ok(())
    }

    /// Copies caller bytes into owned staging without submitting or waiting.
    pub fn stage_image_upload(
        &mut self,
        batch: &mut UploadBatch,
        region: ImageUploadRegion,
        data: &[u8],
    ) -> Result<()> {
        self.own_upload_batch(batch)?;
        let image = self
            .images
            .get_ref(region.image)
            .ok_or(GPUError::SlotError())?;
        let info = &self
            .image_infos
            .get_ref(image.info_handle)
            .ok_or(GPUError::SlotError())?
            .info;
        if info.samples != SampleCount::S1
            || info.format == Format::D24S8
            || region.range.level_count != 1
            || region.range.layer_count == 0
            || region.range.base_mip >= info.mip_levels
            || region
                .range
                .base_layer
                .checked_add(region.range.layer_count)
                .is_none_or(|n| n > info.layers)
        {
            return Err(invalid("Invalid image upload subresource"));
        }
        let dims = crate::gpu::mip_dimensions(info.dim, region.range.base_mip);
        if (0..3).any(|axis| {
            region.extent[axis] == 0
                || region.offset[axis]
                    .checked_add(region.extent[axis])
                    .is_none_or(|n| n > dims[axis])
        }) {
            return Err(invalid("Image upload region exceeds the mip extent"));
        }
        let expected = region
            .extent
            .into_iter()
            .chain([
                region.range.layer_count,
                channel_count(&info.format),
                bytes_per_channel(&info.format),
            ])
            .try_fold(1usize, |size, n| size.checked_mul(n as usize));
        if expected != Some(data.len()) || data.len() > u32::MAX as usize {
            return Err(invalid("Image upload region byte count mismatch"));
        }
        let staging = self.make_buffer(&BufferInfo {
            debug_name: "Image upload staging",
            byte_size: data.len() as u32,
            visibility: MemoryVisibility::CpuAndGpu,
            initial_data: None,
            ..Default::default()
        })?;
        let staged = self
            .map_buffer_mut::<u8>(BufferView::new(staging))
            .map(|mapped| mapped.copy_from_slice(data))
            .and_then(|_| self.flush_buffer(BufferView::new(staging)))
            .and_then(|_| self.unmap_buffer(staging));
        if let Err(error) = staged {
            self.destroy_buffer(staging);
            return Err(error);
        }
        batch.staging.push(staging);
        batch.bytes += data.len() as u64;
        batch.append(
            CommandStream::new()
                .begin()
                .copy_buffer_to_image_region(&CopyBufferImageRegion {
                    copy: CopyBufferImage {
                        src: staging,
                        dst: region.image,
                        range: region.range,
                        src_offset: 0,
                        amount: data.len() as u32,
                    },
                    offset: region.offset,
                    extent: region.extent,
                })
                .end(),
        );
        Ok(())
    }

    /// Completes a fully uploaded mip prefix and transitions all levels for sampling.
    /// Call only after every region in the prefix has been recorded, in queue order.
    pub fn finish_image_upload(
        &mut self,
        batch: &mut UploadBatch,
        image: Handle<Image>,
        provided_levels: u32,
    ) -> Result<()> {
        self.own_upload_batch(batch)?;
        let info = self.image_info(image);
        if provided_levels == 0 || provided_levels > info.mip_levels {
            return Err(invalid("Invalid uploaded mip prefix"));
        }
        let mut stream = CommandStream::new().begin();
        for level in (provided_levels - 1)..info.mip_levels.saturating_sub(1) {
            let src = crate::gpu::mip_dimensions(info.dim, level);
            let dst = crate::gpu::mip_dimensions(info.dim, level + 1);
            stream = stream.blit_images(&crate::driver::command::BlitImage {
                src: image,
                dst: image,
                src_range: SubresourceRange::new(level, 1, 0, info.layers),
                dst_range: SubresourceRange::new(level + 1, 1, 0, info.layers),
                filter: Filter::Linear,
                src_region: Rect2D {
                    x: 0,
                    y: 0,
                    w: src[0],
                    h: src[1],
                },
                dst_region: Rect2D {
                    x: 0,
                    y: 0,
                    w: dst[0],
                    h: dst[1],
                },
            });
        }
        stream.prepare_image_for(image, crate::ResourceUse::Sampled);
        batch.append(stream.end());
        Ok(())
    }

    pub fn discard_upload_batch(&mut self, batch: UploadBatch) -> Result<()> {
        if batch
            .owner
            .is_some_and(|owner| owner != self.device.handle().as_raw())
        {
            return Err(invalid("Upload batch belongs to another context"));
        }
        for buffer in batch.staging {
            self.destroy_buffer(buffer);
        }
        Ok(())
    }

    /// One submission for every image/region in the batch. Never waits for the GPU.
    pub fn submit_upload_batch(&mut self, mut batch: UploadBatch) -> Result<UploadTicket> {
        self.own_upload_batch(&mut batch)?;
        let owner = batch.owner.unwrap();
        let Some(stream) = batch.commands.take() else {
            return Ok(UploadTicket {
                owner,
                fence: None,
                commands: None,
                staging: Vec::new(),
                bytes: 0,
            });
        };
        let ptr = self as *mut _;
        self.gfx_pool.bind_context(ptr);
        let mut commands = match self.gfx_pool.begin("Image upload batch", false) {
            Ok(commands) => commands,
            Err(error) => {
                self.discard_upload_batch(batch)?;
                return Err(error);
            }
        };
        let result = stream
            .append(&mut commands)
            .and_then(|_| self.submit(&mut commands, &Default::default()));
        match result {
            Ok(fence) => Ok(UploadTicket {
                owner,
                fence: Some(fence),
                commands: Some(commands),
                staging: batch.staging,
                bytes: batch.bytes,
            }),
            Err(error) => {
                self.destroy_cmd_queue(commands);
                self.discard_upload_batch(batch)?;
                Err(error)
            }
        }
    }

    /// Nonblocking, repeatable completion check. Completion reclaims staging and commands.
    pub fn poll_upload(&mut self, ticket: &mut UploadTicket) -> Result<bool> {
        if ticket.owner != self.device.handle().as_raw() {
            return Err(invalid("Upload ticket belongs to another context"));
        }
        if let Some(fence) = ticket.fence {
            if !self.poll_fence(fence)? {
                return Ok(false);
            }
            ticket.fence = None;
            if let Some(commands) = ticket.commands.take() {
                self.destroy_cmd_queue(commands);
            }
            for buffer in ticket.staging.drain(..) {
                self.destroy_buffer(buffer);
            }
        }
        Ok(true)
    }

    /// Explicit blocking compatibility path; runtime loaders should use `poll_upload`.
    pub fn wait_upload(&mut self, ticket: &mut UploadTicket) -> Result<()> {
        if ticket.owner != self.device.handle().as_raw() {
            return Err(invalid("Upload ticket belongs to another context"));
        }
        if let Some(fence) = ticket.fence {
            self.wait(fence)?;
        }
        self.poll_upload(ticket).map(|_| ())
    }
}
