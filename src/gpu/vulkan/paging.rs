use super::*;

impl VulkanContext {
    /// Allocation requirement for a single-level sampled 2D page, without allocating memory.
    /// Matches make_image with one layer, no storage usage, and one sample.
    pub fn sampled_page_allocation_bytes(&self, extent: [u32; 2], format: Format) -> Result<u64> {
        if extent.contains(&0) || format == Format::D24S8 {
            return Err(GPUError::LibraryError(
                "Invalid sampled page extent or format".into(),
            ));
        }
        let info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(lib_to_vk_image_format(&format))
            .extent(vk::Extent3D {
                width: extent[0],
                height: extent[1],
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        unsafe {
            let image = self
                .device
                .create_image(&info, self.allocation_callbacks.as_deref())?;
            let requirements = self.device.get_image_memory_requirements(image);
            self.device
                .destroy_image(image, self.allocation_callbacks.as_deref());
            Ok(requirements.size)
        }
    }

    pub fn image_allocation_bytes(&self, image: Handle<Image>) -> Result<u64> {
        let image = self.images.get_ref(image).ok_or(GPUError::SlotError())?;
        Ok(self.allocator.get_allocation_info(image.alloc.vma()?).size)
    }

    /// The caller must establish completion of all GPU writers before reading mapped memory.
    pub fn invalidate_buffer(&mut self, view: BufferView) -> Result<()> {
        let buffer = self
            .buffers
            .get_ref(view.handle)
            .ok_or(GPUError::SlotError())?;
        let info = self.allocator.get_allocation_info(buffer.alloc.vma()?);
        let size = if view.size == 0 {
            info.size.saturating_sub(view.offset)
        } else {
            view.size
        };
        if view
            .offset
            .checked_add(size)
            .is_none_or(|end| end > info.size)
        {
            return Err(GPUError::LibraryError("Invalid mapped buffer range".into()));
        }
        self.allocator.invalidate_allocation(
            buffer.alloc.vma()?,
            view.offset as usize,
            size as usize,
        )?;
        Ok(())
    }

    pub fn flush_buffer_range(&mut self, view: BufferView) -> Result<()> {
        let buffer = self
            .buffers
            .get_ref(view.handle)
            .ok_or(GPUError::SlotError())?;
        let info = self.allocator.get_allocation_info(buffer.alloc.vma()?);
        let size = if view.size == 0 {
            info.size.saturating_sub(view.offset)
        } else {
            view.size
        };
        if view
            .offset
            .checked_add(size)
            .is_none_or(|end| end > info.size)
        {
            return Err(GPUError::LibraryError("Invalid mapped buffer range".into()));
        }
        self.allocator.flush_allocation(
            buffer.alloc.vma()?,
            view.offset as usize,
            size as usize,
        )?;
        Ok(())
    }
}
