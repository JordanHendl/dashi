use super::*;
use crate::gpu::external::*;

pub(super) fn extension_names() -> [&'static CStr; 2] {
    #[cfg(windows)]
    {
        [
            ash::extensions::khr::ExternalMemoryWin32::name(),
            ash::extensions::khr::ExternalSemaphoreWin32::name(),
        ]
    }
    #[cfg(unix)]
    {
        [
            ash::extensions::khr::ExternalMemoryFd::name(),
            ash::extensions::khr::ExternalSemaphoreFd::name(),
        ]
    }
}

pub(super) fn enabled_capabilities(names: &[*const c_char]) -> ExternalResourceCapabilities {
    let extensions = extension_names();
    let has = |name| names.iter().any(|p| unsafe { CStr::from_ptr(*p) == name });
    ExternalResourceCapabilities {
        memory: has(extensions[0]),
        binary_semaphore: has(extensions[1]),
    }
}

fn memory_type() -> vk::ExternalMemoryHandleTypeFlags {
    #[cfg(windows)]
    {
        vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32
    }
    #[cfg(unix)]
    {
        vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD
    }
}
fn semaphore_type() -> vk::ExternalSemaphoreHandleTypeFlags {
    #[cfg(windows)]
    {
        vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32
    }
    #[cfg(unix)]
    {
        vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD
    }
}
fn invalid(message: &str) -> GPUError {
    GPUError::LibraryError(message.to_owned())
}

pub(super) fn device_identity(
    instance: &ash::Instance,
    device: vk::PhysicalDevice,
) -> DeviceIdentity {
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut properties = vk::PhysicalDeviceProperties2::builder().push_next(&mut id);
    unsafe { instance.get_physical_device_properties2(device, &mut properties) };
    DeviceIdentity {
        device_uuid: id.device_uuid,
        driver_uuid: id.driver_uuid,
        device_luid: id.device_luid,
        luid_valid: id.device_luid_valid,
    }
}

fn image_usage(info: &ImageInfo) -> Result<vk::ImageUsageFlags> {
    if info.dim[0] == 0
        || info.dim[1] == 0
        || info.dim[2] != 1
        || info.layers != 1
        || info.mip_levels != 1
        || info.samples != SampleCount::S1
        || info.cube_compatible
        || info.format == Format::D24S8
    {
        return Err(invalid("External images require nonempty 2D single-layer, single-mip, single-sample color images"));
    }
    Ok(vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::COLOR_ATTACHMENT
        | if info.storage {
            vk::ImageUsageFlags::STORAGE
        } else {
            vk::ImageUsageFlags::empty()
        })
}
fn buffer_usage(info: &BufferInfo) -> Result<vk::BufferUsageFlags> {
    if info.byte_size == 0 || !matches!(info.visibility, MemoryVisibility::Gpu) {
        return Err(invalid(
            "External buffers must be nonempty and device-local",
        ));
    }
    let mut usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
    for (bit, flag) in [
        (BufferUsage::VERTEX, vk::BufferUsageFlags::VERTEX_BUFFER),
        (BufferUsage::INDEX, vk::BufferUsageFlags::INDEX_BUFFER),
        (BufferUsage::UNIFORM, vk::BufferUsageFlags::UNIFORM_BUFFER),
        (BufferUsage::STORAGE, vk::BufferUsageFlags::STORAGE_BUFFER),
        (BufferUsage::INDIRECT, vk::BufferUsageFlags::INDIRECT_BUFFER),
    ] {
        if info.usage.contains(bit) {
            usage |= flag;
        }
    }
    Ok(usage)
}

impl VulkanContext {
    pub fn device_identity(&self) -> DeviceIdentity {
        device_identity(&self.instance, self.pdevice)
    }
    pub fn external_resource_capabilities(&self) -> ExternalResourceCapabilities {
        self.external_caps
    }

    pub(super) fn resource_allocation_size(&self, allocation: &Allocation) -> (u64, bool) {
        match allocation {
            Allocation::Vma(a) => self.allocation_size_and_locality(a),
            Allocation::External { info, .. } => (info.allocation_size, true),
            Allocation::Borrowed => (0, false),
        }
    }
    fn record_external_allocation(&mut self, size: u64) {
        let activity = &mut self.allocation_activity;
        activity.created += 1;
        activity.bytes_created += size;
        activity.local_bytes += size;
        activity.peak_local_bytes = activity.peak_local_bytes.max(activity.local_bytes);
    }
    fn check_device(&self, id: DeviceIdentity) -> Result<()> {
        let own = self.device_identity();
        if own.device_uuid != id.device_uuid || own.driver_uuid != id.driver_uuid {
            return Err(invalid(
                "External resource belongs to a different Vulkan GPU or driver",
            ));
        }
        Ok(())
    }
    fn check_memory(
        &self,
        requirements: vk::MemoryRequirements,
        info: ExternalMemoryInfo,
    ) -> Result<()> {
        self.check_device(info.device)?;
        if info.handle_type != memory_type().as_raw()
            || info.allocation_size < requirements.size
            || info.memory_type_index >= 32
            || requirements.memory_type_bits & (1 << info.memory_type_index) == 0
        {
            return Err(invalid(
                "External allocation does not match resource memory requirements",
            ));
        }
        Ok(())
    }
    fn memory_info(&self, requirements: vk::MemoryRequirements) -> Result<ExternalMemoryInfo> {
        let properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.pdevice)
        };
        let index = (0..properties.memory_type_count)
            .find(|i| {
                requirements.memory_type_bits & (1 << i) != 0
                    && properties.memory_types[*i as usize]
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or_else(|| invalid("No compatible device-local external memory type"))?;
        Ok(ExternalMemoryInfo {
            device: self.device_identity(),
            allocation_size: requirements.size,
            memory_type_index: index,
            handle_type: memory_type().as_raw(),
        })
    }
    fn create_shared_image(&self, info: &ImageInfo, importing: bool) -> Result<vk::Image> {
        if !self.external_caps.memory {
            return Err(invalid("Vulkan opaque external memory is unavailable"));
        }
        let usage = image_usage(info)?;
        let mut external_query =
            vk::PhysicalDeviceExternalImageFormatInfo::builder().handle_type(memory_type());
        let query = vk::PhysicalDeviceImageFormatInfo2::builder()
            .format(lib_to_vk_image_format(&info.format))
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .push_next(&mut external_query);
        let mut external_properties = vk::ExternalImageFormatProperties::default();
        let mut properties =
            vk::ImageFormatProperties2::builder().push_next(&mut external_properties);
        unsafe {
            self.instance.get_physical_device_image_format_properties2(
                self.pdevice,
                &query,
                &mut properties,
            )
        }?;
        let required = if importing {
            vk::ExternalMemoryFeatureFlags::IMPORTABLE
        } else {
            vk::ExternalMemoryFeatureFlags::EXPORTABLE
        };
        if !external_properties
            .external_memory_properties
            .external_memory_features
            .contains(required)
        {
            return Err(invalid(
                "Image format/usage does not support external memory",
            ));
        }
        let mut external = vk::ExternalMemoryImageCreateInfo::builder().handle_types(memory_type());
        let create = vk::ImageCreateInfo::builder()
            .push_next(&mut external)
            .image_type(vk::ImageType::TYPE_2D)
            .format(lib_to_vk_image_format(&info.format))
            .extent(vk::Extent3D {
                width: info.dim[0],
                height: info.dim[1],
                depth: 1,
            })
            .array_layers(1)
            .mip_levels(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        Ok(unsafe { self.device.create_image(&create, None) }?)
    }
    fn create_shared_buffer(&self, info: &BufferInfo, importing: bool) -> Result<vk::Buffer> {
        if !self.external_caps.memory {
            return Err(invalid("Vulkan opaque external memory is unavailable"));
        }
        let usage = buffer_usage(info)?;
        let query = vk::PhysicalDeviceExternalBufferInfo::builder()
            .usage(usage)
            .handle_type(memory_type());
        let mut properties = vk::ExternalBufferProperties::default();
        unsafe {
            self.instance
                .get_physical_device_external_buffer_properties(
                    self.pdevice,
                    &query,
                    &mut properties,
                )
        };
        let required = if importing {
            vk::ExternalMemoryFeatureFlags::IMPORTABLE
        } else {
            vk::ExternalMemoryFeatureFlags::EXPORTABLE
        };
        if !properties
            .external_memory_properties
            .external_memory_features
            .contains(required)
        {
            return Err(invalid("Buffer usage does not support external memory"));
        }
        let mut external =
            vk::ExternalMemoryBufferCreateInfo::builder().handle_types(memory_type());
        let create = vk::BufferCreateInfo::builder()
            .size(info.byte_size as u64)
            .usage(usage)
            .push_next(&mut external);
        Ok(unsafe { self.device.create_buffer(&create, None) }?)
    }
    fn allocate_export(
        &self,
        info: ExternalMemoryInfo,
        image: vk::Image,
        buffer: vk::Buffer,
    ) -> Result<vk::DeviceMemory> {
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::builder()
            .image(image)
            .buffer(buffer);
        let mut export = vk::ExportMemoryAllocateInfo::builder().handle_types(memory_type());
        let allocate = vk::MemoryAllocateInfo::builder()
            .allocation_size(info.allocation_size)
            .memory_type_index(info.memory_type_index)
            .push_next(&mut dedicated)
            .push_next(&mut export);
        Ok(unsafe { self.device.allocate_memory(&allocate, None) }?)
    }
    unsafe fn allocate_import<T>(
        &self,
        info: ExternalMemoryInfo,
        handle: OSHandle<T>,
        image: vk::Image,
        buffer: vk::Buffer,
    ) -> Result<vk::DeviceMemory> {
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::builder()
            .image(image)
            .buffer(buffer);
        #[cfg(windows)]
        let mut import = {
            use std::os::windows::io::AsRawHandle;
            vk::ImportMemoryWin32HandleInfoKHR::builder()
                .handle_type(memory_type())
                .handle(handle.inner.as_raw_handle())
        };
        #[cfg(unix)]
        let mut import = {
            use std::os::fd::AsRawFd;
            vk::ImportMemoryFdInfoKHR::builder()
                .handle_type(memory_type())
                .fd(handle.inner.as_raw_fd())
        };
        let allocate = vk::MemoryAllocateInfo::builder()
            .allocation_size(info.allocation_size)
            .memory_type_index(info.memory_type_index)
            .push_next(&mut dedicated)
            .push_next(&mut import);
        let memory = unsafe { self.device.allocate_memory(&allocate, None) }?;
        #[cfg(unix)]
        {
            use std::os::fd::IntoRawFd;
            let _ = handle.inner.into_raw_fd();
        }
        Ok(memory)
    }
    fn insert_external_image(
        &mut self,
        image: vk::Image,
        memory: vk::DeviceMemory,
        metadata: ExternalMemoryInfo,
        info: &ImageInfo,
        exportable: bool,
    ) -> Result<Handle<Image>> {
        let result = (|| {
            unsafe { self.device.bind_image_memory(image, memory, 0) }?;
            let info_handle = self
                .image_infos
                .insert(ImageInfoRecord::new(info))
                .ok_or(GPUError::SlotError())?;
            let result = self
                .images
                .insert(Image {
                    img: image,
                    alloc: Allocation::External {
                        memory,
                        info: metadata,
                        exportable,
                    },
                    layouts: vec![if exportable {
                        vk::ImageLayout::UNDEFINED
                    } else {
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL
                    }],
                    externally_owned: !exportable,
                    info_handle,
                })
                .ok_or(GPUError::SlotError());
            if result.is_err() {
                self.image_infos.release(info_handle);
            }
            result
        })();
        if result.is_err() {
            unsafe {
                self.device.destroy_image(image, None);
                self.device.free_memory(memory, None);
            }
        } else {
            self.record_external_allocation(metadata.allocation_size);
        }
        result
    }
    fn insert_external_buffer(
        &mut self,
        buffer: vk::Buffer,
        memory: vk::DeviceMemory,
        metadata: ExternalMemoryInfo,
        info: &BufferInfo,
        exportable: bool,
    ) -> Result<Handle<Buffer>> {
        let result = (|| {
            unsafe { self.device.bind_buffer_memory(buffer, memory, 0) }?;
            let info_handle = self
                .buffer_infos
                .insert(BufferInfoRecord::new(info))
                .ok_or(GPUError::SlotError())?;
            let result = self
                .buffers
                .insert(Buffer {
                    buf: buffer,
                    alloc: Allocation::External {
                        memory,
                        info: metadata,
                        exportable,
                    },
                    size: info.byte_size,
                    offset: 0,
                    suballocated: false,
                    externally_owned: !exportable,
                    info_handle,
                })
                .ok_or(GPUError::SlotError());
            if result.is_err() {
                self.buffer_infos.release(info_handle);
            }
            result
        })();
        if result.is_err() {
            unsafe {
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
        } else {
            self.record_external_allocation(metadata.allocation_size);
        }
        result
    }
    pub fn make_external_image(&mut self, info: &ImageInfo) -> Result<Handle<Image>> {
        let image = self.create_shared_image(info, false)?;
        let result = (|| {
            let requirements = unsafe { self.device.get_image_memory_requirements(image) };
            let metadata = self.memory_info(requirements)?;
            let memory = self.allocate_export(metadata, image, vk::Buffer::null())?;
            Ok((metadata, memory))
        })();
        let (metadata, memory) = match result {
            Ok(r) => r,
            Err(e) => {
                unsafe {
                    self.device.destroy_image(image, None);
                }
                return Err(e);
            }
        };
        let handle = self.insert_external_image(image, memory, metadata, info, true)?;
        if let Err(e) = self.init_image(handle, info) {
            self.destroy_image(handle);
            return Err(e);
        }
        Ok(handle)
    }
    pub fn make_external_buffer(&mut self, info: &BufferInfo) -> Result<Handle<Buffer>> {
        let buffer = self.create_shared_buffer(info, false)?;
        let result = (|| {
            let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
            let metadata = self.memory_info(requirements)?;
            let memory = self.allocate_export(metadata, vk::Image::null(), buffer)?;
            Ok((metadata, memory))
        })();
        let (metadata, memory) = match result {
            Ok(r) => r,
            Err(e) => {
                unsafe {
                    self.device.destroy_buffer(buffer, None);
                }
                return Err(e);
            }
        };
        let handle = self.insert_external_buffer(buffer, memory, metadata, info, true)?;
        if let Err(e) = self.init_buffer(handle, info) {
            self.destroy_buffer(handle);
            return Err(e);
        }
        Ok(handle)
    }
    fn export_memory<T>(
        &self,
        allocation: &Allocation,
    ) -> Result<(OSHandle<T>, ExternalMemoryInfo)> {
        let Allocation::External {
            memory,
            info,
            exportable: true,
        } = allocation
        else {
            return Err(invalid("Resource was not created as exportable"));
        };
        #[cfg(windows)]
        let handle = {
            use std::os::windows::io::{FromRawHandle, OwnedHandle};
            let loader =
                ash::extensions::khr::ExternalMemoryWin32::new(&self.instance, &self.device);
            let query = vk::MemoryGetWin32HandleInfoKHR::builder()
                .memory(*memory)
                .handle_type(memory_type());
            unsafe {
                OSHandle::new(OwnedHandle::from_raw_handle(
                    loader.get_memory_win32_handle(&query)?,
                ))
            }
        };
        #[cfg(unix)]
        let handle = {
            use std::os::fd::{FromRawFd, OwnedFd};
            let loader = ash::extensions::khr::ExternalMemoryFd::new(&self.instance, &self.device);
            let query = vk::MemoryGetFdInfoKHR::builder()
                .memory(*memory)
                .handle_type(memory_type());
            unsafe { OSHandle::new(OwnedFd::from_raw_fd(loader.get_memory_fd(&query)?)) }
        };
        Ok((handle, *info))
    }
    pub fn export_image(&self, image: Handle<Image>) -> Result<ExportedImage> {
        let resource = self.images.get_ref(image).ok_or(GPUError::SlotError())?;
        let (handle, memory) = self.export_memory(&resource.alloc)?;
        let info = self.image_info(image);
        Ok(ExportedImage {
            handle,
            descriptor: ExternalImageDescriptor {
                memory,
                extent: info.dim,
                format: lib_to_vk_image_format(&info.format).as_raw(),
                usage: image_usage(info)?.as_raw(),
            },
        })
    }
    pub fn export_buffer(&self, buffer: Handle<Buffer>) -> Result<ExportedBuffer> {
        let resource = self.buffers.get_ref(buffer).ok_or(GPUError::SlotError())?;
        if resource.suballocated {
            return Err(invalid("Cannot export a buffer suballocation"));
        }
        let (handle, memory) = self.export_memory(&resource.alloc)?;
        let info = self.buffer_info(buffer);
        Ok(ExportedBuffer {
            handle,
            descriptor: ExternalBufferDescriptor {
                memory,
                byte_size: info.byte_size as u64,
                usage: buffer_usage(info)?.as_raw(),
            },
        })
    }
    pub unsafe fn import_image(
        &mut self,
        info: &ImageInfo,
        exported: ExportedImage,
    ) -> Result<Handle<Image>> {
        let descriptor = exported.descriptor;
        if info.initial_data.is_some()
            || info.dim != descriptor.extent
            || lib_to_vk_image_format(&info.format).as_raw() != descriptor.format
            || image_usage(info)?.as_raw() != descriptor.usage
        {
            return Err(invalid(
                "Imported image description does not match exported image",
            ));
        }
        let image = self.create_shared_image(info, true)?;
        let result = (|| {
            self.check_memory(
                unsafe { self.device.get_image_memory_requirements(image) },
                descriptor.memory,
            )?;
            unsafe {
                self.allocate_import(
                    descriptor.memory,
                    exported.handle,
                    image,
                    vk::Buffer::null(),
                )
            }
        })();
        let memory = match result {
            Ok(m) => m,
            Err(e) => {
                unsafe {
                    self.device.destroy_image(image, None);
                }
                return Err(e);
            }
        };
        self.insert_external_image(image, memory, descriptor.memory, info, false)
    }
    pub unsafe fn import_buffer(
        &mut self,
        info: &BufferInfo,
        exported: ExportedBuffer,
    ) -> Result<Handle<Buffer>> {
        let descriptor = exported.descriptor;
        if info.initial_data.is_some()
            || info.byte_size as u64 != descriptor.byte_size
            || buffer_usage(info)?.as_raw() != descriptor.usage
        {
            return Err(invalid(
                "Imported buffer description does not match exported buffer",
            ));
        }
        let buffer = self.create_shared_buffer(info, true)?;
        let result = (|| {
            self.check_memory(
                unsafe { self.device.get_buffer_memory_requirements(buffer) },
                descriptor.memory,
            )?;
            unsafe {
                self.allocate_import(
                    descriptor.memory,
                    exported.handle,
                    vk::Image::null(),
                    buffer,
                )
            }
        })();
        let memory = match result {
            Ok(m) => m,
            Err(e) => {
                unsafe {
                    self.device.destroy_buffer(buffer, None);
                }
                return Err(e);
            }
        };
        self.insert_external_buffer(buffer, memory, descriptor.memory, info, false)
    }
    pub fn make_external_semaphore(&mut self) -> Result<Handle<Semaphore>> {
        if !self.external_caps.binary_semaphore {
            return Err(invalid("Vulkan external binary semaphores unavailable"));
        }
        let query =
            vk::PhysicalDeviceExternalSemaphoreInfo::builder().handle_type(semaphore_type());
        let mut properties = vk::ExternalSemaphoreProperties::default();
        unsafe {
            self.instance
                .get_physical_device_external_semaphore_properties(
                    self.pdevice,
                    &query,
                    &mut properties,
                )
        };
        if !properties.external_semaphore_features.contains(
            vk::ExternalSemaphoreFeatureFlags::EXPORTABLE
                | vk::ExternalSemaphoreFeatureFlags::IMPORTABLE,
        ) {
            return Err(invalid("Opaque binary semaphore sharing is unsupported"));
        }
        let mut export = vk::ExportSemaphoreCreateInfo::builder().handle_types(semaphore_type());
        let create = vk::SemaphoreCreateInfo::builder().push_next(&mut export);
        let raw = unsafe {
            self.device
                .create_semaphore(&create, self.allocation_callbacks.as_deref())
        }?;
        match self.semaphores.insert(Semaphore {
            raw,
            exportable: true,
        }) {
            Some(handle) => Ok(handle),
            None => {
                unsafe {
                    self.device
                        .destroy_semaphore(raw, self.allocation_callbacks.as_deref());
                }
                Err(GPUError::SlotError())
            }
        }
    }
    pub fn export_semaphore(&self, semaphore: Handle<Semaphore>) -> Result<ExportedSemaphore> {
        let semaphore = self
            .semaphores
            .get_ref(semaphore)
            .ok_or(GPUError::SlotError())?;
        if !semaphore.exportable {
            return Err(invalid("Semaphore was not created as exportable"));
        }
        #[cfg(windows)]
        let handle = {
            use std::os::windows::io::{FromRawHandle, OwnedHandle};
            let loader =
                ash::extensions::khr::ExternalSemaphoreWin32::new(&self.instance, &self.device);
            let query = vk::SemaphoreGetWin32HandleInfoKHR::builder()
                .semaphore(semaphore.raw)
                .handle_type(semaphore_type());
            unsafe {
                OSHandle::new(OwnedHandle::from_raw_handle(
                    loader.get_semaphore_win32_handle(&query)?,
                ))
            }
        };
        #[cfg(unix)]
        let handle = {
            use std::os::fd::{FromRawFd, OwnedFd};
            let loader =
                ash::extensions::khr::ExternalSemaphoreFd::new(&self.instance, &self.device);
            let query = vk::SemaphoreGetFdInfoKHR::builder()
                .semaphore(semaphore.raw)
                .handle_type(semaphore_type());
            unsafe { OSHandle::new(OwnedFd::from_raw_fd(loader.get_semaphore_fd(&query)?)) }
        };
        Ok(ExportedSemaphore {
            handle,
            device: self.device_identity(),
        })
    }
    pub unsafe fn import_semaphore(
        &mut self,
        exported: ExportedSemaphore,
    ) -> Result<Handle<Semaphore>> {
        if !self.external_caps.binary_semaphore {
            return Err(invalid("Vulkan external semaphores unavailable"));
        }
        self.check_device(exported.device)?;
        let handle = self.make_semaphore()?;
        let raw = self.semaphores.get_ref(handle).unwrap().raw;
        #[cfg(windows)]
        let result = {
            use std::os::windows::io::AsRawHandle;
            let loader =
                ash::extensions::khr::ExternalSemaphoreWin32::new(&self.instance, &self.device);
            let import = vk::ImportSemaphoreWin32HandleInfoKHR::builder()
                .semaphore(raw)
                .handle_type(semaphore_type())
                .handle(exported.handle.inner.as_raw_handle());
            unsafe { loader.import_semaphore_win32_handle(&import) }
        };
        #[cfg(unix)]
        let result = {
            use std::os::fd::AsRawFd;
            let loader =
                ash::extensions::khr::ExternalSemaphoreFd::new(&self.instance, &self.device);
            let import = vk::ImportSemaphoreFdInfoKHR::builder()
                .semaphore(raw)
                .handle_type(semaphore_type())
                .fd(exported.handle.inner.as_raw_fd());
            unsafe { loader.import_semaphore_fd(&import) }
        };
        if let Err(e) = result {
            self.destroy_semaphore(handle);
            return Err(e.into());
        }
        #[cfg(unix)]
        {
            use std::os::fd::IntoRawFd;
            let _ = exported.handle.inner.into_raw_fd();
        }
        Ok(handle)
    }
    fn external_queue_family(&self, queue: QueueType) -> u32 {
        match queue {
            QueueType::Graphics => self.gfx_queue.family,
            QueueType::Compute => {
                self.compute_queue
                    .as_ref()
                    .unwrap_or(&self.gfx_queue)
                    .family
            }
            QueueType::Transfer => {
                self.transfer_queue
                    .as_ref()
                    .or(self.compute_queue.as_ref())
                    .unwrap_or(&self.gfx_queue)
                    .family
            }
        }
    }
    pub fn transfer_external_image(
        &mut self,
        queue: &mut CommandQueue,
        image: Handle<Image>,
        acquire: bool,
    ) -> Result<()> {
        if !queue.dirty || queue.curr_rp.is_some() || !std::ptr::eq(queue.ctx, self) {
            return Err(invalid("External ownership barrier needs a recording queue from this context outside a render pass"));
        }
        let family = self.external_queue_family(queue.queue_type);
        let resource = self.images.get_ref(image).ok_or(GPUError::SlotError())?;
        if !matches!(resource.alloc, Allocation::External { .. })
            || resource.externally_owned != acquire
        {
            return Err(invalid("Invalid external image ownership transition"));
        }
        let range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        if !acquire && resource.layouts[0] != layout {
            let transition = vk::ImageMemoryBarrier::builder()
                .image(resource.img)
                .subresource_range(range)
                .old_layout(resource.layouts[0])
                .new_layout(layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                .dst_access_mask(vk::AccessFlags::MEMORY_READ)
                .build();
            unsafe {
                self.device.cmd_pipeline_barrier(
                    queue.cmd_buf,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[transition],
                );
            }
        }
        let barrier = vk::ImageMemoryBarrier::builder()
            .image(resource.img)
            .subresource_range(range)
            .old_layout(layout)
            .new_layout(layout)
            .src_queue_family_index(if acquire {
                vk::QUEUE_FAMILY_EXTERNAL
            } else {
                family
            })
            .dst_queue_family_index(if acquire {
                family
            } else {
                vk::QUEUE_FAMILY_EXTERNAL
            })
            .src_access_mask(if acquire {
                vk::AccessFlags::empty()
            } else {
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
            })
            .dst_access_mask(if acquire {
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
            } else {
                vk::AccessFlags::empty()
            })
            .build();
        unsafe {
            self.device.cmd_pipeline_barrier(
                queue.cmd_buf,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }
        self.images.with_mut(image, |r| {
            r.layouts[0] = layout;
            r.externally_owned = !acquire;
        });
        self.resource_states.request_image_state(
            image,
            SubresourceRange::default(),
            UsageBits::COPY_SRC,
            Layout::TransferSrc,
            queue.queue_type,
        );
        Ok(())
    }
    pub fn transfer_external_buffer(
        &mut self,
        queue: &mut CommandQueue,
        buffer: Handle<Buffer>,
        acquire: bool,
    ) -> Result<()> {
        if !queue.dirty || queue.curr_rp.is_some() || !std::ptr::eq(queue.ctx, self) {
            return Err(invalid("External ownership barrier needs a recording queue from this context outside a render pass"));
        }
        let family = self.external_queue_family(queue.queue_type);
        let resource = self.buffers.get_ref(buffer).ok_or(GPUError::SlotError())?;
        if !matches!(resource.alloc, Allocation::External { .. })
            || resource.suballocated
            || resource.externally_owned != acquire
        {
            return Err(invalid("Invalid external buffer ownership transition"));
        }
        let barrier = vk::BufferMemoryBarrier::builder()
            .buffer(resource.buf)
            .offset(0)
            .size(vk::WHOLE_SIZE)
            .src_queue_family_index(if acquire {
                vk::QUEUE_FAMILY_EXTERNAL
            } else {
                family
            })
            .dst_queue_family_index(if acquire {
                family
            } else {
                vk::QUEUE_FAMILY_EXTERNAL
            })
            .src_access_mask(if acquire {
                vk::AccessFlags::empty()
            } else {
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
            })
            .dst_access_mask(if acquire {
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE
            } else {
                vk::AccessFlags::empty()
            })
            .build();
        unsafe {
            self.device.cmd_pipeline_barrier(
                queue.cmd_buf,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );
        }
        self.buffers
            .with_mut(buffer, |r| r.externally_owned = !acquire);
        queue.recorded_buffer_states.remove(&buffer);
        self.resource_states
            .request_buffer_state(buffer, UsageBits::COPY_SRC, queue.queue_type);
        Ok(())
    }
}
