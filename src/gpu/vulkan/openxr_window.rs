//! OpenXR graphics bootstrap. Window displays remain independent of XR sessions.
use super::*;
use openxr as xr;

pub struct XrGraphics {
    pub instance: xr::Instance,
    pub system: xr::SystemId,
    requirements: xr::vulkan::Requirements,
}

fn xr_error(error: impl std::fmt::Display) -> GPUError {
    GPUError::LibraryError(format!("OpenXR: {error}"))
}

impl XrGraphics {
    pub fn load(application_name: &str) -> Result<std::sync::Arc<Self>> {
        let entry = unsafe { xr::Entry::load() }.map_err(xr_error)?;
        let available = entry.enumerate_extensions().map_err(xr_error)?;
        if !available.khr_vulkan_enable2 {
            return Err(xr_error("XR_KHR_vulkan_enable2 is required"));
        }
        let mut extensions = xr::ExtensionSet::default();
        extensions.khr_vulkan_enable2 = true;
        let instance = entry
            .create_instance(
                &xr::ApplicationInfo {
                    application_name,
                    application_version: 1,
                    engine_name: "dashi",
                    engine_version: 1,
                    api_version: xr::Version::new(1, 0, 0),
                },
                &extensions,
                &[],
            )
            .map_err(xr_error)?;
        let system = instance
            .system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
            .map_err(xr_error)?;
        let requirements = instance
            .graphics_requirements::<xr::Vulkan>(system)
            .map_err(xr_error)?;
        Ok(std::sync::Arc::new(Self {
            instance,
            system,
            requirements,
        }))
    }

    pub(super) fn api_version(&self) -> Result<u32> {
        let upper = self
            .requirements
            .max_api_version_supported
            .min(xr::Version::new(1, 3, 0));
        if upper < self.requirements.min_api_version_supported || upper < xr::Version::new(1, 2, 0)
        {
            return Err(xr_error("runtime must support Vulkan 1.2 or later"));
        }
        Ok(vk::make_api_version(
            0,
            upper.major() as _,
            upper.minor() as _,
            0,
        ))
    }

    pub(super) unsafe fn create_instance(
        &self,
        entry: &ash::Entry,
        info: &vk::InstanceCreateInfo,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> Result<ash::Instance> {
        let api = (*info.p_application_info).api_version;
        let version = xr::Version::new(
            vk::api_version_major(api) as _,
            vk::api_version_minor(api) as _,
            0,
        );
        if version < self.requirements.min_api_version_supported
            || version > self.requirements.max_api_version_supported
        {
            return Err(xr_error(format!(
                "Vulkan {version} outside runtime graphics requirements"
            )));
        }
        let mut raw = std::ptr::null();
        let mut result = 0;
        let status = (self
            .instance
            .exts()
            .khr_vulkan_enable2
            .as_ref()
            .unwrap()
            .create_vulkan_instance)(
            self.instance.as_raw(),
            &xr::sys::VulkanInstanceCreateInfoKHR {
                ty: xr::sys::VulkanInstanceCreateInfoKHR::TYPE,
                next: std::ptr::null(),
                system_id: self.system,
                create_flags: xr::sys::VulkanInstanceCreateFlagsKHR::EMPTY,
                pfn_get_instance_proc_addr: Some(std::mem::transmute(
                    entry.static_fn().get_instance_proc_addr,
                )),
                vulkan_create_info: info as *const _ as _,
                vulkan_allocator: allocator.map_or(std::ptr::null(), |a| a as *const _ as _),
            },
            &mut raw,
            &mut result,
        );
        if status.into_raw() < 0 {
            return Err(xr_error(status));
        }
        if result < 0 {
            return Err(vk::Result::from_raw(result).into());
        }
        Ok(ash::Instance::load(
            entry.static_fn(),
            vk::Instance::from_raw(raw as u64),
        ))
    }

    pub(super) unsafe fn physical_device(
        &self,
        instance: &ash::Instance,
    ) -> Result<vk::PhysicalDevice> {
        self.instance
            .vulkan_graphics_device(self.system, instance.handle().as_raw() as _)
            .map(|raw| vk::PhysicalDevice::from_raw(raw as u64))
            .map_err(xr_error)
    }

    pub(super) unsafe fn create_device(
        &self,
        entry: &ash::Entry,
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        info: &vk::DeviceCreateInfo,
        allocator: Option<&vk::AllocationCallbacks>,
    ) -> Result<ash::Device> {
        let mut raw = std::ptr::null();
        let mut result = 0;
        let status = (self
            .instance
            .exts()
            .khr_vulkan_enable2
            .as_ref()
            .unwrap()
            .create_vulkan_device)(
            self.instance.as_raw(),
            &xr::sys::VulkanDeviceCreateInfoKHR {
                ty: xr::sys::VulkanDeviceCreateInfoKHR::TYPE,
                next: std::ptr::null(),
                system_id: self.system,
                create_flags: xr::sys::VulkanDeviceCreateFlagsKHR::EMPTY,
                pfn_get_instance_proc_addr: Some(std::mem::transmute(
                    entry.static_fn().get_instance_proc_addr,
                )),
                vulkan_physical_device: physical_device.as_raw() as _,
                vulkan_create_info: info as *const _ as _,
                vulkan_allocator: allocator.map_or(std::ptr::null(), |a| a as *const _ as _),
            },
            &mut raw,
            &mut result,
        );
        if status.into_raw() < 0 {
            return Err(xr_error(status));
        }
        if result < 0 {
            return Err(vk::Result::from_raw(result).into());
        }
        Ok(ash::Device::load(
            instance.fp_v1_0(),
            vk::Device::from_raw(raw as u64),
        ))
    }
}

impl VulkanContext {
    /// Creates a session only on a device initialized with the same OpenXR bootstrap.
    /// The caller must destroy session children and the session before this context.
    pub fn create_xr_session(
        &self,
    ) -> Result<(
        xr::Session<xr::Vulkan>,
        xr::FrameWaiter,
        xr::FrameStream<xr::Vulkan>,
    )> {
        let xr = self
            .xr
            .as_ref()
            .ok_or_else(|| xr_error("graphics context was not initialized for XR"))?;
        unsafe {
            xr.instance.create_session::<xr::Vulkan>(
                xr.system,
                &xr::vulkan::SessionCreateInfo {
                    instance: self.instance.handle().as_raw() as _,
                    physical_device: self.pdevice.as_raw() as _,
                    device: self.device.handle().as_raw() as _,
                    queue_family_index: self.gfx_queue.family,
                    queue_index: 0,
                },
            )
        }
        .map_err(xr_error)
    }

    /// Registers an image owned by the runtime. Destroying the handle releases cached views,
    /// but never frees the runtime's VkImage or memory.
    ///
    /// # Safety
    /// `image` must be an image from this context's XR session, described exactly by `info`.
    /// It must outlive this registration and all GPU uses. The caller must acquire/wait before
    /// use, and submit all work with the required final layout before releasing to OpenXR.
    pub unsafe fn import_xr_image(
        &mut self,
        image: u64,
        info: &ImageInfo,
    ) -> Result<Handle<Image>> {
        if self.xr.is_none()
            || image == 0
            || info.layers == 0
            || info.dim.contains(&0)
            || info.mip_levels != 1
        {
            return Err(xr_error("invalid XR image registration"));
        }
        let info_handle = self
            .image_infos
            .insert(ImageInfoRecord::new(info))
            .ok_or(GPUError::SlotError())?;
        match self.images.insert(Image {
            img: vk::Image::from_raw(image),
            alloc: Allocation::Borrowed,
            externally_owned: true,
            layouts: vec![vk::ImageLayout::UNDEFINED],
            info_handle,
        }) {
            Some(handle) => Ok(handle),
            None => {
                self.image_infos.release(info_handle);
                Err(GPUError::SlotError())
            }
        }
    }
}
