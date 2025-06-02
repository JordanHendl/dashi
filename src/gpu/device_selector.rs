use super::GPUError;
use ash::*;
use std::ffi::CStr;

#[derive(Default, Clone, Copy, PartialEq)]
pub enum DeviceType {
    Dedicated,
    Integrated,
    #[default] Other,
}

impl From<vk::PhysicalDeviceType> for DeviceType {
    fn from(value: vk::PhysicalDeviceType) -> Self {
        match value {
            vk::PhysicalDeviceType::DISCRETE_GPU => return DeviceType::Dedicated,
            vk::PhysicalDeviceType::INTEGRATED_GPU => return DeviceType::Integrated,
            _ => return DeviceType::Other,
        }
    }
}

#[derive(Default, Clone)]
pub struct DeviceInfo {
    name: String,
    kind: DeviceType,
    driver_version: u32,
    bindless_capable: bool,
    display_capable: bool,
    dashi_capable: bool,
}

impl From<vk::PhysicalDeviceProperties> for DeviceInfo {
    fn from(value: vk::PhysicalDeviceProperties) -> Self {
        DeviceInfo {
            name: unsafe {
                CStr::from_ptr(value.device_name.as_ptr())
                    .to_str()
                    .unwrap_or("UNKNOWN")
                    .to_string()
            },
            kind: value.device_type.into(),
            driver_version: value.driver_version,
            bindless_capable: false,
            display_capable: false,
            dashi_capable: false,
        }
    }
}

pub struct SelectedDevice {
    pub(crate) device_id: usize,
    pub info: DeviceInfo,
}

impl std::fmt::Display for SelectedDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
       write!(f, "[Name {} -- Driver Ver {} -- ID {}]", self.info.name, self.info.driver_version, self.device_id) 
    }
}
impl Default for SelectedDevice {
    fn default() -> Self {
        Self { device_id: 0,
            info: Default::default(),
        }
    }
}

#[derive(Default, Clone)]
pub struct DeviceFilter {
    name: Option<String>,
    kind: Option<DeviceType>,
    driver_version: Option<u32>,
    bindless_capable: Option<bool>,
    display_capable: Option<bool>,
}

impl DeviceFilter {
    pub fn add_required_name(&mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self.clone()
    }

    pub fn add_required_type(&mut self, kind: DeviceType) -> Self {
        self.kind = Some(kind);
        self.clone()
    }

    pub fn add_required_driver_version(&mut self, version: u32) -> Self {
        self.driver_version = Some(version);
        self.clone()
    }

    pub fn require_bindless(&mut self) -> Self {
        self.bindless_capable = Some(true);
        self.clone()
    }

    pub fn require_display(&mut self) -> Self {
        self.display_capable = Some(true);
        self.clone()
    }
}

pub struct DeviceSelector {
    devices: Vec<DeviceInfo>,
}

impl DeviceSelector {
    pub fn new() -> Result<DeviceSelector, GPUError> {
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let entry = unsafe { Entry::load() }?;
        let requested_instance_extensions = [
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name().as_ptr(),
            ash::extensions::khr::Surface::name().as_ptr(),
            #[cfg(target_os = "linux")]
            ash::extensions::khr::XlibSurface::name().as_ptr(),
            #[cfg(target_os = "windows")]
            ash::extensions::khr::Win32Surface::name().as_ptr(),
        ];

        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::builder()
                    .application_info(&app_info)
                    .enabled_extension_names(&requested_instance_extensions)
                    .build(),
                None,
            )
        }?;

        let mut infos: Vec<DeviceInfo> = Vec::new();

        let pdevices = unsafe { instance.enumerate_physical_devices()? };
        for device in pdevices {
            let mut info: DeviceInfo =
                unsafe { instance.get_physical_device_properties(device) }.into();

            let enabled_extensions =
                unsafe { instance.enumerate_device_extension_properties(device) }?;

            let mut descriptor_indexing =
                vk::PhysicalDeviceDescriptorIndexingFeatures::builder().build();
            let features = vk::PhysicalDeviceFeatures::builder()
                .shader_clip_distance(true)
                .build();

            let mut features2 = vk::PhysicalDeviceFeatures2::builder()
                .features(features)
                .push_next(&mut descriptor_indexing)
                .build();

            unsafe { instance.get_physical_device_features2(device, &mut features2) };
            if descriptor_indexing.shader_sampled_image_array_non_uniform_indexing > 0
                && descriptor_indexing.descriptor_binding_sampled_image_update_after_bind > 0
                && descriptor_indexing.shader_uniform_buffer_array_non_uniform_indexing > 0
                && descriptor_indexing.descriptor_binding_uniform_buffer_update_after_bind > 0
                && descriptor_indexing.shader_storage_buffer_array_non_uniform_indexing > 0
                && descriptor_indexing.descriptor_binding_storage_buffer_update_after_bind > 0
            {
                info.bindless_capable = true;
            }

            if let Some(_ext) = enabled_extensions.iter().find(|a| {
                return unsafe {
                    CStr::from_ptr(a.extension_name.as_ptr())
                        != ash::extensions::khr::Swapchain::name()
                };
            }) {
                info.display_capable = true;
            }

            info.dashi_capable = true;

            infos.push(info);
        }

        unsafe { instance.destroy_instance(None) };
        Ok(Self { devices: infos })
    }

    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    pub fn select_by_id(&self, id: usize) -> SelectedDevice {
        SelectedDevice { device_id: id, info: self.devices[id].clone() }
    }

    fn check<T: PartialEq>(a: T, b: Option<T>) -> bool {
        if let Some(c) = b {
            return c == a;
        }

        return false;
    }

    pub fn select(&self, filter: DeviceFilter) -> Option<SelectedDevice> {
        let mut max_score = 0;
        let mut best_device = -1;
        for (id, device) in self.devices.iter().enumerate() {
            let mut score = 0;
            if Self::check(&device.name, filter.name.as_ref()) {
                score += 1;
            }
            if Self::check(&device.kind, filter.kind.as_ref()) {
                score += 1;
            }
            if Self::check(&device.display_capable, filter.display_capable.as_ref()) {
                score += 1;
            }
            if Self::check(&device.bindless_capable, filter.bindless_capable.as_ref()) {
                score += 1;
            }

            if device.dashi_capable {
                score += 1;
            }

            if score > max_score {
                max_score = score;
                best_device = id as i32;
            }
        }

        if best_device >= 0 {
            return Some(SelectedDevice {
                device_id: best_device as usize,
                info: self.devices[best_device as usize].clone(),
            });
        }

        return None;
    }
}

impl Default for DeviceSelector {
    fn default() -> Self {
        Self::new().unwrap()
    }
}
