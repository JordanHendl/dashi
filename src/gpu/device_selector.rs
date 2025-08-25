use super::{GPUError, DEBUG_LAYER_NAMES};
use ash::*;
use std::ffi::{c_char, CStr};

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
    bind_table_capable: bool,
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
            bind_table_capable: false,
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
       write!(f, "[Name [{}] -- Driver Ver [{}] -- PCIE ID [{}]]", self.info.name, self.info.driver_version, self.device_id) 
    }
}
impl Default for SelectedDevice {
    fn default() -> Self {
        return DeviceSelector::new().unwrap().select_by_id(0);
    }
}

#[derive(Default, Clone)]
pub struct DeviceFilter {
    name: Option<String>,
    kind: Option<DeviceType>,
    driver_version: Option<u32>,
    bind_table_capable: Option<bool>,
    display_capable: Option<bool>,
}

impl DeviceFilter {
    /// Require the selected device's name to exactly match `name`.
    ///
    /// When used with [`DeviceSelector::select`], the selection will return
    /// [`None`] if no available device satisfies all required criteria.
    pub fn add_required_name(&mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self.clone()
    }

    /// Require the selected device to be of the specified `kind`.
    ///
    /// Selection fails with [`None`] if no device matches every requirement.
    pub fn add_required_type(&mut self, kind: DeviceType) -> Self {
        self.kind = Some(kind);
        self.clone()
    }

    /// Require the device driver version to equal `version`.
    ///
    /// [`DeviceSelector::select`] returns [`None`] when no device meets all
    /// prerequisites.
    pub fn add_required_driver_version(&mut self, version: u32) -> Self {
        self.driver_version = Some(version);
        self.clone()
    }

    /// Require support for bind tables on the selected device.
    ///
    /// [`DeviceSelector::select`] will return [`None`] if this capability is
    /// not available on any device.
    pub fn require_bind_table(&mut self) -> Self {
        self.bind_table_capable = Some(true);
        self.clone()
    }

    /// Require that the device supports presentation to a display (swapchain).
    ///
    /// If no device fulfills all filter requirements, selection yields
    /// [`None`].
    pub fn require_display(&mut self) -> Self {
        self.display_capable = Some(true);
        self.clone()
    }
}

pub struct DeviceSelector {
    devices: Vec<DeviceInfo>,
}

impl DeviceSelector {
    /// Enumerate available Vulkan physical devices and build a selector.
    ///
    /// A Vulkan instance must be creatable on the running platform; otherwise
    /// an error is returned.
    pub fn new() -> Result<DeviceSelector, GPUError> {
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let entry = unsafe { Entry::load() }?;
        let enable_validation = std::env::var("DASHI_VALIDATION").map(|v| v == "1").unwrap_or(false);

        let mut requested_instance_extensions = vec![
            ash::extensions::khr::Surface::name().as_ptr(),
            #[cfg(target_os = "linux")]
            ash::extensions::khr::XlibSurface::name().as_ptr(),
            #[cfg(target_os = "windows")]
            ash::extensions::khr::Win32Surface::name().as_ptr(),
        ];
        if enable_validation {
            requested_instance_extensions.push(ash::extensions::ext::DebugUtils::name().as_ptr());
        }

        let mut layer_names: Vec<*const c_char> = Vec::new();
        if enable_validation {
            let available_layers = entry.enumerate_instance_layer_properties()?;
            for &layer in &DEBUG_LAYER_NAMES {
                let name = unsafe { CStr::from_ptr(layer) };
                if available_layers.iter().any(|prop| unsafe {
                    CStr::from_ptr(prop.layer_name.as_ptr()) == name
                }) {
                    layer_names.push(layer);
                }
            }
        }

        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::builder()
                    .application_info(&app_info)
                    .enabled_extension_names(&requested_instance_extensions)
                    .enabled_layer_names(&layer_names)
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
                info.bind_table_capable = true;
            }

            if Self::has_swapchain_extension(&enabled_extensions) {
                info.display_capable = true;
            }

            info.dashi_capable = true;

            infos.push(info);
        }

        unsafe { instance.destroy_instance(None) };
        Ok(Self { devices: infos })
    }

    /// Retrieve the list of devices discovered during construction.
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Select a device by its index in the discovered device list.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range for `self.devices`.
    pub fn select_by_id(&self, id: usize) -> SelectedDevice {
        SelectedDevice { device_id: id, info: self.devices[id].clone() }
    }

    fn has_swapchain_extension(enabled: &[vk::ExtensionProperties]) -> bool {
        enabled.iter().any(|ext| unsafe {
            CStr::from_ptr(ext.extension_name.as_ptr())
                == ash::extensions::khr::Swapchain::name()
        })
    }

    fn check<T: PartialEq>(a: T, b: Option<T>) -> bool {
        if let Some(c) = b {
            return c == a;
        }

        return false;
    }

    /// Select the highest scoring device that satisfies the provided `filter`.
    ///
    /// Returns [`None`] when no device meets every required criterion.
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
            if Self::check(&device.bind_table_capable, filter.bind_table_capable.as_ref()) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;

    fn make_ext(name: &CStr) -> vk::ExtensionProperties {
        let mut ext_name = [0 as c_char; vk::MAX_EXTENSION_NAME_SIZE];
        let bytes = name.to_bytes_with_nul();
        for (i, b) in bytes.iter().enumerate() {
            ext_name[i] = *b as c_char;
        }
        vk::ExtensionProperties {
            extension_name: ext_name,
            spec_version: 0,
        }
    }

    #[test]
    fn detects_swapchain_extension() {
        let swap_ext = make_ext(ash::extensions::khr::Swapchain::name());
        let other_name = CString::new("VK_OTHER_ext").unwrap();
        let other_ext = make_ext(&other_name);

        let list = [other_ext.clone(), swap_ext];
        assert!(DeviceSelector::has_swapchain_extension(&list));

        let list = [other_ext];
        assert!(!DeviceSelector::has_swapchain_extension(&list));
    }
}
