use crate::gpu::device_selector::{DeviceInfo, DeviceSelector, DeviceType, SelectedDevice};
use super::{GPUError, DEBUG_LAYER_NAMES};
use ash::*;
use std::ffi::{c_char, CStr};

impl From<vk::PhysicalDeviceType> for DeviceType {
    fn from(value: vk::PhysicalDeviceType) -> Self {
        match value {
            vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::Dedicated,
            vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::Integrated,
            _ => DeviceType::Other,
        }
    }
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
        let enable_validation = std::env::var("DASHI_VALIDATION")
            .map(|v| v == "1")
            .unwrap_or(false);

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
            let properties = unsafe { instance.get_physical_device_properties(device) };
            let mut info: DeviceInfo = properties.into();

            let enabled_extensions =
                unsafe { instance.enumerate_device_extension_properties(device) }?;

            let supports_vulkan11 = vk::api_version_major(properties.api_version) > 1
                || (vk::api_version_major(properties.api_version) == 1
                    && vk::api_version_minor(properties.api_version) >= 1);

            if supports_vulkan11 {
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
            }

            if Self::has_swapchain_extension(&enabled_extensions) {
                info.display_capable = true;
            }

            info.dashi_capable = true;

            infos.push(info);
        }

        unsafe { instance.destroy_instance(None) };
        Ok(Self::with_devices(infos))
    }

    fn has_swapchain_extension(enabled: &[vk::ExtensionProperties]) -> bool {
        enabled.iter().any(|ext| unsafe {
            CStr::from_ptr(ext.extension_name.as_ptr())
                == ash::extensions::khr::Swapchain::name()
        })
    }
}

impl Default for SelectedDevice {
    fn default() -> Self {
        DeviceSelector::new().unwrap().select_by_id(0)
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
