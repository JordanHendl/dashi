use super::{vk, VulkanContext};

fn supports_compute_arithmetic(
    api_version: u32,
    properties: &vk::PhysicalDeviceSubgroupProperties,
) -> bool {
    api_version >= vk::API_VERSION_1_1
        && properties
            .supported_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
        && properties
            .supported_operations
            .contains(vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC)
}

impl VulkanContext {
    /// Whether compute shaders can use standard subgroup election and reductions.
    pub fn supports_compute_subgroup_arithmetic(&self) -> bool {
        if self.properties.api_version < vk::API_VERSION_1_1 {
            return false;
        }
        let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::builder().push_next(&mut subgroup);
        unsafe {
            self.instance
                .get_physical_device_properties2(self.pdevice, &mut properties)
        };
        supports_compute_arithmetic(self.properties.api_version, &subgroup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_subgroup_arithmetic_requires_stage_and_both_operations() {
        let supported = vk::PhysicalDeviceSubgroupProperties {
            supported_stages: vk::ShaderStageFlags::COMPUTE,
            supported_operations: vk::SubgroupFeatureFlags::BASIC
                | vk::SubgroupFeatureFlags::ARITHMETIC,
            ..Default::default()
        };
        assert!(supports_compute_arithmetic(vk::API_VERSION_1_1, &supported));
        assert!(!supports_compute_arithmetic(
            vk::API_VERSION_1_0,
            &supported
        ));
        for stages in [
            vk::ShaderStageFlags::empty(),
            vk::ShaderStageFlags::FRAGMENT,
        ] {
            assert!(!supports_compute_arithmetic(
                vk::API_VERSION_1_1,
                &vk::PhysicalDeviceSubgroupProperties {
                    supported_stages: stages,
                    ..supported
                }
            ));
        }
        for operations in [
            vk::SubgroupFeatureFlags::empty(),
            vk::SubgroupFeatureFlags::BASIC,
            vk::SubgroupFeatureFlags::ARITHMETIC,
        ] {
            assert!(!supports_compute_arithmetic(
                vk::API_VERSION_1_1,
                &vk::PhysicalDeviceSubgroupProperties {
                    supported_operations: operations,
                    ..supported
                }
            ));
        }
    }
}
