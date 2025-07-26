#[cfg(feature = "dashi-openxr")]
use openxr as xr;
use ash::vk;
use ash::{Device, Instance};

/// Create an OpenXR Vulkan session and swapchain.
///
/// Returns the created `openxr::Instance`, `openxr::Session`, `openxr::Swapchain`,
/// the swapchain images, and the view configuration.
pub fn create_xr_session(
    vk_instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    queue_family_index: u32,
) -> Result<
    (
        xr::Instance,
        xr::Session<xr::Vulkan>,
        xr::FrameWaiter,
        xr::FrameStream<xr::Vulkan>,
        xr::Swapchain<xr::Vulkan>,
        Vec<xr::vulkan::SwapchainImage>,
        Vec<xr::ViewConfigurationView>,
    ),
    xr::sys::Result,
> {
    #[cfg(feature = "static")]
    let entry = xr::Entry::linked();
    #[cfg(not(feature = "static"))]
    let entry = unsafe { xr::Entry::load()? };

    let exts = entry.enumerate_extensions()?;
    if !exts.khr_vulkan_enable2 {
        return Err(xr::sys::Result::ERROR_EXTENSION_NOT_PRESENT);
    }
    let mut enabled = xr::ExtensionSet::default();
    enabled.khr_vulkan_enable2 = true;

    let instance = entry.create_instance(
        &xr::ApplicationInfo {
            application_name: "dashi",
            application_version: 0,
            engine_name: "dashi",
            engine_version: 0,
            api_version: xr::Version::new(1, 0, 0),
        },
        &enabled,
        &[],
    )?;

    let system = instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)?;

    let (session, waiter, stream) = instance.create_session::<xr::Vulkan>(
        system,
        &xr::vulkan::SessionCreateInfo {
            instance: vk_instance.handle().as_raw() as _,
            physical_device: physical_device.as_raw() as _,
            device: device.handle().as_raw() as _,
            queue_family_index,
            queue_index: 0,
        },
    )?;

    let views = instance.enumerate_view_configuration_views(
        system,
        xr::ViewConfigurationType::PRIMARY_STEREO,
    )?;
    let image_rect_width = views[0].recommended_image_rect_width;
    let image_rect_height = views[0].recommended_image_rect_height;
    let array_size = views.len() as u32;

    let swapchain = session.create_swapchain(&xr::SwapchainCreateInfo {
        create_flags: xr::SwapchainCreateFlags::EMPTY,
        usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT | xr::SwapchainUsageFlags::SAMPLED,
        format: vk::Format::B8G8R8A8_SRGB.as_raw() as _,
        sample_count: 1,
        width: image_rect_width,
        height: image_rect_height,
        face_count: 1,
        array_size,
        mip_count: 1,
    })?;
    let images = swapchain.enumerate_images()?;

    Ok((instance, session, waiter, stream, swapchain, images, views))
}

#[cfg(feature = "dashi-openxr")]
#[derive(Default, Debug)]
pub struct XrInputState {
    pub left_trigger: f32,
    pub right_trigger: f32,
    pub left_active: bool,
    pub right_active: bool,
}

#[cfg(feature = "dashi-openxr")]
pub struct XrInput {
    session: xr::Session<xr::Vulkan>,
    base_space: xr::Space,
    action_set: xr::ActionSet,
    grip_action: xr::Action<xr::Posef>,
    trigger_action: xr::Action<f32>,
    left_path: xr::Path,
    right_path: xr::Path,
    left_space: xr::Space,
    right_space: xr::Space,
}

#[cfg(feature = "dashi-openxr")]
impl XrInput {
    pub fn new(instance: &xr::Instance, session: &xr::Session<xr::Vulkan>) -> xr::Result<Self> {
        let action_set = instance.create_action_set("input", "input", 0)?;
        let left_path = instance.string_to_path("/user/hand/left")?;
        let right_path = instance.string_to_path("/user/hand/right")?;

        let grip_action =
            action_set.create_action::<xr::Posef>("grip_pose", "Grip Pose", &[left_path, right_path])?;
        let trigger_action =
            action_set.create_action::<f32>("trigger", "Trigger", &[left_path, right_path])?;

        instance.suggest_interaction_profile_bindings(
            instance.string_to_path("/interaction_profiles/khr/simple_controller")?,
            &[
                xr::Binding::new(&grip_action, instance.string_to_path("/user/hand/left/input/grip/pose")?),
                xr::Binding::new(&grip_action, instance.string_to_path("/user/hand/right/input/grip/pose")?),
                xr::Binding::new(&trigger_action, instance.string_to_path("/user/hand/left/input/select/value")?),
                xr::Binding::new(&trigger_action, instance.string_to_path("/user/hand/right/input/select/value")?),
            ],
        )?;

        session.attach_action_sets(&[&action_set])?;

        let base_space = session.create_reference_space(xr::ReferenceSpaceType::LOCAL, xr::Posef::IDENTITY)?;
        let left_space = grip_action.create_space(session.clone(), left_path, xr::Posef::IDENTITY)?;
        let right_space = grip_action.create_space(session.clone(), right_path, xr::Posef::IDENTITY)?;

        Ok(Self {
            session: session.clone(),
            base_space,
            action_set,
            grip_action,
            trigger_action,
            left_path,
            right_path,
            left_space,
            right_space,
        })
    }

    pub fn poll_inputs(&mut self) -> xr::Result<XrInputState> {
        self.session.sync_actions(&[(&self.action_set).into()])?;

        let left_trigger = self
            .trigger_action
            .state(&self.session, self.left_path)?
            .current_state;
        let right_trigger = self
            .trigger_action
            .state(&self.session, self.right_path)?
            .current_state;

        let left_active = self.grip_action.is_active(&self.session, self.left_path)?;
        let right_active = self.grip_action.is_active(&self.session, self.right_path)?;

        Ok(XrInputState {
            left_trigger,
            right_trigger,
            left_active,
            right_active,
        })
    }

    pub fn left_space(&self) -> &xr::Space {
        &self.left_space
    }

    pub fn right_space(&self) -> &xr::Space {
        &self.right_space
    }

    pub fn base_space(&self) -> &xr::Space {
        &self.base_space
    }
}
