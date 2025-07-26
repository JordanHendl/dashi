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
