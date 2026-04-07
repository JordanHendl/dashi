use crate::utils::Handle;
use ash::vk;
#[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
use std::{thread, time::Duration};

#[cfg(feature = "dashi-openxr")]
use super::XrSwapchainImage;
use super::{
    DisplayInfo, DisplayStatus, Fence, GPUError, Image, ImageInfo, ImageInfoRecord, ImageView,
    ImageViewType, SampleCount, Semaphore, VulkanContext, WindowBuffering, WindowInfo,
};

#[cfg(feature = "dashi-openxr")]
use super::{openxr_window, XrDisplayInfo};
#[cfg(feature = "dashi-openxr")]
use openxr as xr;

#[cfg(all(feature = "dashi-minifb", not(feature = "dashi-openxr")))]
use super::minifb_window;
#[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
use super::winit_window;
#[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
#[cfg(all(
    feature = "dashi-winit",
    not(feature = "dashi-openxr"),
    target_os = "windows"
))]
use windows_sys::Win32::UI::WindowsAndMessaging::IsIconic;
#[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::ControlFlow,
    platform::run_return::EventLoopExtRunReturn,
};

#[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct WinitEventBatch {
    close_requested: bool,
    resized: bool,
    scale_factor_changed: bool,
}

#[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrepareAction {
    status: Option<DisplayStatus>,
    should_rebuild: bool,
    wait_for_restore: bool,
}

#[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
fn classify_prepare_action(
    swapchain_size: [u32; 2],
    observed_size: [u32; 2],
    needs_rebuild: bool,
    close_requested: bool,
    minimized: bool,
) -> PrepareAction {
    if close_requested {
        return PrepareAction {
            status: Some(DisplayStatus::Closed),
            should_rebuild: false,
            wait_for_restore: false,
        };
    }

    if minimized {
        return PrepareAction {
            status: None,
            should_rebuild: false,
            wait_for_restore: true,
        };
    }

    let resized = observed_size != swapchain_size;
    PrepareAction {
        status: Some(if resized {
            DisplayStatus::Resized {
                size: observed_size,
            }
        } else {
            DisplayStatus::Ready {
                size: observed_size,
            }
        }),
        should_rebuild: needs_rebuild || resized,
        wait_for_restore: false,
    }
}

#[cfg(not(feature = "dashi-openxr"))]
struct SwapchainConfig {
    extent: vk::Extent2D,
    surface_format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    min_image_count: u32,
    pre_transform: vk::SurfaceTransformFlagsKHR,
}

#[cfg(not(feature = "dashi-openxr"))]
struct SwapchainResources {
    swapchain: ash::vk::SwapchainKHR,
    sc_loader: ash::extensions::khr::Swapchain,
    images: Vec<Handle<Image>>,
    views: Vec<vk::ImageView>,
    semaphores: Vec<Handle<Semaphore>>,
    fences: Vec<Handle<Fence>>,
    extent: vk::Extent2D,
}

#[allow(dead_code)]
pub struct Display {
    #[cfg(all(feature = "dashi-sdl2", not(feature = "dashi-openxr")))]
    pub(crate) window: std::cell::Cell<sdl2::video::Window>,
    #[cfg(all(feature = "dashi-minifb", not(feature = "dashi-openxr")))]
    pub(crate) window: minifb::Window,
    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub(crate) window: winit::window::Window,
    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub(crate) event_loop: winit::event_loop::EventLoop<()>,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) swapchain: ash::vk::SwapchainKHR,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) surface: ash::vk::SurfaceKHR,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) images: Vec<Handle<Image>>,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) views: Vec<vk::ImageView>,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) loader: ash::extensions::khr::Surface,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) sc_loader: ash::extensions::khr::Swapchain,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) semaphores: Vec<Handle<Semaphore>>,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) fences: Vec<Handle<Fence>>,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) frame_idx: u32,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) info: DisplayInfo,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) extent: vk::Extent2D,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) needs_rebuild: std::cell::Cell<bool>,
    #[cfg(not(feature = "dashi-openxr"))]
    pub(crate) closed: bool,
    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub(crate) minimized: bool,

    #[cfg(feature = "dashi-openxr")]
    pub(crate) xr_instance: xr::Instance,
    #[cfg(feature = "dashi-openxr")]
    pub(crate) xr_session: xr::Session<xr::Vulkan>,
    #[cfg(feature = "dashi-openxr")]
    pub(crate) xr_waiter: xr::FrameWaiter,
    #[cfg(feature = "dashi-openxr")]
    pub(crate) xr_stream: xr::FrameStream<xr::Vulkan>,
    #[cfg(feature = "dashi-openxr")]
    pub(crate) xr_swapchain: xr::Swapchain<xr::Vulkan>,
    #[cfg(feature = "dashi-openxr")]
    pub(crate) xr_images: Vec<XrSwapchainImage>,
    #[cfg(feature = "dashi-openxr")]
    pub(crate) xr_view_config: Vec<xr::ViewConfigurationView>,
    #[cfg(feature = "dashi-openxr")]
    pub(crate) xr_input: openxr_window::XrInput,
}

impl Display {
    #[cfg(all(feature = "dashi-minifb", not(feature = "dashi-openxr")))]
    pub fn minifb_window(&mut self) -> &mut minifb::Window {
        &mut self.window
    }

    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub fn winit_window(&self) -> &winit::window::Window {
        &self.window
    }

    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub fn winit_event_loop(&mut self) -> &mut winit::event_loop::EventLoop<()> {
        &mut self.event_loop
    }

    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub fn set_resizable(&mut self, resizable: bool) {
        self.info.window.resizable = resizable;
        self.window.set_resizable(resizable);
    }

    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.info.window.size = [width, height];
        self.window.set_inner_size(PhysicalSize::new(width, height));
    }

    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub fn size(&self) -> [u32; 2] {
        let size = self.window.inner_size();
        [size.width, size.height]
    }

    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub fn minimize(&mut self) {
        self.minimized = true;
        self.window.set_minimized(true);
    }

    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub fn restore(&mut self) {
        self.minimized = false;
        self.window.set_minimized(false);
    }

    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    fn pump_events(&mut self, _wait_for_events: bool) -> WinitEventBatch {
        let mut batch = WinitEventBatch::default();
        let window_id = self.window.id();

        self.event_loop.run_return(|event, _target, control_flow| {
            *control_flow = ControlFlow::Poll;

            match event {
                Event::WindowEvent {
                    window_id: event_window_id,
                    event,
                } if event_window_id == window_id => match event {
                    WindowEvent::CloseRequested => batch.close_requested = true,
                    WindowEvent::Resized(_) => batch.resized = true,
                    WindowEvent::ScaleFactorChanged { .. } => batch.scale_factor_changed = true,
                    _ => {}
                },
                Event::MainEventsCleared | Event::LoopDestroyed => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            }
        });

        batch
    }

    #[cfg(all(
        feature = "dashi-winit",
        not(feature = "dashi-openxr"),
        target_os = "windows"
    ))]
    fn is_os_minimized(&self) -> bool {
        match self.window.raw_window_handle() {
            RawWindowHandle::Win32(handle) => unsafe { IsIconic(handle.hwnd as _) != 0 },
            _ => false,
        }
    }

    #[cfg(all(
        feature = "dashi-winit",
        not(feature = "dashi-openxr"),
        not(target_os = "windows")
    ))]
    fn is_os_minimized(&self) -> bool {
        false
    }

    #[cfg(not(feature = "dashi-openxr"))]
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_instance(&self) -> &xr::Instance {
        &self.xr_instance
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_session(&self) -> &xr::Session<xr::Vulkan> {
        &self.xr_session
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_frame_waiter(&mut self) -> &mut xr::FrameWaiter {
        &mut self.xr_waiter
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_frame_stream(&mut self) -> &mut xr::FrameStream<xr::Vulkan> {
        &mut self.xr_stream
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_swapchain(&self) -> &xr::Swapchain<xr::Vulkan> {
        &self.xr_swapchain
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_swapchain_images(&self) -> &[XrSwapchainImage] {
        &self.xr_images
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_view_configuration(&self) -> &[xr::ViewConfigurationView] {
        &self.xr_view_config
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn poll_inputs(&mut self) -> xr::Result<openxr_window::XrInputState> {
        self.xr_input.poll_inputs()
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_input(&self) -> &openxr_window::XrInput {
        &self.xr_input
    }
}

impl VulkanContext {
    #[cfg(not(feature = "dashi-openxr"))]
    fn release_swapchain_images(
        &mut self,
        handles: &mut Vec<Handle<Image>>,
        views: &mut Vec<vk::ImageView>,
    ) {
        for view in views.drain(..) {
            unsafe { self.device.destroy_image_view(view, None) };
        }

        for handle in handles.drain(..) {
            if let Some(image) = self.images.get_ref(handle) {
                self.image_infos.release(image.info_handle);
            }
            self.images.release(handle);
        }
    }

    #[cfg(not(feature = "dashi-openxr"))]
    fn destroy_swapchain_resources(&mut self, dsp: &mut Display) {
        self.release_swapchain_images(&mut dsp.images, &mut dsp.views);

        if dsp.swapchain != vk::SwapchainKHR::null() {
            unsafe { dsp.sc_loader.destroy_swapchain(dsp.swapchain, None) };
            dsp.swapchain = vk::SwapchainKHR::null();
        }

        for sem in dsp.semaphores.drain(..) {
            self.destroy_semaphore(sem);
        }

        for fence in dsp.fences.drain(..) {
            self.destroy_fence(fence);
        }
    }

    #[cfg(not(feature = "dashi-openxr"))]
    fn choose_swapchain_config(
        &self,
        loader: &ash::extensions::khr::Surface,
        surface: vk::SurfaceKHR,
        info: &DisplayInfo,
        requested_size: [u32; 2],
    ) -> Result<SwapchainConfig, GPUError> {
        let capabilities =
            unsafe { loader.get_physical_device_surface_capabilities(self.pdevice, surface)? };
        let formats = unsafe { loader.get_physical_device_surface_formats(self.pdevice, surface)? };
        let present_modes =
            unsafe { loader.get_physical_device_surface_present_modes(self.pdevice, surface)? };

        let mut extent = vk::Extent2D {
            width: requested_size[0],
            height: requested_size[1],
        };
        if capabilities.current_extent.width != std::u32::MAX {
            extent = capabilities.current_extent;
        } else {
            extent.width = std::cmp::max(
                capabilities.min_image_extent.width,
                std::cmp::min(capabilities.max_image_extent.width, extent.width),
            );
            extent.height = std::cmp::max(
                capabilities.min_image_extent.height,
                std::cmp::min(capabilities.max_image_extent.height, extent.height),
            );
        }

        let requested_present_mode = if info.vsync {
            vk::PresentModeKHR::FIFO
        } else {
            vk::PresentModeKHR::IMMEDIATE
        };
        let present_mode = present_modes
            .iter()
            .copied()
            .find(|mode| *mode == requested_present_mode)
            .or_else(|| {
                present_modes
                    .iter()
                    .copied()
                    .find(|mode| *mode == vk::PresentModeKHR::FIFO)
            })
            .ok_or(GPUError::SwapchainConfigError(
                "No compatible present mode available",
            ))?;

        let wanted_format = vk::Format::B8G8R8A8_SRGB;
        let wanted_color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;
        let surface_format = formats
            .iter()
            .copied()
            .find(|format| {
                format.format == wanted_format && format.color_space == wanted_color_space
            })
            .or_else(|| {
                formats
                    .iter()
                    .copied()
                    .find(|format| format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            })
            .ok_or(GPUError::SwapchainConfigError(
                "No compatible SRGB surface format available",
            ))?;

        let requested_images = match info.buffering {
            WindowBuffering::Double => 2,
            WindowBuffering::Triple => 3,
        };
        let mut min_image_count = std::cmp::max(requested_images, capabilities.min_image_count);
        if capabilities.max_image_count > 0 {
            min_image_count = min_image_count.min(capabilities.max_image_count);
        }

        Ok(SwapchainConfig {
            extent,
            surface_format,
            present_mode,
            min_image_count,
            pre_transform: capabilities.current_transform,
        })
    }

    #[cfg(not(feature = "dashi-openxr"))]
    fn wrap_swapchain_images(
        &mut self,
        raw_images: Vec<vk::Image>,
        surface_format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<(Vec<Handle<Image>>, Vec<vk::ImageView>), GPUError> {
        let mut handles = Vec::with_capacity(raw_images.len());
        let mut views = Vec::with_capacity(raw_images.len());

        for raw_img in raw_images {
            let image_info = ImageInfo {
                debug_name: "Swapchain Image",
                dim: [extent.width, extent.height, 1],
                layers: 1,
                format: super::vk_to_lib_image_format(surface_format)?,
                mip_levels: 1,
                samples: SampleCount::S1,
                cube_compatible: false,
                storage: false,
                initial_data: None,
            };
            let info_handle = match self.image_infos.insert(ImageInfoRecord::new(&image_info)) {
                Some(handle) => handle,
                None => {
                    self.release_swapchain_images(&mut handles, &mut views);
                    return Err(GPUError::SlotError());
                }
            };

            let handle = match self.images.insert(Image {
                img: raw_img,
                alloc: unsafe { std::mem::MaybeUninit::zeroed().assume_init() },
                layouts: vec![vk::ImageLayout::UNDEFINED],
                info_handle,
            }) {
                Some(handle) => handle,
                None => {
                    self.image_infos.release(info_handle);
                    self.release_swapchain_images(&mut handles, &mut views);
                    return Err(GPUError::SlotError());
                }
            };

            self.oneshot_transition_image_noview(handle, vk::ImageLayout::PRESENT_SRC_KHR);

            let sub_range = vk::ImageSubresourceRange::builder()
                .base_array_layer(0)
                .layer_count(1)
                .base_mip_level(0)
                .level_count(1)
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .build();

            let view = match unsafe {
                self.device.create_image_view(
                    &vk::ImageViewCreateInfo::builder()
                        .image(raw_img)
                        .format(surface_format)
                        .subresource_range(sub_range)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .build(),
                    None,
                )
            } {
                Ok(view) => view,
                Err(err) => {
                    if let Some(image) = self.images.get_ref(handle) {
                        self.image_infos.release(image.info_handle);
                    }
                    self.images.release(handle);
                    self.release_swapchain_images(&mut handles, &mut views);
                    return Err(err.into());
                }
            };

            handles.push(handle);
            views.push(view);
        }

        Ok((handles, views))
    }

    #[cfg(not(feature = "dashi-openxr"))]
    fn create_swapchain_resources(
        &mut self,
        surface: vk::SurfaceKHR,
        loader: &ash::extensions::khr::Surface,
        info: &DisplayInfo,
        requested_size: [u32; 2],
    ) -> Result<SwapchainResources, GPUError> {
        let config = self.choose_swapchain_config(loader, surface, info, requested_size)?;
        let image_usage = vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        let sc_loader = ash::extensions::khr::Swapchain::new(&self.instance, &self.device);
        let swapchain = unsafe {
            sc_loader.create_swapchain(
                &vk::SwapchainCreateInfoKHR::builder()
                    .surface(surface)
                    .present_mode(config.present_mode)
                    .image_format(config.surface_format.format)
                    .image_array_layers(1)
                    .image_color_space(config.surface_format.color_space)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .image_extent(config.extent)
                    .image_usage(image_usage)
                    .min_image_count(config.min_image_count)
                    .pre_transform(config.pre_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .build(),
                None,
            )?
        };

        let raw_images = unsafe { sc_loader.get_swapchain_images(swapchain)? };
        let (mut images, mut views) =
            self.wrap_swapchain_images(raw_images, config.surface_format.format, config.extent)?;

        let semaphores = match self.make_semaphores(images.len()) {
            Ok(semaphores) => semaphores,
            Err(err) => {
                self.release_swapchain_images(&mut images, &mut views);
                unsafe { sc_loader.destroy_swapchain(swapchain, None) };
                return Err(err);
            }
        };

        let mut fences = Vec::with_capacity(images.len());
        for _ in 0..images.len() {
            match self.make_fence() {
                Ok(fence) => fences.push(fence),
                Err(err) => {
                    for sem in semaphores {
                        self.destroy_semaphore(sem);
                    }
                    for fence in fences {
                        self.destroy_fence(fence);
                    }
                    self.release_swapchain_images(&mut images, &mut views);
                    unsafe { sc_loader.destroy_swapchain(swapchain, None) };
                    return Err(err);
                }
            }
        }

        if let Err(err) = self.init_gpu_timers(images.len()) {
            for sem in semaphores {
                self.destroy_semaphore(sem);
            }
            for fence in fences {
                self.destroy_fence(fence);
            }
            self.release_swapchain_images(&mut images, &mut views);
            unsafe { sc_loader.destroy_swapchain(swapchain, None) };
            return Err(err);
        }

        Ok(SwapchainResources {
            swapchain,
            sc_loader,
            images,
            views,
            semaphores,
            fences,
            extent: config.extent,
        })
    }

    #[cfg(not(feature = "dashi-openxr"))]
    fn rebuild_display_swapchain(
        &mut self,
        dsp: &mut Display,
        requested_size: [u32; 2],
    ) -> Result<(), GPUError> {
        dsp.needs_rebuild.set(true);
        unsafe { self.device.device_wait_idle()? };
        self.destroy_swapchain_resources(dsp);
        dsp.info.window.size = requested_size;

        let resources =
            self.create_swapchain_resources(dsp.surface, &dsp.loader, &dsp.info, requested_size)?;
        dsp.swapchain = resources.swapchain;
        dsp.sc_loader = resources.sc_loader;
        dsp.images = resources.images;
        dsp.views = resources.views;
        dsp.semaphores = resources.semaphores;
        dsp.fences = resources.fences;
        dsp.extent = resources.extent;
        dsp.frame_idx = if dsp.images.len() > 1 { 1 } else { 0 };
        dsp.needs_rebuild.set(false);
        #[cfg(feature = "dashi-winit")]
        {
            dsp.minimized = false;
        }
        Ok(())
    }

    #[cfg(not(feature = "dashi-openxr"))]
    /// Destroys a windowing display and its swapchain.
    ///
    /// # Prerequisites
    /// - Ensure the GPU has finished using all swapchain images.
    /// - Image views should be destroyed before their images and swapchain.
    /// - The context must still be alive.
    pub fn destroy_display(&mut self, mut dsp: Display) {
        self.destroy_swapchain_resources(&mut dsp);
        unsafe { dsp.loader.destroy_surface(dsp.surface, None) };
    }

    #[cfg(feature = "dashi-openxr")]
    /// Destroys an OpenXR display.
    ///
    /// # Prerequisites
    /// - Ensure the GPU has finished using any swapchain images provided by OpenXR.
    /// - The context must still be alive.
    ///
    /// OpenXR resources are cleaned up by their `Drop` implementations.
    pub fn destroy_display(&mut self, _dsp: Display) {
        // OpenXR resources are cleaned up by Drop implementations
    }

    #[cfg(all(feature = "dashi-sdl2", not(feature = "dashi-openxr")))]
    fn make_window(
        &mut self,
        info: &WindowInfo,
    ) -> (std::cell::Cell<sdl2::video::Window>, vk::SurfaceKHR) {
        let mut window = std::cell::Cell::new(
            self.sdl_video
                .as_ref()
                .unwrap()
                .window(&info.title, info.size[0], info.size[1])
                .vulkan()
                .build()
                .expect("Unable to create SDL2 Window!"),
        );

        let surface = window
            .get_mut()
            .vulkan_create_surface(vk::Handle::as_raw(self.instance.handle()) as usize)
            .expect("Unable to create vulkan surface!");

        (window, vk::Handle::from_raw(surface))
    }

    #[cfg(all(feature = "dashi-minifb", not(feature = "dashi-openxr")))]
    fn make_window(&mut self, info: &WindowInfo) -> (minifb::Window, vk::SurfaceKHR) {
        minifb_window::create_window(&self.entry, &self.instance, info).unwrap()
    }

    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    fn make_window(
        &mut self,
        info: &WindowInfo,
    ) -> (
        winit::event_loop::EventLoop<()>,
        winit::window::Window,
        vk::SurfaceKHR,
    ) {
        winit_window::create_window(&self.entry, &self.instance, info).unwrap()
    }

    #[cfg(not(feature = "dashi-openxr"))]
    /// Creates a display and associated swapchain for presenting images.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn make_display(&mut self, info: &DisplayInfo) -> Result<Display, GPUError> {
        if self.headless {
            return Err(GPUError::HeadlessDisplayNotSupported);
        }
        #[cfg(feature = "dashi-winit")]
        let (event_loop, window, surface) = self.make_window(&info.window);
        #[cfg(not(feature = "dashi-winit"))]
        let (window, surface) = self.make_window(&info.window);

        let loader = ash::extensions::khr::Surface::new(&self.entry, &self.instance);
        let resources =
            self.create_swapchain_resources(surface, &loader, info, info.window.size)?;

        Ok(Display {
            window,
            #[cfg(feature = "dashi-winit")]
            event_loop,
            swapchain: resources.swapchain,
            surface,
            images: resources.images,
            loader,
            sc_loader: resources.sc_loader,
            frame_idx: if resources.fences.len() > 1 { 1 } else { 0 },
            semaphores: resources.semaphores,
            fences: resources.fences,
            views: resources.views,
            info: info.clone(),
            extent: resources.extent,
            needs_rebuild: std::cell::Cell::new(false),
            closed: false,
            #[cfg(feature = "dashi-winit")]
            minimized: false,
        })
    }

    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub fn prepare_display(&mut self, dsp: &mut Display) -> Result<DisplayStatus, GPUError> {
        if dsp.closed {
            return Ok(DisplayStatus::Closed);
        }

        loop {
            let batch = dsp.pump_events(dsp.minimized);
            let size = dsp.size();
            let minimized = size[0] == 0 || size[1] == 0 || dsp.is_os_minimized();
            let action = classify_prepare_action(
                [dsp.extent.width, dsp.extent.height],
                size,
                dsp.needs_rebuild.get(),
                batch.close_requested,
                minimized,
            );

            if batch.close_requested {
                dsp.closed = true;
            }

            if action.wait_for_restore {
                dsp.minimized = true;
                thread::sleep(Duration::from_millis(16));
                continue;
            }

            dsp.minimized = false;
            dsp.info.window.size = size;

            if action.should_rebuild {
                self.rebuild_display_swapchain(dsp, size)?;
            }

            return Ok(action.status.expect("ready, resized, or closed"));
        }
    }

    #[cfg(feature = "dashi-openxr")]
    /// Creates an XR display and session for immersive rendering.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn make_xr_display(&mut self, _info: &XrDisplayInfo) -> Result<Display, GPUError> {
        let (xr_instance, session, waiter, stream, swapchain, images, views) =
            openxr_window::create_xr_session(
                &self.instance,
                self.pdevice,
                &self.device,
                self.gfx_queue.family,
            )
            .map_err(|_| GPUError::LibraryError())?;

        let xr_input = openxr_window::XrInput::new(&xr_instance, &session)
            .map_err(|_| GPUError::LibraryError())?;

        Ok(Display {
            xr_instance,
            xr_session: session,
            xr_waiter: waiter,
            xr_stream: stream,
            xr_swapchain: swapchain,
            xr_images: images,
            xr_view_config: views,
            xr_input,
        })
    }

    #[cfg(not(feature = "dashi-openxr"))]
    /// Acquires the next image from a windowed swapchain.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn acquire_new_image(
        &mut self,
        dsp: &mut Display,
    ) -> Result<(ImageView, Handle<Semaphore>, u32, bool), GPUError> {
        if dsp.needs_rebuild.get() {
            return Err(GPUError::DisplayNeedsRebuild);
        }

        let signal_sem_handle = dsp.semaphores[dsp.frame_idx as usize];
        let fence = dsp.fences[dsp.frame_idx as usize];

        self.wait(fence)?;

        let signal_sem = self.semaphores.get_ref(signal_sem_handle).unwrap();
        let acquire_result = unsafe {
            dsp.sc_loader.acquire_next_image2(
                &vk::AcquireNextImageInfoKHR::builder()
                    .swapchain(dsp.swapchain)
                    .semaphore(signal_sem.raw)
                    .fence(self.fences.get_ref(fence).unwrap().raw)
                    .timeout(std::u64::MAX)
                    .device_mask(0x1)
                    .build(),
            )
        };

        let (image_index, suboptimal) = match acquire_result {
            Ok(result) if result.1 => {
                dsp.needs_rebuild.set(true);
                return Err(GPUError::DisplayNeedsRebuild);
            }
            Ok(result) => result,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                dsp.needs_rebuild.set(true);
                return Err(GPUError::DisplayNeedsRebuild);
            }
            Err(err) => return Err(err.into()),
        };

        dsp.frame_idx = image_index;
        let view = ImageView {
            img: dsp.images[image_index as usize],
            range: Default::default(),
            aspect: Default::default(),
            view_type: ImageViewType::Type2D,
        };
        Ok((view, signal_sem_handle, image_index, suboptimal))
    }

    #[cfg(not(feature = "dashi-openxr"))]
    /// Presents the current swapchain image to the display.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn present_display(
        &mut self,
        dsp: &Display,
        wait_sems: &[Handle<Semaphore>],
    ) -> Result<(), GPUError> {
        if dsp.needs_rebuild.get() {
            return Err(GPUError::DisplayNeedsRebuild);
        }

        let raw_wait_sems: Vec<vk::Semaphore> = wait_sems
            .iter()
            .map(|sem| self.semaphores.get_ref(*sem).unwrap().raw)
            .collect();

        let present_result = unsafe {
            dsp.sc_loader.queue_present(
                self.gfx_queue.queue,
                &vk::PresentInfoKHR::builder()
                    .image_indices(&[dsp.frame_idx])
                    .swapchains(&[dsp.swapchain])
                    .wait_semaphores(&raw_wait_sems)
                    .build(),
            )
        };

        match present_result {
            Ok(_) => Ok(()),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                dsp.needs_rebuild.set(true);
                Err(GPUError::DisplayNeedsRebuild)
            }
            Err(err) => Err(err.into()),
        }
    }

    #[cfg(feature = "dashi-openxr")]
    /// Acquires the next image from the XR swapchain.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn acquire_xr_image(
        &mut self,
        dsp: &mut Display,
    ) -> Result<(u32, xr::FrameState), GPUError> {
        let state = dsp.xr_waiter.wait().map_err(|_| GPUError::LibraryError())?;
        dsp.xr_stream
            .begin()
            .map_err(|_| GPUError::LibraryError())?;
        let idx = dsp
            .xr_swapchain
            .acquire_image()
            .map_err(|_| GPUError::LibraryError())?;
        dsp.xr_swapchain
            .wait_image(xr::Duration::INFINITE)
            .map_err(|_| GPUError::LibraryError())?;
        Ok((idx, state))
    }

    #[cfg(feature = "dashi-openxr")]
    /// Presents the current frame to the XR display.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn present_xr_display(
        &mut self,
        dsp: &mut Display,
        state: xr::FrameState,
    ) -> Result<(), GPUError> {
        dsp.xr_swapchain
            .release_image()
            .map_err(|_| GPUError::LibraryError())?;
        dsp.xr_stream
            .end(
                state.predicted_display_time,
                xr::EnvironmentBlendMode::OPAQUE,
                &[],
            )
            .map_err(|_| GPUError::LibraryError())?;
        Ok(())
    }
}

#[cfg(all(test, feature = "dashi-winit", not(feature = "dashi-openxr")))]
mod tests {
    use super::*;

    #[test]
    fn unchanged_size_yields_ready() {
        let action = classify_prepare_action([1280, 720], [1280, 720], false, false, false);

        assert_eq!(
            action,
            PrepareAction {
                status: Some(DisplayStatus::Ready { size: [1280, 720] }),
                should_rebuild: false,
                wait_for_restore: false,
            }
        );
    }

    #[test]
    fn changed_size_yields_resized_and_rebuild() {
        let action = classify_prepare_action([1280, 720], [1600, 900], false, false, false);

        assert_eq!(
            action,
            PrepareAction {
                status: Some(DisplayStatus::Resized { size: [1600, 900] }),
                should_rebuild: true,
                wait_for_restore: false,
            }
        );
    }

    #[test]
    fn dirty_same_size_rebuilds_without_resized_status() {
        let action = classify_prepare_action([1280, 720], [1280, 720], true, false, false);

        assert_eq!(
            action,
            PrepareAction {
                status: Some(DisplayStatus::Ready { size: [1280, 720] }),
                should_rebuild: true,
                wait_for_restore: false,
            }
        );
    }

    #[test]
    fn zero_sized_window_waits_for_restore() {
        let action = classify_prepare_action([1280, 720], [0, 0], true, false, true);

        assert_eq!(
            action,
            PrepareAction {
                status: None,
                should_rebuild: false,
                wait_for_restore: true,
            }
        );
    }

    #[test]
    fn close_request_wins_over_resize_state() {
        let action = classify_prepare_action([1280, 720], [1600, 900], true, true, false);

        assert_eq!(
            action,
            PrepareAction {
                status: Some(DisplayStatus::Closed),
                should_rebuild: false,
                wait_for_restore: false,
            }
        );
    }

    #[test]
    fn minimized_window_waits_for_restore_without_rebuild() {
        let action = classify_prepare_action([1280, 720], [1280, 720], true, false, true);

        assert_eq!(
            action,
            PrepareAction {
                status: None,
                should_rebuild: false,
                wait_for_restore: true,
            }
        );
    }
}
