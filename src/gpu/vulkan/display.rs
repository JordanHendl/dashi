use crate::utils::Handle;
use ash::vk;
#[cfg(feature = "dashi-winit")]
use std::{thread, time::Duration};

use super::{
    DisplayInfo, DisplayStatus, Fence, GPUError, Image, ImageInfo, ImageInfoRecord, ImageView,
    ImageViewType, SampleCount, Semaphore, VulkanContext, WindowBuffering, WindowMode,
};

#[cfg(feature = "dashi-minifb")]
use super::minifb_window;
#[cfg(feature = "dashi-winit")]
use super::winit_window;
#[cfg(feature = "dashi-winit")]
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::ControlFlow,
    platform::run_return::EventLoopExtRunReturn,
};

#[cfg(feature = "dashi-winit")]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct WinitEventBatch {
    close_requested: bool,
    resized: bool,
    scale_factor_changed: bool,
}

#[cfg(feature = "dashi-winit")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PrepareAction {
    status: Option<DisplayStatus>,
    should_rebuild: bool,
    wait_for_restore: bool,
}

#[cfg(feature = "dashi-winit")]
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

struct SwapchainConfig {
    extent: vk::Extent2D,
    surface_format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    min_image_count: u32,
    pre_transform: vk::SurfaceTransformFlagsKHR,
}

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
    #[cfg(feature = "dashi-sdl2")]
    pub(crate) window: std::cell::Cell<sdl2::video::Window>,
    #[cfg(feature = "dashi-minifb")]
    pub(crate) window: minifb::Window,
    #[cfg(feature = "dashi-winit")]
    pub(crate) window: winit::window::Window,
    #[cfg(feature = "dashi-winit")]
    pub(crate) event_loop: winit::event_loop::EventLoop<()>,
    pub(crate) swapchain: ash::vk::SwapchainKHR,
    pub(crate) surface: ash::vk::SurfaceKHR,
    pub(crate) images: Vec<Handle<Image>>,
    pub(crate) views: Vec<vk::ImageView>,
    pub(crate) loader: ash::extensions::khr::Surface,
    pub(crate) sc_loader: ash::extensions::khr::Swapchain,
    pub(crate) semaphores: Vec<Handle<Semaphore>>,
    pub(crate) fences: Vec<Handle<Fence>>,
    pub(crate) frame_idx: u32,
    pub(crate) info: DisplayInfo,
    pub(crate) extent: vk::Extent2D,
    pub(crate) needs_rebuild: std::cell::Cell<bool>,
    pub(crate) closed: bool,
    #[cfg(feature = "dashi-winit")]
    pub(crate) minimized: bool,

}

impl Display {
    #[cfg(feature = "dashi-minifb")]
    pub fn minifb_window(&mut self) -> &mut minifb::Window {
        &mut self.window
    }

    #[cfg(feature = "dashi-winit")]
    pub fn winit_window(&self) -> &winit::window::Window {
        &self.window
    }

    #[cfg(feature = "dashi-winit")]
    pub fn winit_event_loop(&mut self) -> &mut winit::event_loop::EventLoop<()> {
        &mut self.event_loop
    }

    #[cfg(feature = "dashi-winit")]
    pub fn set_resizable(&mut self, resizable: bool) {
        self.info.window.resizable = resizable;
        self.window.set_resizable(resizable);
    }

    #[cfg(feature = "dashi-winit")]
    pub fn set_size(&mut self, width: u32, height: u32) {
        self.info.window.size = [width, height];
        self.window.set_inner_size(PhysicalSize::new(width, height));
    }

    #[cfg(feature = "dashi-winit")]
    pub fn size(&self) -> [u32; 2] {
        let size = self.window.inner_size();
        [size.width, size.height]
    }

    #[cfg(feature = "dashi-winit")]
    pub fn minimize(&mut self) {
        self.minimized = true;
        self.window.set_minimized(true);
    }

    #[cfg(feature = "dashi-winit")]
    pub fn restore(&mut self) {
        self.minimized = false;
        self.window.set_minimized(false);
    }

    #[cfg(feature = "dashi-winit")]
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
        target_os = "windows"
    ))]
    fn is_os_minimized(&self) -> bool {
        false
    }

    #[cfg(all(
        feature = "dashi-winit",
        not(target_os = "windows")
    ))]
    fn is_os_minimized(&self) -> bool {
        false
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }

}

impl VulkanContext {
    fn release_swapchain_images(
        &mut self,
        handles: &mut Vec<Handle<Image>>,
        views: &mut Vec<vk::ImageView>,
    ) {
        for view in views.drain(..) {
            unsafe {
                self.device
                    .destroy_image_view(view, self.allocation_callbacks.as_deref())
            };
        }

        for handle in handles.drain(..) {
            if let Some(image) = self.images.get_ref(handle) {
                self.image_infos.release(image.info_handle);
            }
            self.images.release(handle);
        }
    }

    fn destroy_swapchain_resources(&mut self, dsp: &mut Display) {
        self.release_swapchain_images(&mut dsp.images, &mut dsp.views);

        if dsp.swapchain != vk::SwapchainKHR::null() {
            unsafe {
                dsp.sc_loader
                    .destroy_swapchain(dsp.swapchain, self.allocation_callbacks.as_deref())
            };
            dsp.swapchain = vk::SwapchainKHR::null();
        }

        for sem in dsp.semaphores.drain(..) {
            self.destroy_semaphore(sem);
        }

        for fence in dsp.fences.drain(..) {
            self.destroy_fence(fence);
        }
    }

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
                alloc: super::allocation::Allocation::Borrowed,
                externally_owned: false,
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
                    self.allocation_callbacks.as_deref(),
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
                self.allocation_callbacks.as_deref(),
            )?
        };

        let raw_images = unsafe { sc_loader.get_swapchain_images(swapchain)? };
        let (mut images, mut views) =
            self.wrap_swapchain_images(raw_images, config.surface_format.format, config.extent)?;

        let semaphores = match self.make_semaphores(images.len()) {
            Ok(semaphores) => semaphores,
            Err(err) => {
                self.release_swapchain_images(&mut images, &mut views);
                unsafe {
                    sc_loader.destroy_swapchain(swapchain, self.allocation_callbacks.as_deref())
                };
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
                    unsafe {
                        sc_loader.destroy_swapchain(swapchain, self.allocation_callbacks.as_deref())
                    };
                    return Err(err);
                }
            }
        }

        if let Err(err) = self.ensure_gpu_timers(images.len()) {
            for sem in semaphores {
                self.destroy_semaphore(sem);
            }
            for fence in fences {
                self.destroy_fence(fence);
            }
            self.release_swapchain_images(&mut images, &mut views);
            unsafe { sc_loader.destroy_swapchain(swapchain, self.allocation_callbacks.as_deref()) };
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

    /// Destroys a windowing display and its swapchain.
    ///
    /// # Prerequisites
    /// - Ensure the GPU has finished using all swapchain images.
    /// - Image views should be destroyed before their images and swapchain.
    /// - The context must still be alive.
    pub fn destroy_display(&mut self, mut dsp: Display) {
        self.destroy_swapchain_resources(&mut dsp);
        // SDL creates its surface internally without accepting allocation callbacks.
        #[cfg(feature = "dashi-sdl2")]
        let surface_callbacks = None;
        #[cfg(not(feature = "dashi-sdl2"))]
        let surface_callbacks = self.allocation_callbacks.as_deref();
        unsafe { dsp.loader.destroy_surface(dsp.surface, surface_callbacks) };
    }

    #[cfg(feature = "dashi-sdl2")]
    fn make_window(
        &mut self,
        info: &DisplayInfo,
    ) -> Result<
        (
            std::cell::Cell<sdl2::video::Window>,
            vk::SurfaceKHR,
            [u32; 2],
        ),
        GPUError,
    > {
        if info.window_mode != WindowMode::Windowed {
            return Err(GPUError::UnsupportedWindowMode {
                backend: "SDL2",
                mode: info.window_mode,
            });
        }
        let window_info = &info.window;
        let mut window = std::cell::Cell::new(
            self.sdl_video
                .as_ref()
                .ok_or_else(|| {
                    GPUError::LibraryError("SDL2 video subsystem is unavailable".into())
                })?
                .window(&window_info.title, window_info.size[0], window_info.size[1])
                .vulkan()
                .build()
                .map_err(|err| {
                    GPUError::LibraryError(format!("failed to create SDL2 window: {err}"))
                })?,
        );
        let actual_size = window.get_mut().size();

        let surface = window
            .get_mut()
            .vulkan_create_surface(vk::Handle::as_raw(self.instance.handle()) as usize)
            .map_err(|err| {
                GPUError::LibraryError(format!("failed to create SDL2 Vulkan surface: {err}"))
            })?;

        Ok((
            window,
            vk::Handle::from_raw(surface),
            [actual_size.0, actual_size.1],
        ))
    }

    #[cfg(feature = "dashi-minifb")]
    fn make_window(
        &mut self,
        info: &DisplayInfo,
    ) -> Result<(minifb::Window, vk::SurfaceKHR, [u32; 2]), GPUError> {
        if info.window_mode != WindowMode::Windowed {
            return Err(GPUError::UnsupportedWindowMode {
                backend: "minifb",
                mode: info.window_mode,
            });
        }
        let (window, surface) = minifb_window::create_window(
            &self.entry,
            &self.instance,
            &info.window,
            self.allocation_callbacks.as_deref(),
        )?;
        let actual_size = window.get_size();
        Ok((
            window,
            surface,
            [actual_size.0 as u32, actual_size.1 as u32],
        ))
    }

    #[cfg(feature = "dashi-winit")]
    fn make_window(
        &mut self,
        info: &DisplayInfo,
    ) -> Result<
        (
            winit::event_loop::EventLoop<()>,
            winit::window::Window,
            vk::SurfaceKHR,
            [u32; 2],
        ),
        GPUError,
    > {
        winit_window::create_window(
            &self.entry,
            &self.instance,
            info,
            self.allocation_callbacks.as_deref(),
        )
    }

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
        let (event_loop, window, surface, actual_size) = self.make_window(info)?;
        #[cfg(not(feature = "dashi-winit"))]
        let (window, surface, actual_size) = self.make_window(info)?;

        let mut effective_info = info.clone();
        effective_info.window.size = actual_size;
        if effective_info.window_mode == WindowMode::BorderlessFullscreen {
            effective_info.window.resizable = false;
        }

        let loader = ash::extensions::khr::Surface::new(&self.entry, &self.instance);
        let resources =
            self.create_swapchain_resources(surface, &loader, &effective_info, actual_size)?;

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
            info: effective_info,
            extent: resources.extent,
            needs_rebuild: std::cell::Cell::new(false),
            closed: false,
            #[cfg(feature = "dashi-winit")]
            minimized: false,
        })
    }

    #[cfg(feature = "dashi-winit")]
    pub fn prepare_display_from_state(
        &mut self,
        dsp: &mut Display,
        close_requested: bool,
    ) -> Result<Option<DisplayStatus>, GPUError> {
        if close_requested {
            dsp.closed = true;
        }

        if dsp.closed {
            return Ok(Some(DisplayStatus::Closed));
        }

        let size = dsp.size();
        let minimized = size[0] == 0 || size[1] == 0 || dsp.is_os_minimized();
        let action = classify_prepare_action(
            [dsp.extent.width, dsp.extent.height],
            size,
            dsp.needs_rebuild.get(),
            close_requested,
            minimized,
        );

        if action.wait_for_restore {
            dsp.minimized = true;
            return Ok(None);
        }

        dsp.minimized = false;
        dsp.info.window.size = size;

        if action.should_rebuild {
            self.rebuild_display_swapchain(dsp, size)?;
        }

        Ok(action.status)
    }

    #[cfg(feature = "dashi-winit")]
    pub fn prepare_display(&mut self, dsp: &mut Display) -> Result<DisplayStatus, GPUError> {
        loop {
            let batch = dsp.pump_events(dsp.minimized);
            match self.prepare_display_from_state(dsp, batch.close_requested)? {
                Some(status) => return Ok(status),
                None => {
                    dsp.minimized = true;
                    thread::sleep(Duration::from_millis(16));
                }
            }
        }
    }

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
            // Acquire uses an infinite timeout, so NOT_READY/TIMEOUT means the swapchain state
            // is no longer making forward progress. Force a rebuild instead of spinning forever.
            Err(vk::Result::NOT_READY | vk::Result::TIMEOUT) => {
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
            Ok(_) => {
                #[cfg(feature = "dashi-winit")]
                dsp.window.request_redraw();
                Ok(())
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR) => {
                dsp.needs_rebuild.set(true);
                Err(GPUError::DisplayNeedsRebuild)
            }
            Err(err) => Err(err.into()),
        }
    }

}

#[cfg(all(test, feature = "dashi-winit"))]
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
