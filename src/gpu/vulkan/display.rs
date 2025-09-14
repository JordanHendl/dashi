use crate::utils::Handle;
use ash::vk;

use super::{
    Context, DisplayInfo, Fence, GPUError, Image, ImageView, Semaphore, WindowBuffering,
    WindowInfo,
};

#[cfg(feature = "dashi-openxr")]
use super::{openxr_window, XrDisplayInfo};
#[cfg(feature = "dashi-openxr")]
use openxr as xr;

#[cfg(all(feature = "dashi-minifb", not(feature = "dashi-openxr")))]
use super::minifb_window;
#[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
use super::winit_window;

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
    pub(crate) xr_images: Vec<vk::Image>,
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
    pub fn xr_swapchain_images(&self) -> &[vk::Image] {
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

impl Context {
    #[cfg(not(feature = "dashi-openxr"))]
    /// Destroys a windowing display and its swapchain.
    ///
    /// # Prerequisites
    /// - Ensure the GPU has finished using all swapchain images.
    /// - Image views should be destroyed before their images and swapchain.
    /// - The context must still be alive.
    pub fn destroy_display(&mut self, dsp: Display) {
        for img in &dsp.images {
            self.images.release(*img);
        }

        for view in dsp.views {
            unsafe { self.device.destroy_image_view(view, None) };
        }
        unsafe { dsp.sc_loader.destroy_swapchain(dsp.swapchain, None) };
        unsafe { dsp.loader.destroy_surface(dsp.surface, None) };
        for sem in dsp.semaphores {
            self.destroy_semaphore(sem);
        }

        for fence in &dsp.fences {
            self.destroy_fence(fence.clone());
        }
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
    ) -> (winit::event_loop::EventLoop<()>, winit::window::Window, vk::SurfaceKHR) {
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
        let capabilities = unsafe {
            loader.get_physical_device_surface_capabilities(self.pdevice, surface)?
        };
        let _formats = unsafe {
            loader.get_physical_device_surface_formats(self.pdevice, surface)?
        };
        let _present_modes = unsafe {
            loader.get_physical_device_surface_present_modes(self.pdevice, surface)?
        };

        // Choose extent
        let size = info.window.size;
        let mut chosen_extent = vk::Extent2D {
            width: size[0],
            height: size[1],
        };
        if capabilities.current_extent.width != std::u32::MAX {
            chosen_extent = capabilities.current_extent.clone();
        } else {
            chosen_extent.width = std::cmp::max(
                capabilities.min_image_extent.width,
                std::cmp::min(capabilities.max_image_extent.width, chosen_extent.width),
            );
            chosen_extent.height = std::cmp::max(
                capabilities.min_image_extent.height,
                std::cmp::min(capabilities.max_image_extent.height, chosen_extent.height),
            );
        }

        // Select a present mode.
        let present_mode = if info.vsync {
            vk::PresentModeKHR::FIFO
        } else {
            vk::PresentModeKHR::IMMEDIATE
        };

        let wanted_format = vk::Format::B8G8R8A8_SRGB;
        let wanted_color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;

        let image_usage = vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;

        let num_framebuffers = match info.buffering {
            WindowBuffering::Double => 2,
            WindowBuffering::Triple => 3,
        };
        let swap_loader = ash::extensions::khr::Swapchain::new(&self.instance, &self.device);
        let swapchain = unsafe {
            swap_loader.create_swapchain(
                &vk::SwapchainCreateInfoKHR::builder()
                    .surface(surface)
                    .present_mode(present_mode)
                    .image_format(wanted_format)
                    .image_array_layers(1)
                    .image_color_space(wanted_color_space)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .image_extent(chosen_extent)
                    .image_usage(image_usage)
                    .min_image_count(std::cmp::max(
                        num_framebuffers,
                        capabilities.min_image_count,
                    ))
                    .pre_transform(capabilities.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .build(),
                Default::default(),
            )?
        };

        // Now, we need to make the images!
        let images = unsafe { swap_loader.get_swapchain_images(swapchain)? };
        let mut handles: Vec<Handle<Image>> = Vec::with_capacity(images.len() as usize);
        let mut view_handles: Vec<vk::ImageView> = Vec::with_capacity(images.len() as usize);
        for img in images {
            let raw_img = img;
            match self.images.insert(Image {
                img: raw_img,
                alloc: unsafe { std::mem::MaybeUninit::zeroed().assume_init() },
                layouts: vec![vk::ImageLayout::UNDEFINED],
                sub_layers: vk::ImageSubresourceLayers::builder()
                    .layer_count(1)
                    .mip_level(0)
                    .base_array_layer(0)
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .build(),
                dim: [chosen_extent.width, chosen_extent.height, 1],
                format: super::vk_to_lib_image_format(wanted_format)?,
            }) {
                Some(handle) => {
                    self.oneshot_transition_image_noview(
                        handle,
                        vk::ImageLayout::PRESENT_SRC_KHR,
                    );
                    let sub_range = vk::ImageSubresourceRange::builder()
                        .base_array_layer(0)
                        .layer_count(1)
                        .base_mip_level(0)
                        .level_count(1)
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .build();
                    let h = unsafe {
                        self.device.create_image_view(
                            &vk::ImageViewCreateInfo::builder()
                                .image(raw_img)
                                .format(wanted_format)
                                .subresource_range(sub_range)
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .build(),
                            None,
                        )?
                    };

                    view_handles.push(h);
                    handles.push(handle)
                }
                None => return Err(GPUError::SlotError()),
            };
        }

        let sems = self.make_semaphores(handles.len()).unwrap();
        let mut fences = Vec::with_capacity(handles.len() as usize);
        for _idx in 0..handles.len() {
            fences.push(self.make_fence()?);
        }
        self.init_gpu_timers(handles.len())?;
        Ok(Display {
            window,
            #[cfg(feature = "dashi-winit")]
            event_loop,
            swapchain,
            surface,
            images: handles,
            loader,
            sc_loader: swap_loader,
            frame_idx: 0,
            semaphores: sems,
            fences,
            views: view_handles,
        })
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
        let signal_sem_handle = dsp.semaphores[dsp.frame_idx as usize].clone();
        let fence = dsp.fences[dsp.frame_idx as usize];

        self.wait(fence.clone())?;

        let signal_sem = self.semaphores.get_ref(signal_sem_handle).unwrap();
        let res = unsafe {
            dsp.sc_loader.acquire_next_image2(
                &vk::AcquireNextImageInfoKHR::builder()
                    .swapchain(dsp.swapchain)
                    .semaphore(signal_sem.raw)
                    .fence(
                        self.fences
                            .get_ref(dsp.fences[dsp.frame_idx as usize])
                            .unwrap()
                            .raw,
                    )
                    .timeout(std::u64::MAX)
                    .device_mask(0x1)
                    .build(),
            )
        }?;

        dsp.frame_idx = res.0;
        let view = ImageView { img: dsp.images[res.0 as usize], layer: 0, mip_level: 0, aspect: Default::default() };
        Ok((view, signal_sem_handle, res.0, res.1))
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
        let raw_wait_sems: Vec<vk::Semaphore> = wait_sems
            .iter()
            .map(|sem| self.semaphores.get_ref(sem.clone()).unwrap().raw)
            .collect();

        unsafe {
            dsp.sc_loader.queue_present(
                self.gfx_queue.queue,
                &vk::PresentInfoKHR::builder()
                    .image_indices(&[dsp.frame_idx])
                    .swapchains(&[dsp.swapchain])
                    .wait_semaphores(&raw_wait_sems)
                    .build(),
            )?;
        }

        Ok(())
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
        let state = dsp
            .xr_waiter
            .wait()
            .map_err(|_| GPUError::LibraryError())?;
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
            .end(state.predicted_display_time, xr::EnvironmentBlendMode::OPAQUE, &[])
            .map_err(|_| GPUError::LibraryError())?;
        Ok(())
    }
}

