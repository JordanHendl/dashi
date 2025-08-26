use super::error::GPUError;
use super::structs::WindowInfo;
use ash::{vk, Entry, Instance};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;
use winit::dpi::PhysicalSize;

pub(super) fn create_window(
    entry: &Entry,
    instance: &Instance,
    info: &WindowInfo,

) -> Result<(EventLoop<()>, winit::window::Window, vk::SurfaceKHR), GPUError> {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(info.title.clone())
        .with_inner_size(PhysicalSize::new(info.size[0], info.size[1]))
        .with_resizable(info.resizable)
        .build(&event_loop)
        .map_err(|_| GPUError::LibraryError())?;

    let surface = unsafe { ash_window::create_surface(entry, instance, &window, None)? };

    Ok((event_loop, window, surface))
}
