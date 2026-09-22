use super::error::GPUError;
use crate::gpu::structs::WindowInfo;
use ash::{vk, Entry, Instance};
use minifb::{Window, WindowOptions};

pub(super) fn create_window(
    entry: &Entry,
    instance: &Instance,
    info: &WindowInfo,
    allocation_callbacks: Option<&vk::AllocationCallbacks>,
) -> Result<(Window, vk::SurfaceKHR), GPUError> {
    let mut opts = WindowOptions::default();
    opts.resize = info.resizable;
    let window = Window::new(
        &info.title,
        info.size[0] as usize,
        info.size[1] as usize,
        opts,
    )
    .map_err(|err| GPUError::LibraryError(format!("failed to create minifb window: {err}")))?;

    let surface =
        unsafe { ash_window::create_surface(entry, instance, &window, allocation_callbacks)? };

    Ok((window, surface))
}
