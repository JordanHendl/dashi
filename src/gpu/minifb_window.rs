use crate::gpu::structs::WindowInfo;
use crate::gpu::error::GPUError;
use ash::{vk, Entry, Instance};
use minifb::{Window, WindowOptions};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub(super) fn create_window(
    entry: &Entry,
    instance: &Instance,
    info: &WindowInfo,
) -> Result<(Window, vk::SurfaceKHR), GPUError> {
    let mut opts = WindowOptions::default();
    opts.resize = info.resizable;
    let window = Window::new(&info.title, info.size[0] as usize, info.size[1] as usize, opts)
        .map_err(|_| GPUError::LibraryError())?;

    let surface = unsafe {
        ash_window::create_surface(
            entry,
            instance,
            window.display_handle().map_err(|_| GPUError::LibraryError())?,
            window.window_handle().map_err(|_| GPUError::LibraryError())?,
            None,
        )?
    };

    Ok((window, surface))
}
