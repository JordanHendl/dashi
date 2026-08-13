use super::error::GPUError;
use crate::gpu::structs::{DisplayInfo, MonitorSelection, WindowMode};
use ash::{vk, Entry, Instance};
use winit::dpi::PhysicalSize;
use winit::event_loop::EventLoop;
#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopExtWindows;
use winit::window::{Fullscreen, WindowBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowCreationMode {
    Windowed,
    BorderlessSystemDefault,
    BorderlessPrimary,
    BorderlessPrimaryFallback,
}

const PRIMARY_MONITOR_FALLBACK_WARNING: &str =
  "primary monitor information was unavailable; falling back to the OS-selected/default monitor for borderless fullscreen";

impl WindowCreationMode {
    fn fallback_warning(self) -> Option<&'static str> {
        match self {
            Self::BorderlessPrimaryFallback => Some(PRIMARY_MONITOR_FALLBACK_WARNING),
            _ => None,
        }
    }
}

fn classify_window_creation(
    window_mode: WindowMode,
    monitor_selection: MonitorSelection,
    primary_monitor_available: bool,
) -> WindowCreationMode {
    match (window_mode, monitor_selection) {
        (WindowMode::Windowed, _) => WindowCreationMode::Windowed,
        (WindowMode::BorderlessFullscreen, MonitorSelection::SystemDefault) => {
            WindowCreationMode::BorderlessSystemDefault
        }
        (WindowMode::BorderlessFullscreen, MonitorSelection::Primary)
            if primary_monitor_available =>
        {
            WindowCreationMode::BorderlessPrimary
        }
        (WindowMode::BorderlessFullscreen, MonitorSelection::Primary) => {
            WindowCreationMode::BorderlessPrimaryFallback
        }
    }
}

pub(super) fn create_window(
    entry: &Entry,
    instance: &Instance,
    info: &DisplayInfo,
) -> Result<
    (
        EventLoop<()>,
        winit::window::Window,
        vk::SurfaceKHR,
        [u32; 2],
    ),
    GPUError,
> {
    #[cfg(target_os = "windows")]
    let event_loop = EventLoop::new_any_thread();
    #[cfg(not(target_os = "windows"))]
    let event_loop = EventLoop::new();

    let primary_monitor = if info.window_mode == WindowMode::BorderlessFullscreen
        && info.monitor_selection == MonitorSelection::Primary
    {
        event_loop.primary_monitor()
    } else {
        None
    };
    let creation_mode = classify_window_creation(
        info.window_mode,
        info.monitor_selection,
        primary_monitor.is_some(),
    );
    if let Some(warning) = creation_mode.fallback_warning() {
        tracing::warn!("{warning}");
    }

    let builder = WindowBuilder::new().with_title(info.window.title.clone());
    let builder = match creation_mode {
        WindowCreationMode::Windowed => builder
            .with_inner_size(PhysicalSize::new(info.window.size[0], info.window.size[1]))
            .with_resizable(info.window.resizable),
        WindowCreationMode::BorderlessSystemDefault
        | WindowCreationMode::BorderlessPrimary
        | WindowCreationMode::BorderlessPrimaryFallback => {
            let monitor = if creation_mode == WindowCreationMode::BorderlessPrimary {
                primary_monitor
            } else {
                None
            };
            builder
                .with_decorations(false)
                .with_resizable(false)
                .with_fullscreen(Some(Fullscreen::Borderless(monitor)))
        }
    };

    let window = builder.build(&event_loop).map_err(|err| {
        GPUError::LibraryError(format!(
            "failed to initialize winit window: {err}; params: {info:?}"
        ))
    })?;
    let actual_size = window.inner_size();
    if actual_size.width == 0 || actual_size.height == 0 {
        return Err(GPUError::LibraryError(format!(
            "winit created a zero-sized window for params: {info:?}"
        )));
    }
    window.request_redraw();
    let _ = window.focus_window();

    let surface = unsafe { ash_window::create_surface(entry, instance, &window, None)? };

    Ok((
        event_loop,
        window,
        surface,
        [actual_size.width, actual_size.height],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windowed_creation_ignores_monitor_selection() {
        assert_eq!(
            classify_window_creation(WindowMode::Windowed, MonitorSelection::Primary, true),
            WindowCreationMode::Windowed
        );
    }

    #[test]
    fn borderless_system_default_does_not_require_primary_monitor_information() {
        assert_eq!(
            classify_window_creation(
                WindowMode::BorderlessFullscreen,
                MonitorSelection::SystemDefault,
                false,
            ),
            WindowCreationMode::BorderlessSystemDefault
        );
    }

    #[test]
    fn borderless_primary_uses_primary_monitor_when_available() {
        assert_eq!(
            classify_window_creation(
                WindowMode::BorderlessFullscreen,
                MonitorSelection::Primary,
                true,
            ),
            WindowCreationMode::BorderlessPrimary
        );
    }

    #[test]
    fn borderless_primary_falls_back_when_primary_monitor_is_unavailable() {
        let creation_mode = classify_window_creation(
            WindowMode::BorderlessFullscreen,
            MonitorSelection::Primary,
            false,
        );

        assert_eq!(creation_mode, WindowCreationMode::BorderlessPrimaryFallback);
        assert_eq!(
            creation_mode.fallback_warning(),
            Some(PRIMARY_MONITOR_FALLBACK_WARNING)
        );
    }
}
