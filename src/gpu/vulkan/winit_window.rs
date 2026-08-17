use super::error::GPUError;
use crate::gpu::structs::{DisplayInfo, MonitorSelection, WindowMode};
use ash::{vk, Entry, Instance};
#[cfg(target_os = "linux")]
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use winit::dpi::PhysicalSize;
#[cfg(target_os = "linux")]
use winit::event::Event;
#[cfg(target_os = "linux")]
use winit::event_loop::ControlFlow;
use winit::event_loop::EventLoop;
#[cfg(target_os = "linux")]
use winit::platform::run_return::EventLoopExtRunReturn;
#[cfg(target_os = "linux")]
use winit::platform::unix::EventLoopExtUnix;
#[cfg(target_os = "windows")]
use winit::platform::windows::EventLoopExtWindows;
#[cfg(target_os = "linux")]
use winit::window::Window;
use winit::window::{Fullscreen, WindowBuilder};

#[cfg(target_os = "linux")]
use std::thread;
#[cfg(target_os = "linux")]
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowCreationMode {
    Windowed,
    BorderlessSystemDefault,
    BorderlessPrimary,
    BorderlessPrimaryFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitialExtentSource {
    Authored,
    PrimaryMonitor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InitialExtent {
    size: PhysicalSize<u32>,
    source: InitialExtentSource,
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

fn is_valid_extent(size: PhysicalSize<u32>) -> bool {
    size.width > 0 && size.height > 0
}

fn resolve_initial_extent(
    creation_mode: WindowCreationMode,
    authored_size: PhysicalSize<u32>,
    primary_monitor_size: Option<PhysicalSize<u32>>,
) -> Option<InitialExtent> {
    if creation_mode == WindowCreationMode::BorderlessPrimary {
        if let Some(size) = primary_monitor_size.filter(|size| is_valid_extent(*size)) {
            return Some(InitialExtent {
                size,
                source: InitialExtentSource::PrimaryMonitor,
            });
        }
    }

    is_valid_extent(authored_size).then_some(InitialExtent {
        size: authored_size,
        source: InitialExtentSource::Authored,
    })
}

#[cfg(target_os = "linux")]
fn settle_x11_primary_extent(
    event_loop: &mut EventLoop<()>,
    window: &Window,
    expected_size: PhysicalSize<u32>,
) -> Result<PhysicalSize<u32>, GPUError> {
    if !matches!(
        window.raw_window_handle(),
        RawWindowHandle::Xlib(_) | RawWindowHandle::Xcb(_)
    ) {
        return Ok(window.inner_size());
    }

    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let actual_size = window.inner_size();
        if actual_size == expected_size {
            return Ok(actual_size);
        }

        event_loop.run_return(|event, _target, control_flow| {
            *control_flow = ControlFlow::Poll;
            if matches!(event, Event::MainEventsCleared | Event::LoopDestroyed) {
                *control_flow = ControlFlow::Exit;
            }
        });

        let actual_size = window.inner_size();
        if actual_size == expected_size {
            return Ok(actual_size);
        }
        if Instant::now() >= deadline {
            return Err(GPUError::LibraryError(format!(
                "timed out waiting for X11 borderless-primary extent {}x{}; last observed extent was {}x{}",
                expected_size.width, expected_size.height, actual_size.width, actual_size.height
            )));
        }
        thread::sleep(Duration::from_millis(10));
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
    #[cfg(target_os = "linux")]
    let mut event_loop = EventLoop::new_any_thread();
    #[cfg(target_os = "windows")]
    let event_loop = EventLoop::new_any_thread();
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    let event_loop = EventLoop::new();

    let primary_monitor = if info.window_mode == WindowMode::BorderlessFullscreen
        && info.monitor_selection == MonitorSelection::Primary
    {
        event_loop.primary_monitor()
    } else {
        None
    };
    let primary_monitor_size = primary_monitor.as_ref().map(|monitor| monitor.size());
    let primary_monitor_extent_available =
        primary_monitor_size.map(is_valid_extent).unwrap_or(false);
    let creation_mode = classify_window_creation(
        info.window_mode,
        info.monitor_selection,
        primary_monitor_extent_available,
    );
    if let Some(warning) = creation_mode.fallback_warning() {
        tracing::warn!("{warning}");
    }

    let authored_size = PhysicalSize::new(info.window.size[0], info.window.size[1]);
    let initial_extent = resolve_initial_extent(
        creation_mode,
        authored_size,
        primary_monitor_size,
    )
    .ok_or_else(|| {
        GPUError::LibraryError(format!(
            "failed to resolve a non-zero initial window extent; authored extent={}x{}, primary monitor extent={:?}; params: {info:?}",
            authored_size.width, authored_size.height, primary_monitor_size
        ))
    })?;
    tracing::info!(
        "resolved initial window extent: authored={}x{}, primary_monitor={:?}, selected={}x{}, source={:?}",
        authored_size.width,
        authored_size.height,
        primary_monitor_size,
        initial_extent.size.width,
        initial_extent.size.height,
        initial_extent.source,
    );

    let builder = WindowBuilder::new().with_title(info.window.title.clone());
    let builder = match creation_mode {
        WindowCreationMode::Windowed => builder
            .with_inner_size(initial_extent.size)
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
                .with_inner_size(initial_extent.size)
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
    #[cfg(target_os = "linux")]
    let actual_size = if initial_extent.source == InitialExtentSource::PrimaryMonitor {
        settle_x11_primary_extent(&mut event_loop, &window, initial_extent.size)?
    } else {
        window.inner_size()
    };
    #[cfg(not(target_os = "linux"))]
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

    #[test]
    fn borderless_primary_uses_primary_monitor_physical_extent() {
        let extent = resolve_initial_extent(
            WindowCreationMode::BorderlessPrimary,
            PhysicalSize::new(320, 180),
            Some(PhysicalSize::new(1920, 1080)),
        );

        assert_eq!(
            extent,
            Some(InitialExtent {
                size: PhysicalSize::new(1920, 1080),
                source: InitialExtentSource::PrimaryMonitor,
            })
        );
    }

    #[test]
    fn borderless_system_default_uses_authored_extent() {
        let extent = resolve_initial_extent(
            WindowCreationMode::BorderlessSystemDefault,
            PhysicalSize::new(1366, 768),
            None,
        );

        assert_eq!(
            extent,
            Some(InitialExtent {
                size: PhysicalSize::new(1366, 768),
                source: InitialExtentSource::Authored,
            })
        );
    }

    #[test]
    fn unavailable_primary_monitor_uses_authored_extent() {
        let extent = resolve_initial_extent(
            WindowCreationMode::BorderlessPrimaryFallback,
            PhysicalSize::new(1600, 900),
            None,
        );

        assert_eq!(
            extent,
            Some(InitialExtent {
                size: PhysicalSize::new(1600, 900),
                source: InitialExtentSource::Authored,
            })
        );
    }

    #[test]
    fn invalid_primary_monitor_extent_uses_authored_extent() {
        let extent = resolve_initial_extent(
            WindowCreationMode::BorderlessPrimary,
            PhysicalSize::new(1600, 900),
            Some(PhysicalSize::new(0, 1080)),
        );

        assert_eq!(
            extent,
            Some(InitialExtent {
                size: PhysicalSize::new(1600, 900),
                source: InitialExtentSource::Authored,
            })
        );
    }

    #[test]
    fn initial_extent_must_be_non_zero() {
        assert_eq!(
            resolve_initial_extent(
                WindowCreationMode::BorderlessPrimaryFallback,
                PhysicalSize::new(0, 900),
                None,
            ),
            None
        );
    }
}
