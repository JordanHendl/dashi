#![cfg(all(
    any(target_os = "windows", target_os = "linux"),
    feature = "dashi-winit",
    not(feature = "dashi-openxr")
))]

#[cfg(target_os = "windows")]
mod common;

#[cfg(target_os = "windows")]
use common::ValidationContext;
use dashi::{Context, ContextInfo, DisplayBuilder, DisplayStatus, MonitorSelection, WindowMode};
#[cfg(target_os = "windows")]
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use serial_test::serial;
#[cfg(target_os = "windows")]
use std::mem::size_of;
#[cfg(target_os = "windows")]
use std::thread;
#[cfg(target_os = "windows")]
use std::time::{Duration, Instant};
#[cfg(target_os = "windows")]
use windows_sys::Win32::Foundation::RECT;
#[cfg(target_os = "windows")]
use windows_sys::Win32::Graphics::Gdi::{
    GetMonitorInfoW, MonitorFromWindow, MONITORINFO, MONITOR_DEFAULTTONULL,
};
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetWindowLongPtrW, GetWindowRect, ShowWindowAsync, GWL_STYLE, MONITORINFOF_PRIMARY, SW_RESTORE,
    WS_CAPTION, WS_THICKFRAME,
};
use winit::window::Fullscreen;

#[cfg(target_os = "windows")]
fn wait_for_status<F>(
    ctx: &mut ValidationContext,
    display: &mut dashi::Display,
    timeout: Duration,
    predicate: F,
) -> DisplayStatus
where
    F: Fn(DisplayStatus) -> bool,
{
    let deadline = Instant::now() + timeout;
    loop {
        let status = ctx.prepare_display(display).expect("prepare_display");
        if predicate(status) {
            return status;
        }

        assert!(
            Instant::now() < deadline,
            "timed out waiting for expected display status, last status: {:?}",
            status
        );
        thread::sleep(Duration::from_millis(16));
    }
}

#[cfg(target_os = "windows")]
fn hwnd_from_window(window: &winit::window::Window) -> isize {
    match window.raw_window_handle() {
        RawWindowHandle::Win32(handle) => handle.hwnd as isize,
        other => panic!("expected Win32 window handle, got {:?}", other),
    }
}

#[cfg(target_os = "windows")]
fn rect_bounds(rect: RECT) -> [i32; 4] {
    [rect.left, rect.top, rect.right, rect.bottom]
}

#[test]
#[ignore]
#[serial]
#[cfg(target_os = "windows")]
fn prepare_display_handles_resize_and_minimize() {
    let mut ctx = ValidationContext::windowed(&ContextInfo::default()).expect("windowed context");
    let mut display = DisplayBuilder::new()
        .title("winit_window_controls")
        .size(640, 480)
        .resizable(true)
        .build(&mut ctx)
        .expect("display");

    let _ = ctx.prepare_display(&mut display).expect("initial prepare");

    display.set_size(800, 600);
    let resized = wait_for_status(
        &mut ctx,
        &mut display,
        Duration::from_secs(5),
        |status| matches!(status, DisplayStatus::Resized { size } if size == [800, 600]),
    );
    assert_eq!(resized, DisplayStatus::Resized { size: [800, 600] });

    let hwnd = hwnd_from_window(display.winit_window());
    let restore_thread = thread::spawn(move || {
        thread::sleep(Duration::from_millis(500));
        unsafe {
            ShowWindowAsync(hwnd as _, SW_RESTORE);
        }
    });

    display.minimize();
    let wait_started = Instant::now();
    let status = ctx
        .prepare_display(&mut display)
        .expect("prepare after minimize");
    restore_thread.join().expect("restore thread");

    assert!(
        wait_started.elapsed() >= Duration::from_millis(200),
        "prepare_display should wait for the minimized window to restore"
    );
    assert!(matches!(
        status,
        DisplayStatus::Ready { size: [800, 600] } | DisplayStatus::Resized { size: [800, 600] }
    ));

    ctx.destroy_display(display);
}

#[test]
#[ignore]
#[serial]
fn borderless_primary_uses_primary_physical_extent_without_exclusive_mode() {
    #[cfg(target_os = "windows")]
    {
        let mut ctx =
            ValidationContext::windowed(&ContextInfo::default()).expect("windowed context");
        assert_borderless_primary_extent(&mut ctx);
    }

    #[cfg(target_os = "linux")]
    {
        let mut ctx = Context::new(&ContextInfo::default()).expect("windowed context");
        assert_borderless_primary_extent(&mut ctx);
        ctx.destroy();
    }
}

fn assert_borderless_primary_extent(ctx: &mut Context) {
    let mut display = DisplayBuilder::new()
        .title("borderless_primary")
        .size(320, 180)
        .resizable(true)
        .window_mode(WindowMode::BorderlessFullscreen)
        .monitor_selection(MonitorSelection::Primary)
        .build(ctx)
        .expect("borderless primary display");

    let window = display.winit_window();
    let primary = window
        .primary_monitor()
        .expect("the window system should identify a primary monitor");
    let primary_size = primary.size();
    let physical_extent = [primary_size.width, primary_size.height];
    assert_ne!(physical_extent, [320, 180]);
    assert_eq!(
        window.inner_size(),
        winit::dpi::PhysicalSize::new(primary_size.width, primary_size.height)
    );
    assert_eq!(display.size(), physical_extent);
    assert!(matches!(
        window.fullscreen(),
        Some(Fullscreen::Borderless(Some(_)))
    ));

    #[cfg(target_os = "windows")]
    {
        let hwnd = hwnd_from_window(window);
        let style = unsafe { GetWindowLongPtrW(hwnd as _, GWL_STYLE) } as u32;
        assert_eq!(
            style & WS_CAPTION,
            0,
            "borderless window must not have a caption"
        );
        assert_eq!(
            style & WS_THICKFRAME,
            0,
            "borderless window must not have a resize border"
        );

        let monitor = unsafe { MonitorFromWindow(hwnd as _, MONITOR_DEFAULTTONULL) };
        assert!(!monitor.is_null(), "borderless window must be on a monitor");
        let mut monitor_info = MONITORINFO {
            cbSize: size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        assert_ne!(
            unsafe { GetMonitorInfoW(monitor, &mut monitor_info) },
            0,
            "primary monitor information must be queryable"
        );
        assert_ne!(
            monitor_info.dwFlags & MONITORINFOF_PRIMARY,
            0,
            "borderless window must use the operating system primary monitor"
        );

        let mut window_rect = RECT::default();
        assert_ne!(
            unsafe { GetWindowRect(hwnd as _, &mut window_rect) },
            0,
            "borderless window bounds must be queryable"
        );
        let window_bounds = rect_bounds(window_rect);
        let monitor_bounds = rect_bounds(monitor_info.rcMonitor);
        let work_bounds = rect_bounds(monitor_info.rcWork);
        assert_eq!(
            window_bounds, monitor_bounds,
            "borderless window must cover the monitor's complete physical bounds"
        );
        if monitor_bounds != work_bounds {
            assert_ne!(
                window_bounds, work_bounds,
                "borderless window must include the taskbar area"
            );
        }
    }

    let status = ctx
        .prepare_display(&mut display)
        .expect("prepare borderless display");
    assert_eq!(
        status,
        DisplayStatus::Ready {
            size: physical_extent
        },
        "the initial swapchain must already match the primary monitor extent"
    );

    ctx.destroy_display(display);
}
