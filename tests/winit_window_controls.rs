#![cfg(all(
    target_os = "windows",
    feature = "dashi-winit",
    not(feature = "dashi-openxr")
))]

mod common;

use common::ValidationContext;
use dashi::{ContextInfo, DisplayBuilder, DisplayStatus, MonitorSelection, WindowMode};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use serial_test::serial;
use std::mem::size_of;
use std::thread;
use std::time::{Duration, Instant};
use windows_sys::Win32::Foundation::RECT;
use windows_sys::Win32::Graphics::Gdi::{
    GetMonitorInfoW, MonitorFromWindow, MONITORINFO, MONITOR_DEFAULTTONULL,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetWindowLongPtrW, GetWindowRect, ShowWindowAsync, GWL_STYLE, MONITORINFOF_PRIMARY, SW_RESTORE,
    WS_CAPTION, WS_THICKFRAME,
};
use winit::window::Fullscreen;

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

fn hwnd_from_window(window: &winit::window::Window) -> isize {
    match window.raw_window_handle() {
        RawWindowHandle::Win32(handle) => handle.hwnd as isize,
        other => panic!("expected Win32 window handle, got {:?}", other),
    }
}

fn rect_bounds(rect: RECT) -> [i32; 4] {
    [rect.left, rect.top, rect.right, rect.bottom]
}

#[test]
#[ignore]
#[serial]
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
    let mut ctx = ValidationContext::windowed(&ContextInfo::default()).expect("windowed context");
    let mut display = DisplayBuilder::new()
        .title("borderless_primary")
        .size(320, 180)
        .resizable(true)
        .window_mode(WindowMode::BorderlessFullscreen)
        .monitor_selection(MonitorSelection::Primary)
        .build(&mut ctx)
        .expect("borderless primary display");

    let window = display.winit_window();
    let primary = window
        .primary_monitor()
        .expect("Windows should identify a primary monitor");
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

    let status = ctx
        .prepare_display(&mut display)
        .expect("prepare borderless display");
    assert!(matches!(
        status,
        DisplayStatus::Ready { size } | DisplayStatus::Resized { size }
            if size == physical_extent
    ));

    ctx.destroy_display(display);
}
