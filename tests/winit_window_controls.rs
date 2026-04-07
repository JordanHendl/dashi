#![cfg(all(
    target_os = "windows",
    feature = "dashi-winit",
    not(feature = "dashi-openxr")
))]

mod common;

use common::ValidationContext;
use dashi::{ContextInfo, DisplayBuilder, DisplayStatus};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use serial_test::serial;
use std::thread;
use std::time::{Duration, Instant};
use windows_sys::Win32::UI::WindowsAndMessaging::{ShowWindowAsync, SW_RESTORE};

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
