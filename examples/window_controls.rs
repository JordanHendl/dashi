use dashi::*;
use std::thread;
use std::time::{Duration, Instant};

#[cfg(target_os = "windows")]
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
#[cfg(target_os = "windows")]
use windows_sys::Win32::UI::WindowsAndMessaging::{ShowWindowAsync, SW_RESTORE};

#[cfg(target_os = "windows")]
fn schedule_restore(window: &winit::window::Window, delay: Duration) {
    let hwnd = match window.raw_window_handle() {
        RawWindowHandle::Win32(handle) => handle.hwnd as isize,
        other => panic!("expected Win32 window handle, got {:?}", other),
    };

    thread::spawn(move || {
        thread::sleep(delay);
        unsafe {
            ShowWindowAsync(hwnd as _, SW_RESTORE);
        }
    });
}

fn main() -> Result<(), GPUError> {
    let mut ctx = gpu::Context::new(&ContextInfo::default())?;
    let mut display = DisplayBuilder::new()
        .title("Window Controls")
        .size(800, 600)
        .resizable(true)
        .build(&mut ctx)?;

    let started = Instant::now();
    let mut resized = false;
    #[cfg(target_os = "windows")]
    let mut minimized = false;
    #[cfg(target_os = "windows")]
    let mut restore_called = false;

    loop {
        match ctx.prepare_display(&mut display)? {
            DisplayStatus::Closed => break,
            DisplayStatus::Ready { .. } => {}
            DisplayStatus::Resized { size } => {
                println!("window resized to {}x{}", size[0], size[1]);
            }
        }

        if !resized && started.elapsed() >= Duration::from_secs(1) {
            display.set_size(1024, 720);
            resized = true;
            println!("requested 1024x720");
        }

        #[cfg(target_os = "windows")]
        {
            if !minimized && started.elapsed() >= Duration::from_secs(2) {
                schedule_restore(display.winit_window(), Duration::from_secs(1));
                display.minimize();
                minimized = true;
                println!("minimized window; prepare_display will wait until restore");
            } else if minimized && !restore_called && started.elapsed() >= Duration::from_secs(3) {
                display.restore();
                restore_called = true;
                println!("restore requested");
            }
        }

        thread::sleep(Duration::from_millis(16));
    }

    ctx.destroy_display(display);
    ctx.destroy();
    Ok(())
}
