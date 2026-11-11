//! One persistent Winit loop on the native-window owner thread.
use super::GPUError;
use std::{cell::RefCell, collections::HashSet, rc::Rc, time::Duration};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, Event, StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    platform::pump_events::{EventLoopExtPumpEvents, PumpStatus},
    window::{CustomCursor, CustomCursorSource, Window, WindowAttributes, WindowId},
};

scoped_tls::scoped_thread_local!(static ACTIVE_LOOP: ActiveEventLoop);

thread_local! {
    static NATIVE_EVENTS: RefCell<Option<NativeEventLoop>> = const { RefCell::new(None) };
}

#[derive(Clone)]
pub struct NativeEventLoop {
    events: Rc<RefCell<EventLoop<()>>>,
    closed: Rc<RefCell<HashSet<WindowId>>>,
    pending: Rc<RefCell<Vec<Event<()>>>>,
}

struct Handler<'a, F> {
    callback: F,
    closed: &'a RefCell<HashSet<WindowId>>,
    pending: &'a RefCell<Vec<Event<()>>>,
}

impl<F: FnMut(Event<()>, &ActiveEventLoop)> Handler<'_, F> {
    fn dispatch(&mut self, event: Event<()>, active: &ActiveEventLoop) {
        ACTIVE_LOOP.set(active, || {
            let pending = std::mem::take(&mut *self.pending.borrow_mut());
            for event in pending {
                (self.callback)(event, active);
            }
            (self.callback)(event, active);
        });
    }
}

impl<F: FnMut(Event<()>, &ActiveEventLoop)> ApplicationHandler for Handler<'_, F> {
    fn new_events(&mut self, active: &ActiveEventLoop, cause: StartCause) {
        self.dispatch(Event::NewEvents(cause), active);
    }
    fn resumed(&mut self, active: &ActiveEventLoop) {
        self.dispatch(Event::Resumed, active);
    }
    fn suspended(&mut self, active: &ActiveEventLoop) {
        self.dispatch(Event::Suspended, active);
    }
    fn window_event(&mut self, active: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                self.closed.borrow_mut().insert(window_id);
            }
            WindowEvent::Destroyed => {
                self.closed.borrow_mut().remove(&window_id);
            }
            _ => {}
        }
        self.dispatch(Event::WindowEvent { window_id, event }, active);
    }
    fn device_event(&mut self, active: &ActiveEventLoop, device_id: DeviceId, event: DeviceEvent) {
        self.dispatch(Event::DeviceEvent { device_id, event }, active);
    }
    fn about_to_wait(&mut self, active: &ActiveEventLoop) {
        self.dispatch(Event::AboutToWait, active);
    }
    fn exiting(&mut self, active: &ActiveEventLoop) {
        self.dispatch(Event::LoopExiting, active);
    }
}

impl NativeEventLoop {
    pub fn create_window(&self, attributes: WindowAttributes) -> Result<Window, GPUError> {
        self.with_active(|active| {
            active
                .create_window(attributes)
                .map_err(|e| GPUError::LibraryError(e.to_string()))
        })?
    }

    pub fn shared() -> Result<Self, GPUError> {
        NATIVE_EVENTS.with(|slot| {
            let mut slot = slot.borrow_mut();
            if let Some(events) = &*slot {
                return Ok(events.clone());
            }
            let mut builder = EventLoop::builder();
            #[cfg(target_os = "windows")]
            winit::platform::windows::EventLoopBuilderExtWindows::with_any_thread(
                &mut builder,
                true,
            );
            #[cfg(target_os = "linux")]
            {
                winit::platform::x11::EventLoopBuilderExtX11::with_any_thread(&mut builder, true);
                winit::platform::wayland::EventLoopBuilderExtWayland::with_any_thread(
                    &mut builder,
                    true,
                );
            }
            let events = builder
                .build()
                .map_err(|e| GPUError::LibraryError(e.to_string()))?;
            let native = Self {
                events: Rc::new(RefCell::new(events)),
                closed: Default::default(),
                pending: Default::default(),
            };
            *slot = Some(native.clone());
            Ok(native)
        })
    }

    /// Dispatch all windows once. Callers must route window events by WindowId.
    pub fn pump_events(
        &self,
        callback: impl FnMut(Event<()>, &ActiveEventLoop),
    ) -> Result<(), GPUError> {
        let mut events = self
            .events
            .try_borrow_mut()
            .map_err(|_| GPUError::LibraryError("recursive native event pump".into()))?;
        let status = events.pump_app_events(
            Some(Duration::ZERO),
            &mut Handler {
                callback,
                closed: &self.closed,
                pending: &self.pending,
            },
        );
        match status {
            PumpStatus::Continue => Ok(()),
            PumpStatus::Exit(code) => Err(GPUError::LibraryError(format!(
                "native event loop exited: {code}"
            ))),
        }
    }

    pub(crate) fn with_active<T>(
        &self,
        action: impl FnOnce(&ActiveEventLoop) -> T,
    ) -> Result<T, GPUError> {
        if ACTIVE_LOOP.is_set() {
            return Ok(ACTIVE_LOOP.with(action));
        }
        let mut action = Some(action);
        let mut result = None;
        let mut pending = Vec::new();
        let status = self.pump_events(|event, active| {
            if matches!(event, Event::Resumed | Event::AboutToWait) {
                if let Some(action) = action.take() {
                    result = Some(action(active));
                }
            }
            if matches!(event, Event::WindowEvent { .. } | Event::DeviceEvent { .. }) {
                pending.push(event);
            }
        });
        self.pending.borrow_mut().extend(pending);
        status?;
        result.ok_or_else(|| GPUError::LibraryError("native loop did not become active".into()))
    }

    pub fn create_custom_cursor(
        &self,
        source: CustomCursorSource,
    ) -> Result<CustomCursor, GPUError> {
        if ACTIVE_LOOP.is_set() {
            return Ok(ACTIVE_LOOP.with(|active| active.create_custom_cursor(source)));
        }
        let events = self
            .events
            .try_borrow()
            .map_err(|_| GPUError::LibraryError("native loop unavailable".into()))?;
        Ok(events.create_custom_cursor(source))
    }

    pub(crate) fn close_requested(&self, id: WindowId) -> bool {
        self.closed.borrow().contains(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "opens native windows; requires a desktop session"]
    fn windows_and_cursors_share_loop_across_recreation() {
        let events = NativeEventLoop::shared().unwrap();
        let first = events
            .create_window(Window::default_attributes().with_title("Dashi cursor 1"))
            .unwrap();
        first.request_redraw();
        let second = events
            .create_window(Window::default_attributes().with_title("Dashi cursor 2"))
            .unwrap();
        assert_ne!(first.id(), second.id());
        let source = CustomCursor::from_rgba(vec![255; 16 * 16 * 4], 16, 16, 3, 5).unwrap();
        first.set_cursor(events.create_custom_cursor(source).unwrap());
        let mut created_in_callback = false;
        let mut first_window_event = false;
        events
            .pump_events(|event, _| {
                if matches!(&event, Event::WindowEvent { window_id, .. } if *window_id == first.id()) {
                    first_window_event = true;
                }
                if matches!(event, Event::AboutToWait) {
                    let source =
                        CustomCursor::from_rgba(vec![255; 16 * 16 * 4], 16, 16, 3, 5).unwrap();
                    first.set_cursor(events.create_custom_cursor(source).unwrap());
                    created_in_callback = true;
                }
            })
            .unwrap();
        assert!(created_in_callback);
        assert!(
            first_window_event,
            "window creation must retain existing window events"
        );
        second.set_cursor(winit::window::CursorIcon::Crosshair);
        events.pump_events(|_, _| {}).unwrap();
        first.set_cursor_visible(false);
        first.set_cursor_visible(true);
        first.set_cursor(winit::window::CursorIcon::Default);
        drop(first);
        drop(second);
        events.pump_events(|_, _| {}).unwrap();
        let shared = NativeEventLoop::shared().unwrap();
        assert!(Rc::ptr_eq(&events.events, &shared.events));
        let recreated = shared.create_window(Window::default_attributes()).unwrap();
        shared.pump_events(|_, _| {}).unwrap();
        drop(recreated);
        shared.pump_events(|_, _| {}).unwrap();
        assert!(shared.closed.borrow().is_empty());
    }
}
