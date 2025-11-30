use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use ash::vk;
use dashi::gpu::vulkan::{Context, ContextInfo, GPUError};

unsafe extern "system" fn validation_error_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    _p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    user_data: *mut c_void,
) -> vk::Bool32 {
    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
        && message_type.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
    {
        if let Some(flag) = (!user_data.is_null()).then(|| &*(user_data as *const AtomicBool)) {
            flag.store(true, Ordering::SeqCst);
        }
    }

    vk::FALSE
}

pub struct ValidationContext {
    ctx: Option<Context>,
    guard: Option<ValidationGuard>,
}

impl ValidationContext {
    pub fn headless(info: &ContextInfo) -> Result<Self, GPUError> {
        let original_validation = std::env::var("DASHI_VALIDATION").ok();
        std::env::set_var("DASHI_VALIDATION", "1");

        let ctx = match Context::headless(info) {
            Ok(ctx) => ctx,
            Err(err) => {
                if let Some(value) = &original_validation {
                    std::env::set_var("DASHI_VALIDATION", value);
                } else {
                    std::env::remove_var("DASHI_VALIDATION");
                }
                return Err(err);
            }
        };

        let guard = ValidationGuard::new(&ctx, original_validation)?;

        Ok(Self {
            ctx: Some(ctx),
            guard: Some(guard),
        })
    }

    pub fn as_mut_ptr(&mut self) -> *mut Context {
        self.ctx
            .as_mut()
            .map(|ctx| ctx as *mut Context)
            .expect("context should be present")
    }
}

impl std::ops::Deref for ValidationContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        self.ctx.as_ref().expect("context should be present")
    }
}

impl std::ops::DerefMut for ValidationContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ctx.as_mut().expect("context should be present")
    }
}

impl Drop for ValidationContext {
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx.take() {
            if let Some(mut guard) = self.guard.take() {
                guard.teardown(&ctx);
            }

            ctx.destroy();
        }
    }
}

struct ValidationGuard {
    original_validation: Option<String>,
    validation_flag: Arc<AtomicBool>,
    validation_ptr: Option<*const AtomicBool>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl ValidationGuard {
    fn new(ctx: &Context, original_validation: Option<String>) -> Result<Self, GPUError> {
        let validation_flag = Arc::new(AtomicBool::new(false));
        let validation_ptr = Arc::into_raw(Arc::clone(&validation_flag));

        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
            .pfn_user_callback(Some(validation_error_callback))
            .user_data(validation_ptr as *mut c_void);

        let debug_messenger = ctx.create_debug_messenger(&messenger_info)?;

        Ok(Self {
            original_validation,
            validation_flag,
            validation_ptr: Some(validation_ptr),
            debug_messenger: Some(debug_messenger),
        })
    }

    fn teardown(&mut self, ctx: &Context) {
        if let Some(messenger) = self.debug_messenger.take() {
            ctx.destroy_debug_messenger(messenger);
        }

        if let Some(ptr) = self.validation_ptr.take() {
            unsafe {
                let _ = Arc::from_raw(ptr);
            }
        }

        if let Some(value) = &self.original_validation {
            std::env::set_var("DASHI_VALIDATION", value);
        } else {
            std::env::remove_var("DASHI_VALIDATION");
        }

        assert!(
            !self.validation_flag.load(Ordering::SeqCst),
            "Vulkan validation layers reported an API usage error"
        );
    }
}
