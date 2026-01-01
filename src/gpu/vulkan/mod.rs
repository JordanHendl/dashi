mod error;
use crate::{
    cmd::{CommandStream, Executable},
    driver::{
        command::{CopyBuffer, CopyBufferImage},
        state::StateTracker,
    },
    utils::{Handle, Pool},
    UsageBits,
};
use ash::vk::Handle as VkHandle;
use ash::*;
pub use error::*;
use std::{
    collections::{HashMap, HashSet},
    ffi::{c_char, c_void, CStr, CString},
    mem::ManuallyDrop,
};
use vk_mem::Alloc;

pub mod device_selector;
pub use crate::gpu::device_selector::*;
pub mod structs;
pub use structs::*;
pub mod builders;
pub mod commands;
pub mod timing;
pub use timing::*;
#[cfg(feature = "dashi-minifb")]
pub mod minifb_window;
#[cfg(feature = "dashi-openxr")]
pub mod openxr_window;
#[cfg(feature = "dashi-winit")]
pub mod winit_window;

mod conversions;
pub use conversions::*;

pub mod memory;
pub use memory::*;
pub mod image;
pub use image::*;
pub mod display;
pub use display::*;

mod descriptor_sets;
pub use descriptor_sets::*;

mod pipelines;
pub use pipelines::*;

mod command_pool;
pub use command_pool::CommandPool;

/// Names of debugging layers that should be enabled when validation is requested.
/// Only includes the standard Vulkan validation layer to avoid enabling any extra layers.
pub const DEBUG_LAYER_NAMES: [*const c_char; 1] =
    [b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const c_char];

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) };
    eprintln!(
        "[{:?}][{:?}] {}",
        message_severity,
        message_type,
        message.to_string_lossy()
    );
    vk::FALSE
}

#[derive(Debug)]
pub struct DebugMessenger {
    raw_handle: u64,
    callback_state: Box<DebugMessengerCallbackState>,
}

impl DebugMessenger {
    fn raw(&self) -> vk::DebugUtilsMessengerEXT {
        vk::DebugUtilsMessengerEXT::from_raw(self.raw_handle)
    }
}

#[derive(Clone, Debug)]
struct DebugMessengerCallbackState {
    callback: DebugMessengerCallback,
    user_data: *mut c_void,
}

fn vk_severity_flags(flags: DebugMessageSeverity) -> vk::DebugUtilsMessageSeverityFlagsEXT {
    let mut vk_flags = vk::DebugUtilsMessageSeverityFlagsEXT::empty();
    if flags.contains(DebugMessageSeverity::VERBOSE) {
        vk_flags |= vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE;
    }
    if flags.contains(DebugMessageSeverity::INFO) {
        vk_flags |= vk::DebugUtilsMessageSeverityFlagsEXT::INFO;
    }
    if flags.contains(DebugMessageSeverity::WARNING) {
        vk_flags |= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING;
    }
    if flags.contains(DebugMessageSeverity::ERROR) {
        vk_flags |= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
    }
    vk_flags
}

fn vk_message_type_flags(flags: DebugMessageType) -> vk::DebugUtilsMessageTypeFlagsEXT {
    let mut vk_flags = vk::DebugUtilsMessageTypeFlagsEXT::empty();
    if flags.contains(DebugMessageType::GENERAL) {
        vk_flags |= vk::DebugUtilsMessageTypeFlagsEXT::GENERAL;
    }
    if flags.contains(DebugMessageType::VALIDATION) {
        vk_flags |= vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;
    }
    if flags.contains(DebugMessageType::PERFORMANCE) {
        vk_flags |= vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE;
    }
    vk_flags
}

fn debug_severity_from_vk(flags: vk::DebugUtilsMessageSeverityFlagsEXT) -> DebugMessageSeverity {
    let mut severity = DebugMessageSeverity::empty();
    if flags.contains(vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE) {
        severity |= DebugMessageSeverity::VERBOSE;
    }
    if flags.contains(vk::DebugUtilsMessageSeverityFlagsEXT::INFO) {
        severity |= DebugMessageSeverity::INFO;
    }
    if flags.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        severity |= DebugMessageSeverity::WARNING;
    }
    if flags.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        severity |= DebugMessageSeverity::ERROR;
    }
    severity
}

fn debug_message_type_from_vk(flags: vk::DebugUtilsMessageTypeFlagsEXT) -> DebugMessageType {
    let mut message_type = DebugMessageType::empty();
    if flags.contains(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL) {
        message_type |= DebugMessageType::GENERAL;
    }
    if flags.contains(vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION) {
        message_type |= DebugMessageType::VALIDATION;
    }
    if flags.contains(vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE) {
        message_type |= DebugMessageType::PERFORMANCE;
    }
    message_type
}

unsafe extern "system" fn user_debug_trampoline(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    user_data: *mut c_void,
) -> vk::Bool32 {
    let state = &mut *(user_data as *mut DebugMessengerCallbackState);

    let message = if p_callback_data.is_null() || unsafe { (*p_callback_data).p_message.is_null() }
    {
        // SAFETY: static null-terminated string.
        unsafe { CStr::from_bytes_with_nul_unchecked(b"\0") }
    } else {
        unsafe { CStr::from_ptr((*p_callback_data).p_message) }
    };

    let handled = (state.callback)(
        debug_severity_from_vk(message_severity),
        debug_message_type_from_vk(message_type),
        message,
        state.user_data,
    );

    if handled {
        vk::TRUE
    } else {
        vk::FALSE
    }
}

#[derive(Clone, Default)]
pub struct RenderPass {
    pub(super) raw: vk::RenderPass,
    pub(super) fb: vk::Framebuffer,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) attachment_formats: Vec<Format>,
    pub(super) subpass_samples: Vec<SubpassSampleInfo>,
    pub(super) subpass_formats: Vec<SubpassAttachmentFormats>,
}

#[derive(Clone)]
pub struct Fence {
    raw: vk::Fence,
}

impl Default for Fence {
    fn default() -> Self {
        Self {
            raw: Default::default(),
        }
    }
}

impl Fence {
    fn new(raw: vk::Fence) -> Self {
        Self { raw }
    }
}

#[derive(Copy, Clone, Default)]
pub struct Semaphore {
    raw: vk::Semaphore,
}

#[derive(Clone)]
pub struct CommandQueue {
    cmd_buf: vk::CommandBuffer,
    fence: Handle<Fence>,
    dirty: bool,
    ctx: *mut VulkanContext,
    pool: *mut CommandPool,
    curr_rp: Option<Handle<RenderPass>>,
    curr_subpass: Option<usize>,
    curr_pipeline: Option<Handle<GraphicsPipeline>>,
    last_op_access: vk::AccessFlags,
    last_op_stage: vk::PipelineStageFlags,
    curr_attachments: Vec<(Handle<VkImageView>, vk::ImageLayout)>,
    queue_type: QueueType,
    is_secondary: bool,
}

impl Default for CommandQueue {
    fn default() -> Self {
        Self {
            cmd_buf: Default::default(),
            fence: Default::default(),
            dirty: false,
            ctx: std::ptr::null_mut(),
            pool: std::ptr::null_mut(),
            curr_rp: None,
            curr_subpass: None,
            curr_pipeline: None,
            last_op_access: vk::AccessFlags::TRANSFER_READ,
            last_op_stage: vk::PipelineStageFlags::ALL_COMMANDS,
            curr_attachments: Vec::new(),
            queue_type: QueueType::Graphics,
            is_secondary: false,
        }
    }
}

#[derive(Default)]
#[allow(dead_code)]
pub(super) struct Queue {
    queue: vk::Queue,
    family: u32,
}

pub struct VulkanContext {
    pub(super) entry: ash::Entry,
    pub(super) instance: ash::Instance,
    pub(super) pdevice: vk::PhysicalDevice,
    pub(super) device: ash::Device,
    pub(super) properties: ash::vk::PhysicalDeviceProperties,
    pub(super) descriptor_indexing_features: vk::PhysicalDeviceDescriptorIndexingFeatures,
    pub(super) gfx_pool: CommandPool,
    pub(super) compute_pool: Option<CommandPool>,
    pub(super) transfer_pool: Option<CommandPool>,
    pub(super) allocator: ManuallyDrop<vk_mem::Allocator>,
    pub(super) gfx_queue: Queue,
    pub(super) compute_queue: Option<Queue>,
    pub(super) transfer_queue: Option<Queue>,
    pub(super) buffers: Pool<Buffer>,
    pub(super) buffer_infos: Pool<BufferInfoRecord>,
    pub(super) render_passes: Pool<RenderPass>,
    pub(super) semaphores: Pool<Semaphore>,
    pub(super) fences: Pool<Fence>,
    pub(super) images: Pool<Image>,
    pub(super) image_infos: Pool<ImageInfoRecord>,
    pub(super) image_views: Pool<VkImageView>,
    pub(super) image_view_cache: HashMap<ImageView, Handle<VkImageView>>,
    pub(super) samplers: Pool<Sampler>,
    pub(super) bind_table_layouts: Pool<BindTableLayout>,
    pub(super) bind_tables: Pool<BindTable>,
    pub(super) gfx_pipeline_layouts: Pool<GraphicsPipelineLayout>,
    pub(super) gfx_pipelines: Pool<GraphicsPipeline>,
    pub(super) compute_pipeline_layouts: Pool<ComputePipelineLayout>,
    pub(super) compute_pipelines: Pool<ComputePipeline>,
    empty_set_layout: vk::DescriptorSetLayout,
    pub(super) resource_states: StateTracker,

    pub(super) gpu_timers: Vec<GpuTimer>,
    pub(super) timestamp_period: f32,

    /// Indicates whether the context was created in headless mode
    headless: bool,

    #[cfg(feature = "dashi-sdl2")]
    pub(super) sdl_context: sdl2::Sdl,
    #[cfg(feature = "dashi-sdl2")]
    pub(super) sdl_video: Option<sdl2::VideoSubsystem>,
    pub(super) debug_utils: Option<ash::extensions::ext::DebugUtils>,
    pub(super) debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl std::panic::UnwindSafe for VulkanContext {}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        if let (Some(utils), Some(messenger)) = (&self.debug_utils, self.debug_messenger) {
            unsafe {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
        }
    }
}

struct ResolvedBindTableBindings {
    storage: Vec<Vec<IndexedResource>>,
    bindings: Vec<(u32, usize)>,
}

impl ResolvedBindTableBindings {
    fn new(table_bindings: &[IndexedBindingInfo]) -> Self {
        let mut storage = Vec::with_capacity(table_bindings.len());
        let mut bindings = Vec::with_capacity(table_bindings.len());

        for binding in table_bindings {
            storage.push(binding.resources.to_vec());
            let idx = storage.len() - 1;
            bindings.push((binding.binding, idx));
        }

        Self { storage, bindings }
    }

    fn bindings(&self) -> Vec<IndexedBindingInfo<'_>> {
        self.bindings
            .iter()
            .map(|(binding, idx)| IndexedBindingInfo {
                binding: *binding,
                resources: &self.storage[*idx],
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::driver::command::{BeginDrawing, CommandSink, EndDrawing};
    use crate::gpu::driver::command::{CommandEncoder, Draw, GraphicsPipelineStateUpdate};
    use serial_test::serial;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    unsafe extern "system" fn validation_flag_callback(
        _message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        _p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
        user_data: *mut c_void,
    ) -> vk::Bool32 {
        if !user_data.is_null() {
            let flag: &AtomicBool = &*(user_data as *const AtomicBool);
            flag.store(true, Ordering::SeqCst);
        }

        vk::FALSE
    }
}

impl VulkanContext {
    fn init_core(
        info: &ContextInfo,
        windowed: bool,
        enable_validation: bool,
    ) -> Result<
        (
            ash::Entry,
            ash::Instance,
            vk::PhysicalDevice,
            ash::Device,
            ash::vk::PhysicalDeviceProperties,
            vk::PhysicalDeviceDescriptorIndexingFeatures,
            vk_mem::Allocator,
            Queue,
            Option<Queue>,
            Option<Queue>,
        ),
        GPUError,
    > {
        // === copy the first ~150 lines of Context::new up through
        //     command-pool + allocator creation + queue setup + debug_utils
        //
        //     then return all of those out.
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        // Create instance
        let entry = unsafe { Entry::load() }?;
        let mut inst_exts = Vec::new();
        if enable_validation {
            inst_exts.push(ash::extensions::ext::DebugUtils::name().as_ptr());
        }

        let mut inst_layers = Vec::new();
        if enable_validation {
            let available_layers = entry.enumerate_instance_layer_properties()?;
            for &layer in &DEBUG_LAYER_NAMES {
                let name = unsafe { CStr::from_ptr(layer) };
                if available_layers
                    .iter()
                    .any(|prop| unsafe { CStr::from_ptr(prop.layer_name.as_ptr()) == name })
                {
                    inst_layers.push(layer);
                }
            }
        }

        if windowed {
            // only pull in surface‐extensions when we actually need a window
            inst_exts.push(ash::extensions::khr::Surface::name().as_ptr());
            #[cfg(target_os = "linux")]
            inst_exts.push(ash::extensions::khr::XlibSurface::name().as_ptr());
            #[cfg(target_os = "windows")]
            inst_exts.push(ash::extensions::khr::Win32Surface::name().as_ptr());
        }

        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::builder()
                    .application_info(&app_info)
                    .enabled_extension_names(&inst_exts)
                    .enabled_layer_names(&inst_layers)
                    .build(),
                None,
            )
        }?;

        let pdevice = unsafe { instance.enumerate_physical_devices()?[info.device.device_id] };
        let device_prop = unsafe { instance.get_physical_device_properties(pdevice) };

        let queue_prop = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
        let _features = unsafe { instance.get_physical_device_features(pdevice) };

        let mut gfx_family = None;
        let mut compute_family = None;
        let mut transfer_family = None;

        for (idx, prop) in queue_prop.iter().enumerate() {
            if prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) && gfx_family.is_none() {
                gfx_family = Some(idx as u32);
            }
            if prop.queue_flags.contains(vk::QueueFlags::COMPUTE) && compute_family.is_none() {
                compute_family = Some(idx as u32);
            }
            if prop.queue_flags.contains(vk::QueueFlags::TRANSFER) && transfer_family.is_none() {
                transfer_family = Some(idx as u32);
            }
        }

        let gfx_family = gfx_family.unwrap();
        let compute_family = compute_family.unwrap_or(gfx_family);
        let transfer_family = transfer_family.unwrap_or(compute_family);

        let mut gfx_queue = Queue {
            family: gfx_family,
            ..Default::default()
        };
        let mut compute_queue = if compute_family != gfx_family {
            Some(Queue {
                family: compute_family,
                ..Default::default()
            })
        } else {
            None
        };
        let mut transfer_queue = if transfer_family != compute_family {
            Some(Queue {
                family: transfer_family,
                ..Default::default()
            })
        } else {
            None
        };

        let priorities = [1.0];
        let mut unique_families = vec![gfx_family];
        if compute_family != gfx_family {
            unique_families.push(compute_family);
        }
        if transfer_family != compute_family && transfer_family != gfx_family {
            unique_families.push(transfer_family);
        }
        let queue_infos: Vec<_> = unique_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(family)
                    .queue_priorities(&priorities)
                    .build()
            })
            .collect();

        let supports_vulkan11 = vk::api_version_major(device_prop.api_version) > 1
            || (vk::api_version_major(device_prop.api_version) == 1
                && vk::api_version_minor(device_prop.api_version) >= 1);

        let features = vk::PhysicalDeviceFeatures::builder()
            .shader_clip_distance(true)
            .build();

        let mut enabled_descriptor_indexing =
            vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        let mut features16bit = vk::PhysicalDevice16BitStorageFeatures::default();
        let mut features2 = supports_vulkan11.then(|| {
            let mut descriptor_indexing = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
            let mut feature_query = vk::PhysicalDeviceFeatures2::builder()
                .features(features)
                .push_next(&mut descriptor_indexing)
                .build();

            unsafe { instance.get_physical_device_features2(pdevice, &mut feature_query) };

            if descriptor_indexing.descriptor_binding_partially_bound == vk::TRUE {
                enabled_descriptor_indexing.descriptor_binding_partially_bound = vk::TRUE;
            }
            if descriptor_indexing.descriptor_binding_sampled_image_update_after_bind == vk::TRUE {
                enabled_descriptor_indexing.descriptor_binding_sampled_image_update_after_bind =
                    vk::TRUE;
            }
            if descriptor_indexing.descriptor_binding_uniform_buffer_update_after_bind == vk::TRUE {
                enabled_descriptor_indexing.descriptor_binding_uniform_buffer_update_after_bind =
                    vk::TRUE;
            }
            if descriptor_indexing.descriptor_binding_storage_buffer_update_after_bind == vk::TRUE {
                enabled_descriptor_indexing.descriptor_binding_storage_buffer_update_after_bind =
                    vk::TRUE;
            }
            if descriptor_indexing.descriptor_binding_storage_image_update_after_bind == vk::TRUE {
                enabled_descriptor_indexing.descriptor_binding_storage_image_update_after_bind =
                    vk::TRUE;
            }
            if descriptor_indexing.shader_sampled_image_array_non_uniform_indexing == vk::TRUE {
                enabled_descriptor_indexing.shader_sampled_image_array_non_uniform_indexing =
                    vk::TRUE;
            }
            if descriptor_indexing.shader_uniform_buffer_array_non_uniform_indexing == vk::TRUE {
                enabled_descriptor_indexing.shader_uniform_buffer_array_non_uniform_indexing =
                    vk::TRUE;
            }
            if descriptor_indexing.shader_storage_buffer_array_non_uniform_indexing == vk::TRUE {
                enabled_descriptor_indexing.shader_storage_buffer_array_non_uniform_indexing =
                    vk::TRUE;
            }

            let descriptor_features_enabled = enabled_descriptor_indexing
                .descriptor_binding_partially_bound
                == vk::TRUE
                || enabled_descriptor_indexing.descriptor_binding_sampled_image_update_after_bind
                    == vk::TRUE
                || enabled_descriptor_indexing.descriptor_binding_uniform_buffer_update_after_bind
                    == vk::TRUE
                || enabled_descriptor_indexing.descriptor_binding_storage_buffer_update_after_bind
                    == vk::TRUE
                || enabled_descriptor_indexing.descriptor_binding_storage_image_update_after_bind
                    == vk::TRUE
                || enabled_descriptor_indexing.shader_sampled_image_array_non_uniform_indexing
                    == vk::TRUE
                || enabled_descriptor_indexing.shader_uniform_buffer_array_non_uniform_indexing
                    == vk::TRUE
                || enabled_descriptor_indexing.shader_storage_buffer_array_non_uniform_indexing
                    == vk::TRUE;

            let mut f2 = vk::PhysicalDeviceFeatures2::builder().features(features);
            if descriptor_features_enabled {
                f2 = f2.push_next(&mut enabled_descriptor_indexing);
            }

            features16bit = vk::PhysicalDevice16BitStorageFeatures::builder()
                .uniform_and_storage_buffer16_bit_access(true)
                .build();

            f2.build()
        });

        let enabled_extensions =
            unsafe { instance.enumerate_device_extension_properties(pdevice) }?;

        let wanted_extensions: Vec<*const c_char> = if windowed {
            vec![
                ash::extensions::khr::Swapchain::name().as_ptr(),
                vk::KhrImagelessFramebufferFn::name().as_ptr(),
                unsafe {
                    std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_GOOGLE_user_type\0").as_ptr()
                },
                vk::GoogleHlslFunctionality1Fn::name().as_ptr(),
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                KhrPortabilitySubsetFn::name().as_ptr(),
            ]
        } else {
            vec![
                vk::KhrImagelessFramebufferFn::name().as_ptr(),
                unsafe {
                    std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_GOOGLE_user_type\0").as_ptr()
                },
                vk::GoogleHlslFunctionality1Fn::name().as_ptr(),
            ]
        };

        let extensions_to_enable: Vec<*const c_char> = wanted_extensions
            .into_iter()
            .filter(|a| {
                return enabled_extensions
                    .iter()
                    .find(|ext| {
                        let astr = unsafe { CStr::from_ptr(ext.extension_name.as_ptr()).to_str() }
                            .unwrap()
                            .to_string();
                        let bstr = unsafe { CStr::from_ptr(*(a)).to_str() }
                            .unwrap()
                            .to_string();

                        return astr == bstr;
                    })
                    .is_some();
            })
            .collect();

        let mut imageless = vk::PhysicalDeviceImagelessFramebufferFeatures::default();
        imageless.imageless_framebuffer = vk::TRUE;

        let mut device_ci = vk::DeviceCreateInfo::builder()
            .enabled_extension_names(&extensions_to_enable)
            .queue_create_infos(&queue_infos);

        if let Some(ref mut f2) = features2 {
            device_ci = device_ci.push_next(f2);
            if supports_vulkan11 {
                device_ci = device_ci.push_next(&mut features16bit);
                device_ci = device_ci.push_next(&mut imageless);
            }
        } else {
            device_ci = device_ci.enabled_features(&features);
        }

        let device = unsafe { instance.create_device(pdevice, &device_ci.build(), None) }?;

        gfx_queue.queue = unsafe { device.get_device_queue(gfx_family, 0) };
        if let Some(ref mut q) = compute_queue {
            q.queue = unsafe { device.get_device_queue(compute_family, 0) };
        }
        if let Some(ref mut q) = transfer_queue {
            q.queue = unsafe { device.get_device_queue(transfer_family, 0) };
        }

        let allocator = vk_mem::Allocator::new(vk_mem::AllocatorCreateInfo::new(
            &instance, &device, pdevice,
        ))?;

        return Ok((
            entry,
            instance,
            pdevice,
            device,
            device_prop,
            enabled_descriptor_indexing,
            allocator,
            gfx_queue,
            compute_queue,
            transfer_queue,
        ));
    }

    /// Construct a [`Context`] without any windowing support.
    ///
    /// This variant skips surface creation and is intended for off-screen
    /// rendering or compute-only workloads. For general usage requirements
    /// (device selection, fence lifetimes, and queue limitations) see
    /// [`Self::new`].
    pub fn headless(info: &ContextInfo) -> Result<Self> {
        #[cfg(feature = "webgpu")]
        if info.web_surface.is_some() {
            return Err(GPUError::SwapchainConfigError(
                "Web surface data is only supported by the WebGPU backend",
            ));
        }
        let enable_validation = std::env::var("DASHI_VALIDATION")
            .map(|v| v == "1")
            .unwrap_or(false);
        let (
            entry,
            instance,
            pdevice,
            device,
            properties,
            descriptor_indexing_features,
            allocator,
            gfx_queue,
            compute_queue,
            transfer_queue,
        ) = Self::init_core(info, false, enable_validation)?;

        let (debug_utils, debug_messenger) = if enable_validation {
            let debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);
            let messenger_ci = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));
            let debug_messenger = unsafe {
                debug_utils
                    .create_debug_utils_messenger(&messenger_ci, None)
                    .unwrap()
            };
            (Some(debug_utils), Some(debug_messenger))
        } else {
            (None, None)
        };

        let gfx_pool = CommandPool::new(&device, gfx_queue.family, QueueType::Graphics)?;
        let compute_pool = if compute_queue.is_some() {
            Some(CommandPool::new(
                &device,
                compute_queue.as_ref().unwrap().family,
                QueueType::Compute,
            )?)
        } else {
            None
        };
        let transfer_pool = if transfer_queue.is_some() {
            Some(CommandPool::new(
                &device,
                transfer_queue.as_ref().unwrap().family,
                QueueType::Transfer,
            )?)
        } else {
            None
        };

        let supports_update_after_bind_layouts = descriptor_indexing_features
            .descriptor_binding_sampled_image_update_after_bind
            == vk::TRUE
            || descriptor_indexing_features.descriptor_binding_uniform_buffer_update_after_bind
                == vk::TRUE
            || descriptor_indexing_features.descriptor_binding_storage_buffer_update_after_bind
                == vk::TRUE
            || descriptor_indexing_features.descriptor_binding_storage_image_update_after_bind
                == vk::TRUE;

        let empty_set_layout =
            Self::create_empty_set_layout(&device, supports_update_after_bind_layouts)?;

        let mut ctx = VulkanContext {
            entry,
            instance,
            pdevice,
            device,
            properties,
            descriptor_indexing_features,
            gfx_pool,
            compute_pool,
            transfer_pool,
            allocator: ManuallyDrop::new(allocator),
            gfx_queue,
            compute_queue,
            transfer_queue,

            buffers: Default::default(),
            buffer_infos: Default::default(),
            render_passes: Default::default(),
            semaphores: Default::default(),
            fences: Default::default(),
            images: Default::default(),
            image_infos: Default::default(),
            image_views: Default::default(),
            image_view_cache: HashMap::new(),
            samplers: Default::default(),
            bind_table_layouts: Default::default(),
            bind_tables: Default::default(),
            gfx_pipeline_layouts: Default::default(),
            gfx_pipelines: Default::default(),
            compute_pipeline_layouts: Default::default(),
            compute_pipelines: Default::default(),
            empty_set_layout,
            resource_states: StateTracker::new(),
            gpu_timers: Vec::new(),
            timestamp_period: properties.limits.timestamp_period,
            headless: true,

            #[cfg(feature = "dashi-sdl2")]
            sdl_context: sdl2::init().unwrap(),
            #[cfg(feature = "dashi-sdl2")]
            sdl_video: None,
            debug_utils,
            debug_messenger,
        };
        ctx.init_gpu_timers(1)?;
        Ok(ctx)
    }

    /// Construct a [`Context`] with windowing support.
    ///
    /// # Requirements
    /// - A physical device must be selected using a `DeviceSelector`.
    /// - The window backend used here must match the backend supported by
    ///   the selected device.
    /// - Fences returned from [`Self::submit`] must remain valid until
    ///   [`Self::wait`] or [`Self::release_list_on_next_submit`] is called.
    /// - Vulkan queues require command lists to be finished before submission;
    ///   [`Self::submit`] will end a still-recording list automatically.
    pub fn new(info: &ContextInfo) -> Result<Self> {
        #[cfg(feature = "webgpu")]
        if info.web_surface.is_some() {
            return Err(GPUError::SwapchainConfigError(
                "Web surface data is only supported by the WebGPU backend",
            ));
        }
        let enable_validation = std::env::var("DASHI_VALIDATION")
            .map(|v| v == "1")
            .unwrap_or(false);
        let (
            entry,
            instance,
            pdevice,
            device,
            properties,
            descriptor_indexing_features,
            allocator,
            gfx_queue,
            compute_queue,
            transfer_queue,
        ) = Self::init_core(info, true, enable_validation)?;

        let (debug_utils, debug_messenger) = if enable_validation {
            let debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);
            let messenger_ci = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(vulkan_debug_callback));
            let debug_messenger = unsafe {
                debug_utils
                    .create_debug_utils_messenger(&messenger_ci, None)
                    .unwrap()
            };
            (Some(debug_utils), Some(debug_messenger))
        } else {
            (None, None)
        };

        #[cfg(feature = "dashi-sdl2")]
        let sdl_context = sdl2::init().unwrap();
        #[cfg(feature = "dashi-sdl2")]
        let sdl_video = sdl_context.video().unwrap();

        let gfx_pool = CommandPool::new(&device, gfx_queue.family, QueueType::Graphics)?;
        let compute_pool = if compute_queue.is_some() {
            Some(CommandPool::new(
                &device,
                compute_queue.as_ref().unwrap().family,
                QueueType::Compute,
            )?)
        } else {
            None
        };
        let transfer_pool = if transfer_queue.is_some() {
            Some(CommandPool::new(
                &device,
                transfer_queue.as_ref().unwrap().family,
                QueueType::Transfer,
            )?)
        } else {
            None
        };

        let supports_update_after_bind_layouts = descriptor_indexing_features
            .descriptor_binding_sampled_image_update_after_bind
            == vk::TRUE
            || descriptor_indexing_features.descriptor_binding_uniform_buffer_update_after_bind
                == vk::TRUE
            || descriptor_indexing_features.descriptor_binding_storage_buffer_update_after_bind
                == vk::TRUE
            || descriptor_indexing_features.descriptor_binding_storage_image_update_after_bind
                == vk::TRUE;

        let empty_set_layout =
            Self::create_empty_set_layout(&device, supports_update_after_bind_layouts)?;

        let mut ctx = VulkanContext {
            entry,
            instance,
            pdevice,
            device,
            properties,
            descriptor_indexing_features,
            gfx_pool,
            compute_pool,
            transfer_pool,
            allocator: ManuallyDrop::new(allocator),
            gfx_queue,
            compute_queue,
            transfer_queue,

            buffers: Default::default(),
            buffer_infos: Default::default(),
            render_passes: Default::default(),
            semaphores: Default::default(),
            fences: Default::default(),
            images: Default::default(),
            image_infos: Default::default(),
            image_views: Default::default(),
            image_view_cache: HashMap::new(),
            samplers: Default::default(),
            bind_table_layouts: Default::default(),
            bind_tables: Default::default(),
            gfx_pipeline_layouts: Default::default(),
            gfx_pipelines: Default::default(),
            compute_pipeline_layouts: Default::default(),
            compute_pipelines: Default::default(),
            empty_set_layout,
            resource_states: StateTracker::new(),
            gpu_timers: Vec::new(),
            timestamp_period: properties.limits.timestamp_period,
            headless: false,

            #[cfg(feature = "dashi-sdl2")]
            sdl_context,
            #[cfg(feature = "dashi-sdl2")]
            sdl_video: Some(sdl_video),
            debug_utils,
            debug_messenger,
        };
        ctx.init_gpu_timers(1)?;
        Ok(ctx)
    }

    /// Query hardware limits in API-agnostic terms.
    ///
    /// This information can be used to size bind tables, buffers, or push
    /// constant blocks so that shaders never exceed what the device supports.
    pub fn limits(&self) -> ContextLimits {
        let limits = &self.properties.limits;

        ContextLimits {
            max_sampler_array_len: limits.max_per_stage_descriptor_samplers,
            max_sampled_texture_array_len: limits.max_per_stage_descriptor_sampled_images,
            max_storage_texture_array_len: limits.max_per_stage_descriptor_storage_images,
            max_uniform_buffer_range: limits.max_uniform_buffer_range,
            max_storage_buffer_range: limits.max_storage_buffer_range,
            max_push_constant_size: limits.max_push_constants_size,
            max_color_attachments: limits.max_color_attachments,
            max_bound_bind_tables: limits.max_bound_descriptor_sets,
        }
    }

    #[cfg(feature = "dashi-sdl2")]
    /// Access the underlying SDL context for window creation.
    ///
    /// Only available when the `dashi-sdl2` feature is enabled. Contexts
    /// created with [`Self::headless`] do not initialize SDL.
    pub fn get_sdl_ctx(&mut self) -> &mut sdl2::Sdl {
        return &mut self.sdl_context;
    }

    fn set_name<T>(&self, obj: T, name: &str, t: vk::ObjectType)
    where
        T: ash::vk::Handle,
    {
        if let Some(utils) = &self.debug_utils {
            unsafe {
                let name: CString = CString::new(name.to_string()).unwrap();
                utils
                    .set_debug_utils_object_name(
                        self.device.handle(),
                        &vk::DebugUtilsObjectNameInfoEXT::builder()
                            .object_name(&name)
                            .object_type(t)
                            .object_handle(ash::vk::Handle::as_raw(obj))
                            .build(),
                    )
                    .expect("Error writing debug name");
            }
        }
    }

    /// Block until `fence` signals and reset it for reuse.
    ///
    /// The fence must have been obtained from [`Self::submit`]. This call
    /// ensures the associated GPU work is complete before the fence is
    /// reused or the command list is destroyed.
    pub fn wait(&mut self, fence: Handle<Fence>) -> Result<()> {
        let fence = self.fences.get_ref(fence).unwrap();
        let _res = unsafe {
            self.device
                .wait_for_fences(&[fence.raw], true, std::u64::MAX)
        }?;

        unsafe { self.device.reset_fences(&[fence.raw]) }?;

        Ok(())
    }

    /// Retrieve a queue handle for the requested type, applying fallback
    /// semantics if a dedicated queue was not created.
    fn queue(&self, ty: QueueType) -> vk::Queue {
        match ty {
            QueueType::Graphics => self.gfx_queue.queue,
            QueueType::Compute => self.compute_queue.as_ref().unwrap_or(&self.gfx_queue).queue,
            QueueType::Transfer => {
                self.transfer_queue
                    .as_ref()
                    .or(self.compute_queue.as_ref())
                    .unwrap_or(&self.gfx_queue)
                    .queue
            }
        }
    }

    pub fn record(
        &mut self,
        stream: CommandStream<Executable>,
        info: &CommandQueueInfo,
    ) -> Result<CommandQueue> {
        todo!()
    }

    /// Create a new command pool for the specified queue type.
    pub fn make_command_pool(&self, ty: QueueType) -> Result<CommandPool> {
        let family = match ty {
            QueueType::Graphics => self.gfx_queue.family,
            QueueType::Compute => {
                self.compute_queue
                    .as_ref()
                    .unwrap_or(&self.gfx_queue)
                    .family
            }
            QueueType::Transfer => {
                self.transfer_queue
                    .as_ref()
                    .or(self.compute_queue.as_ref())
                    .unwrap_or(&self.gfx_queue)
                    .family
            }
        };
        CommandPool::new(&self.device, family, ty)
    }

    /// Retrieve a mutable reference to a queue's command pool.
    pub fn pool_mut(&mut self, ty: QueueType) -> &mut CommandPool {
        let ctx_ptr = self as *mut VulkanContext;
        let pool = match ty {
            QueueType::Graphics => &mut self.gfx_pool,
            QueueType::Compute => self.compute_pool.as_mut().unwrap_or(&mut self.gfx_pool),
            QueueType::Transfer => self
                .transfer_pool
                .as_mut()
                .unwrap_or(self.compute_pool.as_mut().unwrap_or(&mut self.gfx_pool)),
        };
        pool.bind_context(ctx_ptr);
        pool
    }

    /// Retrieve an immutable reference to a queue's command pool.
    pub fn pool(&self, ty: QueueType) -> &CommandPool {
        let ctx_ptr = self as *const VulkanContext as *mut VulkanContext;
        let pool = match ty {
            QueueType::Graphics => &self.gfx_pool,
            QueueType::Compute => self.compute_pool.as_ref().unwrap_or(&self.gfx_pool),
            QueueType::Transfer => self
                .transfer_pool
                .as_ref()
                .unwrap_or(self.compute_pool.as_ref().unwrap_or(&self.gfx_pool)),
        };
        pool.bind_context(ctx_ptr);
        pool
    }

    fn oneshot_transition_image(&mut self, img: ImageView, layout: vk::ImageLayout) {
        let view_handle = self.get_or_create_image_view(&img).unwrap();
        let ctx_ptr = self as *mut _;
        self.gfx_pool.bind_context(ctx_ptr);
        let mut list = self.gfx_pool.begin("", false).unwrap();
        self.transition_image(list.cmd_buf, view_handle, layout);
        let fence = self.submit(&mut list, &Default::default()).unwrap();
        self.wait(fence).unwrap();
        self.destroy_cmd_queue(list);
    }

    fn oneshot_transition_image_noview(&mut self, img: Handle<Image>, layout: vk::ImageLayout) {
        let tmp_view = ImageView {
            img,
            range: Default::default(),
            aspect: Default::default(),
        };
        let view_handle = self.get_or_create_image_view(&tmp_view).unwrap();

        let ctx_ptr = self as *mut _;
        self.gfx_pool.bind_context(ctx_ptr);
        let mut list = self.gfx_pool.begin("oneshot_transition", false).unwrap();
        self.transition_image(list.cmd_buf, view_handle, layout);
        let fence = self.submit(&mut list, &Default::default()).unwrap();
        self.wait(fence).unwrap();
        self.destroy_cmd_queue(list);
    }

    fn transition_image_stages(
        &mut self,
        cmd: vk::CommandBuffer,
        img: Handle<VkImageView>,
        layout: vk::ImageLayout,
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
    ) {
        let view = self.image_views.get_mut_ref(img).unwrap();
        let img = self.images.get_mut_ref(view.img).unwrap();
        let base = view.range.base_mip_level as usize;
        let count = view.range.level_count as usize;
        let old_layout = img.layouts[base];
        let new_layout = if layout == vk::ImageLayout::UNDEFINED {
            vk::ImageLayout::GENERAL
        } else {
            layout
        };
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                src,
                dst,
                vk::DependencyFlags::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier::builder()
                    .new_layout(new_layout)
                    .old_layout(old_layout)
                    .image(img.img)
                    .src_access_mask(src_access)
                    .dst_access_mask(dst_access)
                    .subresource_range(view.range)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .build()],
            )
        };

        for i in base..base + count {
            if let Some(l) = img.layouts.get_mut(i) {
                *l = layout;
            }
        }
    }

    /// Utility: given an image’s old & new layouts, pick the correct
    /// src/dst pipeline stages and access masks.
    fn barrier_masks_for_transition(
        &self,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) -> (
        vk::PipelineStageFlags,
        vk::AccessFlags,
        vk::PipelineStageFlags,
        vk::AccessFlags,
    ) {
        use vk::{AccessFlags as AF, ImageLayout as L, PipelineStageFlags as PS};

        match (old_layout, new_layout) {
            // UNDEFINED → TRANSFER_DST (first use as a copy/blit target)
            (L::UNDEFINED, L::TRANSFER_DST_OPTIMAL) => (
                PS::TOP_OF_PIPE,
                AF::empty(),
                PS::TRANSFER,
                AF::TRANSFER_WRITE,
            ),

            // COLOR_ATTACHMENT_OPTIMAL → TRANSFER_SRC (reading from a color buffer)
            (L::COLOR_ATTACHMENT_OPTIMAL, L::TRANSFER_SRC_OPTIMAL) => (
                PS::COLOR_ATTACHMENT_OUTPUT,
                AF::COLOR_ATTACHMENT_WRITE,
                PS::TRANSFER,
                AF::TRANSFER_READ,
            ),

            // TRANSFER_DST → SHADER_READ (e.g. sample from an image you just wrote)
            (L::TRANSFER_DST_OPTIMAL, L::SHADER_READ_ONLY_OPTIMAL) => (
                PS::TRANSFER,
                AF::TRANSFER_WRITE,
                PS::FRAGMENT_SHADER,
                AF::SHADER_READ,
            ),

            // SHADER_READ → COLOR_ATTACHMENT (rendering over a previously sampled image)
            (L::SHADER_READ_ONLY_OPTIMAL, L::COLOR_ATTACHMENT_OPTIMAL) => (
                PS::FRAGMENT_SHADER,
                AF::SHADER_READ,
                PS::COLOR_ATTACHMENT_OUTPUT,
                AF::COLOR_ATTACHMENT_WRITE,
            ),

            // TRANSFER_SRC → GENERAL (reset to general for arbitrary use)
            (L::TRANSFER_SRC_OPTIMAL, L::GENERAL) | (L::TRANSFER_DST_OPTIMAL, L::GENERAL) => (
                PS::TRANSFER,
                AF::TRANSFER_READ | AF::TRANSFER_WRITE,
                PS::ALL_COMMANDS,
                AF::empty(),
            ),

            // Any other combination → worst-case “all commands”
            _ => (PS::ALL_COMMANDS, AF::empty(), PS::ALL_COMMANDS, AF::empty()),
        }
    }

    fn transition_image(
        &mut self,
        cmd: vk::CommandBuffer,
        img: Handle<VkImageView>,
        layout: vk::ImageLayout,
    ) {
        let (src_stage, src_access, dst_stage, dst_access) = {
            let view = self.image_views.get_ref(img).unwrap();
            let image = self.images.get_ref(view.img).unwrap();
            let base = view.range.base_mip_level as usize;
            let old_layout = image.layouts[base];
            let new_layout = if layout == vk::ImageLayout::UNDEFINED {
                vk::ImageLayout::GENERAL
            } else {
                layout
            };
            self.barrier_masks_for_transition(old_layout, new_layout)
        };

        self.transition_image_stages(
            cmd, img, layout, src_stage, dst_stage, src_access, dst_access,
        );
    }

    /// Finish recording and submit a command list to the graphics queue.
    ///
    /// Returns a fence handle that can be waited on to know when the GPU has
    /// completed the work.
    ///
    /// # Requirements
    /// - The command list must be finished before submission. If it is still
    ///   recording, this function calls `vkEndCommandBuffer` automatically.
    /// - The returned fence must remain valid until the work has completed and
    ///   [`Self::wait`] or [`Self::release_list_on_next_submit`] is called.
    pub fn submit(
        &mut self,
        cmd: &mut CommandQueue,
        info: &SubmitInfo,
    ) -> Result<Handle<Fence>, GPUError> {
        if cmd.dirty {
            unsafe { self.device.end_command_buffer(cmd.cmd_buf)? };
            cmd.dirty = false;
        }

        let raw_wait_sems: Vec<vk::Semaphore> = info
            .wait_sems
            .into_iter()
            .map(|a| self.semaphores.get_ref(a.clone()).unwrap().raw)
            .collect();

        let raw_signal_sems: Vec<vk::Semaphore> = info
            .signal_sems
            .into_iter()
            .map(|a| self.semaphores.get_ref(a.clone()).unwrap().raw)
            .collect();

        let stage_masks = vec![vk::PipelineStageFlags::ALL_COMMANDS; raw_wait_sems.len()];
        let queue = self.queue(cmd.queue_type);
        unsafe {
            self.device.queue_submit(
                queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[cmd.cmd_buf])
                    .signal_semaphores(&raw_signal_sems)
                    .wait_dst_stage_mask(&stage_masks)
                    .wait_semaphores(&raw_wait_sems)
                    .build()],
                self.fences.get_ref(cmd.fence).unwrap().raw.clone(),
            )?
        };

        return Ok(cmd.fence.clone());
    }

    pub fn make_semaphores(&mut self, count: usize) -> Result<Vec<Handle<Semaphore>>, GPUError> {
        let mut f = Vec::with_capacity(count);
        for _i in 0..count {
            f.push(self.make_semaphore()?);
        }

        Ok(f)
    }

    pub fn make_semaphore(&mut self) -> Result<Handle<Semaphore>, GPUError> {
        Ok(self
            .semaphores
            .insert(Semaphore {
                raw: unsafe {
                    self.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::builder().build(), None)
                }?,
            })
            .unwrap())
    }

    pub fn make_fence(&mut self) -> Result<Handle<Fence>, GPUError> {
        let f = unsafe {
            self.device.create_fence(
                &vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build(),
                None,
            )
        }?;

        return Ok(self.fences.insert(Fence::new(f)).unwrap());
    }

    pub fn make_sampler(&mut self, info: &SamplerInfo) -> Result<Handle<Sampler>, GPUError> {
        let new_info: vk::SamplerCreateInfo = (*info).into();
        let sampler = unsafe { self.device.create_sampler(&new_info, None) }?;

        if let Some(h) = self.samplers.insert(Sampler { sampler }) {
            return Ok(h);
        } else {
            return Err(GPUError::SlotError());
        }
    }

    pub(crate) fn get_or_create_image_view(
        &mut self,
        info: &ImageView,
    ) -> Result<Handle<VkImageView>, GPUError> {
        if let Some(h) = self.image_view_cache.get(info) {
            return Ok(*h);
        }

        let img = self.images.get_ref(info.img).unwrap();
        let img_info = self
            .image_infos
            .get_ref(img.info_handle)
            .ok_or(GPUError::SlotError())?;
        let aspect: vk::ImageAspectFlags = info.aspect.into();
        let sub_range = vk::ImageSubresourceRange::builder()
            .base_array_layer(info.range.base_layer)
            .layer_count(info.range.layer_count)
            .base_mip_level(info.range.base_mip)
            .level_count(info.range.level_count)
            .aspect_mask(aspect)
            .build();

        let view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(img.img)
                    .format(lib_to_vk_image_format(&img_info.info.format))
                    .subresource_range(sub_range)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .build(),
                Default::default(),
            )?
        };

        let handle = match self.image_views.insert(VkImageView {
            view,
            range: sub_range,
            img: info.img,
        }) {
            Some(h) => h,
            None => return Err(GPUError::SlotError()),
        };

        self.image_view_cache.insert(*info, handle);
        Ok(handle)
    }

    pub fn make_image(&mut self, info: &ImageInfo) -> Result<Handle<Image>, GPUError> {
        let mut base_usage_flags = vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::SAMPLED;

        if info.format == Format::D24S8 {
            base_usage_flags = base_usage_flags | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        } else {
            base_usage_flags = base_usage_flags | vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }

        let (image, allocation) = unsafe {
            self.allocator.create_image(
                &vk::ImageCreateInfo::builder()
                    .extent(vk::Extent3D {
                        width: info.dim[0] as u32,
                        height: info.dim[1] as u32,
                        depth: 1,
                    })
                    .array_layers(info.layers)
                    .format(lib_to_vk_image_format(&info.format))
                    .mip_levels(info.mip_levels as u32)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(base_usage_flags)
                    .image_type(vk::ImageType::TYPE_2D)
                    .samples(info.samples.into())
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .flags(vk::ImageCreateFlags::empty())
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .queue_family_indices(&[0])
                    .build(),
                &vk_mem::AllocationCreateInfo {
                    usage: vk_mem::MemoryUsage::Auto,
                    ..Default::default()
                },
            )
        }?;

        self.set_name(image, info.debug_name, vk::ObjectType::IMAGE);

        let info_handle = self
            .image_infos
            .insert(ImageInfoRecord::new(info))
            .ok_or(GPUError::SlotError())?;

        match self.images.insert(Image {
            img: image,
            alloc: allocation,
            layouts: vec![vk::ImageLayout::UNDEFINED; info.mip_levels as usize],
            info_handle,
        }) {
            Some(h) => {
                self.init_image(h, &info)?;
                return Ok(h);
            }
            None => {
                self.image_infos.release(info_handle);
                Err(GPUError::SlotError())
            }
        }
    }

    pub fn image_info(&self, image: Handle<Image>) -> &ImageInfo<'static> {
        let img = self.images.get_ref(image).expect("Invalid image handle");
        let info = self
            .image_infos
            .get_ref(img.info_handle)
            .expect("Invalid image info handle");
        &info.info
    }

    pub fn flush_buffer(&mut self, buffer: BufferView) -> Result<()> {
        let buf = match self.buffers.get_ref(buffer.handle) {
            Some(it) => it,
            None => return Err(GPUError::SlotError()),
        };

        let buffer_size = buf.size as u64;

        let info = self.allocator.get_allocation_info(&buf.alloc);
        self.allocator
            .flush_allocation(&buf.alloc, info.offset as usize, info.size as usize)?;
        Ok(())
    }

    pub fn map_buffer_mut<T>(&mut self, view: BufferView) -> Result<&mut [T]> {
        let buf = match self.buffers.get_ref(view.handle) {
            Some(it) => it,
            None => return Err(GPUError::SlotError()),
        };

        let mut alloc: vk_mem::Allocation = unsafe { std::mem::transmute_copy(&buf.alloc) };
        let mapped = unsafe { self.allocator.map_memory(&mut alloc) }?;
        let mut typed_map: *mut T = unsafe { std::mem::transmute(mapped) };
        let base_offset = buf.offset as u64 + view.offset as u64;
        typed_map = unsafe { typed_map.add(base_offset as usize) };
        let buffer_size = buf.size as u64;
        let available = buffer_size.saturating_sub(view.offset.min(buffer_size));
        let size = if view.size == 0 {
            buffer_size
        } else {
            view.size.min(available)
        };
        return Ok(unsafe {
            std::slice::from_raw_parts_mut(typed_map, size as usize / std::mem::size_of::<T>())
        });
    }

    pub fn map_buffer<T>(&mut self, view: BufferView) -> Result<&[T]> {
        let buf = match self.buffers.get_ref(view.handle) {
            Some(it) => it,
            None => return Err(GPUError::SlotError()),
        };

        let mut alloc: vk_mem::Allocation = unsafe { std::mem::transmute_copy(&buf.alloc) };
        let mapped = unsafe { self.allocator.map_memory(&mut alloc) }?;
        let mut typed_map: *mut T = unsafe { std::mem::transmute(mapped) };
        let base_offset = buf.offset as u64 + view.offset;
        typed_map = unsafe { typed_map.add(base_offset as usize) };
        let buffer_size = buf.size as u64;
        let available = buffer_size.saturating_sub(view.offset.min(buffer_size));
        let size = if view.size == 0 {
            buffer_size
        } else {
            view.size.min(available)
        };
        return Ok(unsafe {
            std::slice::from_raw_parts(typed_map, size as usize / std::mem::size_of::<T>())
        });
    }

    pub fn unmap_buffer(&self, buf: Handle<Buffer>) -> Result<()> {
        let buf = match self.buffers.get_ref(buf) {
            Some(it) => it,
            None => return Err(GPUError::SlotError()),
        };

        let mut alloc: vk_mem::Allocation = unsafe { std::mem::transmute_copy(&buf.alloc) };
        unsafe { self.allocator.unmap_memory(&mut alloc) };

        return Ok(());
    }
    fn init_buffer(&mut self, buf: Handle<Buffer>, info: &BufferInfo) -> Result<()> {
        if info.initial_data.is_none() {
            return Ok(());
        }

        match info.visibility {
            MemoryVisibility::Gpu => {
                let mut new_info = info.clone();
                new_info.visibility = MemoryVisibility::CpuAndGpu;

                let staging = self.make_buffer(&new_info)?;

                let ctx_ptr = self as *mut _;
                self.gfx_pool.bind_context(ctx_ptr);
                let mut list = self.gfx_pool.begin("", false)?;
                let stream = CommandStream::new().begin().copy_buffers(&CopyBuffer {
                    src: staging,
                    dst: buf,
                    src_offset: 0,
                    dst_offset: 0,
                    amount: unsafe { info.initial_data.unwrap_unchecked().len() } as u32,
                });

                stream.end().append(&mut list)?;

                let fence = self.submit(&mut list, &Default::default())?;
                self.wait(fence.clone())?;
                self.destroy_cmd_queue(list);
                self.destroy_buffer(staging);
            }
            MemoryVisibility::CpuAndGpu => {
                let mapped: &mut [u8] = self.map_buffer_mut(BufferView::new(buf))?;
                mapped.copy_from_slice(unsafe { info.initial_data.unwrap_unchecked() });
                self.unmap_buffer(buf)?;
            }
        }

        return Ok(());
    }

    fn init_image(&mut self, image: Handle<Image>, info: &ImageInfo) -> Result<()> {
        if info.initial_data.is_none() {
            return Ok(());
        }

        let staging = self.make_buffer(&BufferInfo {
            debug_name: "",
            byte_size: (info.dim[0]
                * info.dim[1]
                * info.dim[2]
                * channel_count(&info.format)
                * bytes_per_channel(&info.format)) as u32,
            visibility: MemoryVisibility::CpuAndGpu,
            initial_data: info.initial_data,
            ..Default::default()
        })?;

        let ctx_ptr = self as *mut _;
        self.gfx_pool.bind_context(ctx_ptr);
        let mut list = self.gfx_pool.begin("", false)?;
        let mut cmd = CommandStream::new()
            .begin()
            .copy_buffer_to_image(&CopyBufferImage {
                src: staging,
                dst: image,
                range: SubresourceRange {
                    base_mip: 0,
                    level_count: 1,
                    base_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            });

        if info.mip_levels > 1 {
            for i in 0..info.mip_levels - 1 {
                cmd = cmd.copy_buffer_to_image(&CopyBufferImage {
                    src: staging,
                    dst: image,
                    range: SubresourceRange {
                        base_mip: i,
                        level_count: 1,
                        base_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                });
            }
        }

        cmd.end().append(&mut list)?;
        let fence = self.submit(&mut list, &Default::default())?;
        self.wait(fence)?;
        self.destroy_cmd_queue(list);
        self.destroy_buffer(staging);
        Ok(())
    }

    /// Block until the current device finishes all queued work.
    pub fn sync_current_device(&mut self) {
        unsafe { self.device.device_wait_idle().unwrap() };
    }

    /// Allocate `count` GPU timers.
    ///
    /// This must be called before any of the GPU timer helpers are used.
    /// Existing timers are destroyed and replaced with the new set.
    pub fn init_gpu_timers(&mut self, count: usize) -> Result<()> {
        for timer in self.gpu_timers.drain(..) {
            unsafe { timer.destroy(&self.device) };
        }
        let mut timers = Vec::with_capacity(count);
        for _ in 0..count {
            timers.push(GpuTimer::new(&self.device)?);
        }
        self.gpu_timers = timers;
        Ok(())
    }

    /// Begin timing for `frame` on the provided command list.
    ///
    /// [`init_gpu_timers`] must be called beforehand, and the matching
    /// [`gpu_timer_end`] must be invoked on the **same** `CommandQueue`.
    pub fn gpu_timer_begin(&mut self, list: &mut CommandQueue, frame: usize) {
        if let Some(t) = self.gpu_timers.get(frame) {
            unsafe { t.begin(&self.device, list.cmd_buf) };
        }
    }

    /// End timing for `frame` on the provided command list.
    ///
    /// Must pair with a preceding [`gpu_timer_begin`] call on the same list.
    pub fn gpu_timer_end(&mut self, list: &mut CommandQueue, frame: usize) {
        if let Some(t) = self.gpu_timers.get(frame) {
            unsafe { t.end(&self.device, list.cmd_buf) };
        }
    }

    /// Return the elapsed GPU time for `frame` in milliseconds.
    ///
    /// Only valid once the associated command list has been submitted and
    /// the GPU has finished executing it (e.g. by waiting on the fence).
    pub fn get_elapsed_gpu_time_ms(&mut self, frame: usize) -> Option<f32> {
        self.gpu_timers
            .get(frame)
            .and_then(|t| t.resolve(&self.device, self.timestamp_period).ok())
    }

    pub fn make_dynamic_allocator(
        &mut self,
        info: &DynamicAllocatorInfo,
    ) -> Result<DynamicAllocator> {
        let buffer = self.make_buffer(&BufferInfo {
            debug_name: info.debug_name,
            byte_size: info.byte_size,
            usage: info.usage,
            visibility: MemoryVisibility::CpuAndGpu,
            ..Default::default()
        })?;

        let min_alloc_size = info.allocation_size
            + (info.allocation_size
                % self.properties.limits.min_uniform_buffer_offset_alignment as u32);
        return Ok(DynamicAllocator {
            allocator: offset_allocator::Allocator::new(info.byte_size, info.num_allocations),
            pool: buffer,
            ptr: self.map_buffer_mut(BufferView::new(buffer))?.as_mut_ptr(),
            min_alloc_size,
        });
    }

    pub fn suballoc_from(
        &mut self,
        parent: Handle<Buffer>,
        offset: u32,
        size: u32,
    ) -> Option<Handle<Buffer>> {
        let src = self.buffers.get_ref(parent).unwrap();
        let mut cpy = src.clone();

        if src.size - cpy.offset + offset < size {
            return None;
        }

        cpy.size = size;
        cpy.offset += offset;
        cpy.suballocated = true;
        let parent_info = self.buffer_info(parent);
        let info_handle = match self.buffer_infos.insert(BufferInfoRecord::new(&BufferInfo {
            debug_name: parent_info.debug_name,
            byte_size: size,
            visibility: parent_info.visibility,
            usage: parent_info.usage,
            initial_data: None,
        })) {
            Some(handle) => handle,
            None => return None,
        };
        cpy.info_handle = info_handle;
        match self.buffers.insert(cpy) {
            Some(handle) => {
                return Some(handle);
            }
            None => {
                self.buffer_infos.release(info_handle);
                return None;
            }
        }
    }

    pub fn make_buffer(&mut self, info: &BufferInfo) -> Result<Handle<Buffer>, GPUError> {
        let usage = vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::UNIFORM_BUFFER;

        let mappable = matches!(info.visibility, MemoryVisibility::CpuAndGpu);
        let create_info = vk_mem::AllocationCreateInfo {
            usage: if mappable {
                vk_mem::MemoryUsage::AutoPreferHost
            } else {
                vk_mem::MemoryUsage::Auto
            },
            flags: if mappable {
                vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM
            } else {
                vk_mem::AllocationCreateFlags::empty()
            },
            ..Default::default()
        };

        unsafe {
            let (buffer, allocation) = self.allocator.create_buffer(
                &ash::vk::BufferCreateInfo::builder()
                    .size(info.byte_size as u64)
                    .usage(usage)
                    .build(),
                &create_info,
            )?;

            self.set_name(buffer, info.debug_name, vk::ObjectType::BUFFER);
            let info_handle = self
                .buffer_infos
                .insert(BufferInfoRecord::new(info))
                .ok_or(GPUError::SlotError())?;
            match self.buffers.insert(Buffer {
                buf: buffer,
                alloc: allocation,
                size: info.byte_size,
                offset: 0,
                suballocated: false,
                info_handle,
            }) {
                Some(handle) => {
                    self.init_buffer(handle, info)?;
                    return Ok(handle);
                }
                None => {
                    self.buffer_infos.release(info_handle);
                    return Err(GPUError::SlotError());
                }
            }
        }
    }

    pub fn buffer_info(&self, buffer: Handle<Buffer>) -> &BufferInfo<'static> {
        let buf = self.buffers.get_ref(buffer).expect("Invalid buffer handle");
        let info = self
            .buffer_infos
            .get_ref(buf.info_handle)
            .expect("Invalid buffer info handle");
        &info.info
    }

    /// Completes any outstanding resource cleanup on the context.
    ///
    /// # Prerequisites
    /// - The context must still be alive.
    /// - Ensure the GPU has finished using any resources scheduled for cleanup.
    ///
    /// Currently this method is a no-op and exists for API completeness.
    pub fn clean_up(&mut self) {}

    /// Creates an additional debug messenger using the context's debug utils interface.
    ///
    /// Validation layers must be enabled via `DASHI_VALIDATION=1` for this to succeed.
    pub fn create_debug_messenger(
        &self,
        info: &DebugMessengerCreateInfo,
    ) -> Result<DebugMessenger> {
        let debug_utils = self.debug_utils.as_ref().ok_or(GPUError::Unimplemented(
            "Debug utils unavailable; enable validation to install callbacks",
        ))?;

        let mut callback_state = Box::new(DebugMessengerCallbackState {
            callback: info.user_callback,
            user_data: info.user_data,
        });

        let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vk_severity_flags(info.message_severity))
            .message_type(vk_message_type_flags(info.message_type))
            .pfn_user_callback(Some(user_debug_trampoline))
            .user_data(callback_state.as_mut() as *mut _ as *mut c_void);

        let messenger = unsafe {
            debug_utils
                .create_debug_utils_messenger(&messenger_info, None)
                .map_err(GPUError::from)?
        };

        Ok(DebugMessenger {
            raw_handle: messenger.as_raw(),
            callback_state,
        })
    }

    /// Destroys a debug messenger created via [`create_debug_messenger`].
    pub fn destroy_debug_messenger(&self, messenger: DebugMessenger) {
        if let Some(utils) = &self.debug_utils {
            unsafe { utils.destroy_debug_utils_messenger(messenger.raw(), None) };
        }
    }

    /// Destroys the GPU context and all resources owned by it.
    ///
    /// # Prerequisites
    /// - Ensure the GPU has completed all work referencing resources created by this
    ///   context.
    /// - Destroy dependent objects (for example, image views before their images)
    ///   prior to calling this method if they are still externally referenced.
    ///
    /// After calling this method the context and all of its resources become invalid.
    pub fn destroy(mut self) {
        if let Some(messenger) = self.debug_messenger.take() {
            if let Some(utils) = &self.debug_utils {
                unsafe { utils.destroy_debug_utils_messenger(messenger, None) };
            }
        }

        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.empty_set_layout, None);
        }

        // Bind table layouts
        self.bind_table_layouts
            .for_each_occupied_mut(|layout| unsafe {
                self.device
                    .destroy_descriptor_set_layout(layout.layout, None);
                self.device.destroy_descriptor_pool(layout.pool, None);
            });

        // Semaphores
        self.semaphores.for_each_occupied_mut(|s| {
            unsafe { self.device.destroy_semaphore(s.raw, None) };
        });

        // Fences
        self.fences.for_each_occupied_mut(|f| {
            unsafe { self.device.destroy_fence(f.raw, None) };
        });

        // Samplers
        self.samplers.for_each_occupied_mut(|s| {
            unsafe { self.device.destroy_sampler(s.sampler, None) };
        });

        // Image views
        self.image_views.for_each_occupied_mut(|view| {
            unsafe { self.device.destroy_image_view(view.view, None) };
        });
        self.image_view_cache.clear();

        // Images
        self.images.for_each_occupied_mut(|img| {
            unsafe { self.allocator.destroy_image(img.img, &mut img.alloc) };
        });
        self.image_infos.clear();

        // Buffers
        self.buffers.for_each_occupied_mut(|buf| {
            if !buf.suballocated {
                unsafe { self.allocator.destroy_buffer(buf.buf, &mut buf.alloc) };
            }
        });
        self.buffer_infos.clear();

        // Render passes
        self.render_passes.for_each_occupied_mut(|rp| {
            unsafe { self.device.destroy_render_pass(rp.raw, None) };
            unsafe { self.device.destroy_framebuffer(rp.fb, None) };
        });

        // Graphics pipeline layouts
        self.gfx_pipeline_layouts.for_each_occupied_mut(|layout| {
            for stage in &layout.shader_stages {
                unsafe {
                    self.device.destroy_shader_module(stage.module, None);
                }
            }
            unsafe { self.device.destroy_pipeline_layout(layout.layout, None) };
        });

        // Graphics pipelines
        self.gfx_pipelines.for_each_occupied_mut(|pipeline| unsafe {
            self.device.destroy_pipeline(pipeline.raw, None);
        });

        // Compute pipeline layouts
        self.compute_pipeline_layouts
            .for_each_occupied_mut(|layout| unsafe {
                self.device
                    .destroy_shader_module(layout.shader_stage.module, None);
                self.device.destroy_pipeline_layout(layout.layout, None);
            });

        // Compute pipelines
        self.compute_pipelines
            .for_each_occupied_mut(|pipeline| unsafe {
                self.device.destroy_pipeline(pipeline.raw, None);
            });

        for timer in self.gpu_timers.drain(..) {
            unsafe { timer.destroy(&self.device) };
        }

        // Command pools
        self.gfx_pool.destroy();
        if let Some(pool) = self.compute_pool.as_mut() {
            pool.destroy();
        }
        if let Some(pool) = self.transfer_pool.as_mut() {
            pool.destroy();
        }

        // Destroy allocator before tearing down device and instance
        unsafe {
            ManuallyDrop::drop(&mut self.allocator);
        }

        // Device and instance
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }

    /// Destroys a [`DynamicAllocator`] and its backing buffer.
    ///
    /// # Prerequisites
    /// - All allocations made from the allocator must no longer be in use by the GPU.
    /// - The allocator's buffer must not be mapped.
    /// - The context must still be alive.
    pub fn destroy_dynamic_allocator(&mut self, alloc: DynamicAllocator) {
        self.unmap_buffer(alloc.pool).unwrap();
        self.destroy_buffer(alloc.pool);
    }

    /// Destroys a buffer and frees its memory.
    ///
    /// # Prerequisites
    /// - Ensure all GPU work using the buffer has completed.
    /// - The context must still be alive.
    pub fn destroy_buffer(&mut self, handle: Handle<Buffer>) {
        let buf = self.buffers.get_mut_ref(handle).unwrap();
        if !buf.suballocated {
            unsafe { self.allocator.destroy_buffer(buf.buf, &mut buf.alloc) };
        }
        self.buffer_infos.release(buf.info_handle);
        self.buffers.release(handle);
    }

    /// Destroys a semaphore.
    ///
    /// # Prerequisites
    /// - The semaphore must not be in use by the GPU.
    /// - The context must still be alive.
    pub fn destroy_semaphore(&mut self, handle: Handle<Semaphore>) {
        let sem = self.semaphores.get_mut_ref(handle).unwrap();
        unsafe { self.device.destroy_semaphore(sem.raw, None) };
        self.semaphores.release(handle);
    }

    /// Destroys a fence.
    ///
    /// # Prerequisites
    /// - Wait for the fence to be signaled before destroying it.
    /// - The context must still be alive.
    pub fn destroy_fence(&mut self, handle: Handle<Fence>) {
        let fence = self.fences.get_mut_ref(handle).unwrap();
        unsafe { self.device.destroy_fence(fence.raw, None) };
        self.fences.release(handle);
    }

    /// Destroys an image view.
    ///
    /// # Prerequisites
    /// - Ensure no GPU work references the image view.
    /// - Destroy views before destroying the underlying image.
    /// - The context must still be alive.
    pub fn destroy_image_view(&mut self, handle: Handle<VkImageView>) {
        if let Some(img) = self.image_views.get_mut_ref(handle) {
            unsafe { self.device.destroy_image_view(img.view, None) };
        }
        self.image_view_cache.retain(|_, v| *v != handle);
        self.image_views.release(handle);
    }

    /// Destroys an image and frees its memory.
    ///
    /// # Prerequisites
    /// - All views created from the image must be destroyed first.
    /// - Ensure the GPU has finished using the image.
    /// - The context must still be alive.
    pub fn destroy_image(&mut self, handle: Handle<Image>) {
        // Destroy any cached views associated with this image
        let to_destroy: Vec<Handle<VkImageView>> = self
            .image_view_cache
            .iter()
            .filter_map(|(info, view_handle)| {
                if info.img == handle {
                    Some(*view_handle)
                } else {
                    None
                }
            })
            .collect();
        for view_handle in &to_destroy {
            if let Some(view) = self.image_views.get_ref(*view_handle) {
                unsafe { self.device.destroy_image_view(view.view, None) };
            }
            self.image_views.release(*view_handle);
        }
        self.image_view_cache
            .retain(|_, view_handle| !to_destroy.contains(view_handle));

        let img = self.images.get_mut_ref(handle).unwrap();
        unsafe { self.allocator.destroy_image(img.img, &mut img.alloc) };
        self.image_infos.release(img.info_handle);
        self.images.release(handle);
    }

    /// Destroys a render pass.
    ///
    /// # Prerequisites
    /// - Ensure no GPU work is using the render pass.
    /// - The context must still be alive.
    pub fn destroy_render_pass(&mut self, handle: Handle<RenderPass>) {
        let rp = self.render_passes.get_ref(handle).unwrap();
        unsafe {
            self.device.destroy_framebuffer(rp.fb, None);
            self.device.destroy_render_pass(rp.raw, None);
        }
        self.render_passes.release(handle);
    }

    /// Destroys a render target and its framebuffer.
    ///
    /// Destroys a command list and its associated fence.
    ///
    /// # Prerequisites
    /// - Wait for the command list's fence to signal, ensuring the GPU has finished
    ///   executing the list.
    /// - The context must still be alive.
    pub fn destroy_cmd_queue(&mut self, list: CommandQueue) {
        let fence = unsafe { (*list.pool).recycle(list) };
        self.destroy_fence(fence);
    }

    /// Creates a bind table layout used for bind table resources.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn make_bind_table_layout(
        &mut self,
        info: &BindTableLayoutInfo,
    ) -> Result<Handle<BindTableLayout>, GPUError> {
        const MAX_DESCRIPTOR_SETS: u32 = 2048;

        let supports_partially_bound = self
            .descriptor_indexing_features
            .descriptor_binding_partially_bound
            == vk::TRUE;
        let supports_uab = self
            .descriptor_indexing_features
            .descriptor_binding_uniform_buffer_update_after_bind
            == vk::TRUE;
        let supports_sampled_uab = self
            .descriptor_indexing_features
            .descriptor_binding_sampled_image_update_after_bind
            == vk::TRUE;
        let supports_storage_uab = self
            .descriptor_indexing_features
            .descriptor_binding_storage_buffer_update_after_bind
            == vk::TRUE;
        let supports_storage_image_uab = self
            .descriptor_indexing_features
            .descriptor_binding_storage_image_update_after_bind
            == vk::TRUE;

        let mut flags = Vec::new();
        let mut bindings = Vec::new();
        let mut uses_update_after_bind = false;
        for shader_info in info.shaders.iter() {
            for variable in shader_info.variables.iter() {
                let descriptor_type = match variable.var_type {
                    BindTableVariableType::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
                    BindTableVariableType::DynamicUniform => {
                        vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    }
                    BindTableVariableType::Storage => vk::DescriptorType::STORAGE_BUFFER,
                    BindTableVariableType::SampledImage => {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    }
                    BindTableVariableType::Image => vk::DescriptorType::SAMPLED_IMAGE,
                    BindTableVariableType::Sampler => vk::DescriptorType::SAMPLER,
                    BindTableVariableType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
                    BindTableVariableType::DynamicStorage => {
                        vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                    }
                };

                let stage_flags = match shader_info.shader_type {
                    ShaderType::Vertex => vk::ShaderStageFlags::VERTEX,
                    ShaderType::Fragment => vk::ShaderStageFlags::FRAGMENT,
                    ShaderType::Compute => vk::ShaderStageFlags::COMPUTE,
                    ShaderType::All => vk::ShaderStageFlags::ALL,
                };

                let mut binding_flags = vk::DescriptorBindingFlags::empty();
                if supports_partially_bound {
                    binding_flags |= vk::DescriptorBindingFlags::PARTIALLY_BOUND;
                }
                let binding_supports_uab = match variable.var_type {
                    BindTableVariableType::Uniform | BindTableVariableType::DynamicUniform => {
                        supports_uab
                    }
                    BindTableVariableType::DynamicStorage | BindTableVariableType::Storage => {
                        supports_storage_uab
                    }
                    BindTableVariableType::SampledImage | BindTableVariableType::Image => {
                        supports_sampled_uab
                    }
                    BindTableVariableType::Sampler => false,
                    BindTableVariableType::StorageImage => supports_storage_image_uab,
                };
                let binding_supports_uab = false && binding_supports_uab;
                if binding_supports_uab {
                    binding_flags |= vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
                    uses_update_after_bind = true;
                }

                flags.push(binding_flags);
                let layout_binding = vk::DescriptorSetLayoutBinding::builder()
                    .binding(variable.binding)
                    .descriptor_type(descriptor_type)
                    .descriptor_count(variable.count) // Assuming one per binding
                    .stage_flags(stage_flags)
                    .build();

                bindings.push(layout_binding);
            }
        }
        let mut layout_binding_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&flags);

        let mut layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
        if uses_update_after_bind {
            layout_info =
                layout_info.flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL);
        }

        let mut layout_binding_info = layout_binding_info.build();
        if flags.iter().any(|f| !f.is_empty()) {
            layout_info = layout_info.push_next(&mut layout_binding_info);
        }

        let layout_info = layout_info.build();

        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)?
        };

        let pool_sizes = bindings
            .iter()
            .map(|binding| {
                vk::DescriptorPoolSize::builder()
                    .ty(binding.descriptor_type)
                    .descriptor_count(binding.descriptor_count)
                    .build()
            })
            .collect::<Vec<_>>();

        let mut pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(MAX_DESCRIPTOR_SETS);
        if uses_update_after_bind {
            pool_info = pool_info.flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
        }

        let descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None)? };

        self.set_name(
            descriptor_set_layout,
            info.debug_name,
            vk::ObjectType::DESCRIPTOR_SET_LAYOUT,
        );
        self.set_name(
            descriptor_pool,
            info.debug_name,
            vk::ObjectType::DESCRIPTOR_POOL,
        );

        // Step 4: Return the BindTableLayout
        return Ok(self
            .bind_table_layouts
            .insert(BindTableLayout {
                pool: descriptor_pool,
                layout: descriptor_set_layout,
                variables: info
                    .shaders
                    .iter()
                    .flat_map(|shader| shader.variables.iter().cloned())
                    .collect(),
                update_after_bind: uses_update_after_bind,
                partially_bound: supports_partially_bound,
            })
            .unwrap());
    }

    pub fn table_layouts_compatible(
        &self,
        bg1: Handle<BindTableLayout>,
        bg2: Handle<BindTableLayout>,
    ) -> bool {
        let bg1 = self.bind_table_layouts.get_ref(bg1).unwrap();
        let bg2 = self.bind_table_layouts.get_ref(bg2).unwrap();
        bg1.variables == bg2.variables
    }

    pub fn bind_table_compatible(
        &self,
        layout: Handle<BindTableLayout>,
        bindings: &[IndexedBindingInfo],
    ) -> bool {
        let layout = self.bind_table_layouts.get_ref(layout).unwrap();
        indexed_bindings_compatible_with_layout(&layout.variables, bindings)
    }

    pub fn bind_table_info_compatible(&self, info: &BindTableInfo) -> bool {
        let resolved = self.resolve_bind_table_bindings(info.bindings);

        let bindings = resolved.bindings();

        self.bind_table_compatible(info.layout, &bindings)
    }

    pub fn bind_table_layout_flags(
        &self,
        layout: Handle<BindTableLayout>,
    ) -> DescriptorBindingFlagsInfo {
        let layout = self.bind_table_layouts.get_ref(layout).unwrap();
        DescriptorBindingFlagsInfo {
            update_after_bind: layout.update_after_bind,
            partially_bound: layout.partially_bound,
        }
    }

    fn usage_bits_for_variable(var_type: BindTableVariableType) -> UsageBits {
        match var_type {
            BindTableVariableType::Uniform | BindTableVariableType::DynamicUniform => {
                UsageBits::UNIFORM_READ
            }
            BindTableVariableType::Storage | BindTableVariableType::DynamicStorage => {
                UsageBits::STORAGE_READ | UsageBits::STORAGE_WRITE
            }
            _ => UsageBits::empty(),
        }
    }

    fn buffer_usage_from_resource(
        resource: &ShaderResource,
        var_type: BindTableVariableType,
    ) -> Option<(Handle<Buffer>, UsageBits)> {
        let usage = Self::usage_bits_for_variable(var_type);
        match resource {
            ShaderResource::Buffer(view)
            | ShaderResource::ConstBuffer(view)
            | ShaderResource::StorageBuffer(view) => Some((view.handle, usage)),
            ShaderResource::Dynamic(alloc) | ShaderResource::DynamicStorage(alloc) => {
                Some((alloc.pool, usage))
            }
            _ => None,
        }
    }

    fn resolve_bind_table_bindings(
        &self,
        table_bindings: &[IndexedBindingInfo],
    ) -> ResolvedBindTableBindings {
        ResolvedBindTableBindings::new(table_bindings)
    }

    /// Updates an existing bind table with new resource bindings.
    pub fn update_bind_table(&mut self, info: &BindTableUpdateInfo) -> Result<()> {
        self.write_bind_table_bindings(info.table, info.bindings)
    }

    fn write_bind_table_bindings(
        &mut self,
        table_handle: Handle<BindTable>,
        bindings: &[IndexedBindingInfo],
    ) -> Result<()> {
        let (descriptor_set, layout_handle) = {
            let table = self.bind_tables.get_ref(table_handle).unwrap();
            (table.set, table.layout)
        };
        let layout = self.bind_table_layouts.get_ref(layout_handle).unwrap();
        let Some(requirements) = layout_binding_requirements(&layout.variables) else {
            return Err(GPUError::InvalidBindTableBinding {
                binding: u32::MAX,
                reason: "bind table layout has conflicting bindings".to_string(),
            });
        };
        let mut tracked_states = Vec::new();
        let mut write_descriptor_sets = Vec::with_capacity(9064);
        let mut buffer_infos = Vec::with_capacity(9064);
        let mut image_infos = Vec::with_capacity(9064);
        let mut seen_slots: HashMap<u32, HashSet<u32>> = HashMap::new();

        for binding_info in bindings.iter() {
            let Some((expected_type, expected_count)) = requirements.get(&binding_info.binding)
            else {
                return Err(GPUError::InvalidBindTableBinding {
                    binding: binding_info.binding,
                    reason: "binding is not part of the bind table layout".to_string(),
                });
            };

            let slots = seen_slots.entry(binding_info.binding).or_default();
            for res in binding_info.resources {
                let actual_type = resource_var_type(&res.resource);
                if &actual_type != expected_type {
                    return Err(GPUError::InvalidBindTableBinding {
                        binding: binding_info.binding,
                        reason: format!(
                            "resource type {:?} does not match expected table variable {:?}",
                            actual_type, expected_type
                        ),
                    });
                }

                if res.slot >= *expected_count {
                    return Err(GPUError::InvalidBindTableBinding {
                        binding: binding_info.binding,
                        reason: format!(
                            "slot {} exceeds declared count {}",
                            res.slot, expected_count
                        ),
                    });
                }

                if !slots.insert(res.slot) {
                    return Err(GPUError::InvalidBindTableBinding {
                        binding: binding_info.binding,
                        reason: format!("slot {} written more than once", res.slot),
                    });
                }

                match &res.resource {
                    ShaderResource::Buffer(view) => {
                        if let Some((buffer, usage)) =
                            Self::buffer_usage_from_resource(&res.resource, *expected_type)
                        {
                            tracked_states.push((buffer, usage));
                        }
                        let buffer = self.buffers.get_ref(view.handle).unwrap();
                        let buffer_size = buffer.size as u64;
                        let available = buffer_size.saturating_sub(view.offset.min(buffer_size));
                        let size = if view.size == 0 {
                            available
                        } else {
                            view.size.min(available)
                        };
                        let buffer_info = vk::DescriptorBufferInfo::builder()
                            .buffer(buffer.buf)
                            .offset(buffer.offset as u64 + view.offset)
                            .range(size)
                            .build();

                        buffer_infos.push(buffer_info);

                        let write_descriptor_set = vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set)
                            .dst_binding(binding_info.binding)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(&buffer_infos[buffer_infos.len() - 1..])
                            .dst_array_element(res.slot)
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::SampledImage(image_view, sampler) => {
                        let handle = self.get_or_create_image_view(image_view)?;
                        let image = self.image_views.get_ref(handle).unwrap();
                        let sampler = self.samplers.get_ref(*sampler).unwrap();

                        let image_info = vk::DescriptorImageInfo::builder()
                            .image_view(image.view)
                            .image_layout(vk::ImageLayout::GENERAL)
                            .sampler(sampler.sampler)
                            .build();

                        image_infos.push(image_info);

                        let write_descriptor_set = vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set)
                            .dst_binding(binding_info.binding)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .dst_array_element(res.slot)
                            .image_info(&image_infos[image_infos.len() - 1..])
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::Image(image_view) => {
                        let handle = self.get_or_create_image_view(image_view)?;
                        let image = self.image_views.get_ref(handle).unwrap();

                        let image_info = vk::DescriptorImageInfo::builder()
                            .image_view(image.view)
                            .image_layout(vk::ImageLayout::GENERAL)
                            .build();

                        image_infos.push(image_info);

                        let write_descriptor_set = vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set)
                            .dst_binding(binding_info.binding)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .dst_array_element(res.slot)
                            .image_info(&image_infos[image_infos.len() - 1..])
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::Sampler(sampler_handle) => {
                        let sampler = self.samplers.get_ref(*sampler_handle).unwrap();

                        let image_info = vk::DescriptorImageInfo::builder()
                            .sampler(sampler.sampler)
                            .build();

                        image_infos.push(image_info);

                        let write_descriptor_set = vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set)
                            .dst_binding(binding_info.binding)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .dst_array_element(res.slot)
                            .image_info(&image_infos[image_infos.len() - 1..])
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::Dynamic(alloc) => {
                        if let Some((buffer, usage)) =
                            Self::buffer_usage_from_resource(&res.resource, *expected_type)
                        {
                            tracked_states.push((buffer, usage));
                        }
                        let buffer = self.buffers.get_ref(alloc.pool).unwrap();

                        let buffer_info = vk::DescriptorBufferInfo::builder()
                            .buffer(buffer.buf)
                            .offset(0)
                            .range(alloc.min_alloc_size as u64)
                            .build();

                        buffer_infos.push(buffer_info);

                        let write_descriptor_set = vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set)
                            .dst_binding(binding_info.binding)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
                            .buffer_info(&buffer_infos[buffer_infos.len() - 1..])
                            .dst_array_element(res.slot)
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::DynamicStorage(alloc) => {
                        if let Some((buffer, usage)) =
                            Self::buffer_usage_from_resource(&res.resource, *expected_type)
                        {
                            tracked_states.push((buffer, usage));
                        }
                        let buffer = self.buffers.get_ref(alloc.pool).unwrap();
                        let buffer_info = vk::DescriptorBufferInfo::builder()
                            .buffer(buffer.buf)
                            .offset(0)
                            .range(alloc.min_alloc_size as u64)
                            .build();

                        buffer_infos.push(buffer_info);

                        let write_descriptor_set = vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set)
                            .dst_binding(binding_info.binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC)
                            .dst_array_element(res.slot)
                            .buffer_info(&buffer_infos[buffer_infos.len() - 1..])
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::StorageBuffer(view) => {
                        if let Some((buffer, usage)) =
                            Self::buffer_usage_from_resource(&res.resource, *expected_type)
                        {
                            tracked_states.push((buffer, usage));
                        }
                        let buffer = self.buffers.get_ref(view.handle).unwrap();
                        let buffer_size = buffer.size as u64;
                        let available = buffer_size.saturating_sub(view.offset.min(buffer_size));
                        let size = if view.size == 0 {
                            available
                        } else {
                            view.size.min(available)
                        };

                        let buffer_info = vk::DescriptorBufferInfo::builder()
                            .buffer(buffer.buf)
                            .offset(buffer.offset as u64 + view.offset)
                            .range(size)
                            .build();

                        buffer_infos.push(buffer_info);

                        let write_descriptor_set = vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set)
                            .dst_binding(binding_info.binding)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .dst_array_element(res.slot)
                            .buffer_info(&buffer_infos[buffer_infos.len() - 1..])
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::ConstBuffer(view) => {
                        if let Some((buffer, usage)) =
                            Self::buffer_usage_from_resource(&res.resource, *expected_type)
                        {
                            tracked_states.push((buffer, usage));
                        }
                        let buffer = self.buffers.get_ref(view.handle).unwrap();

                        let size = if view.size == 0 {
                            buffer.size as u64
                        } else {
                            view.size
                        };
                        let buffer_info = vk::DescriptorBufferInfo::builder()
                            .buffer(buffer.buf)
                            .offset(view.offset as u64)
                            .range(size as u64)
                            .build();

                        buffer_infos.push(buffer_info);
                        let write_descriptor_set = vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set)
                            .dst_binding(binding_info.binding)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER) // Assuming a uniform buffer for now
                            .buffer_info(&buffer_infos[buffer_infos.len() - 1..])
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                }
            }
        }

        unsafe {
            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        let table = self.bind_tables.get_mut_ref(table_handle).unwrap();
        table.buffer_states = tracked_states;

        Ok(())
    }

    /// Creates a bind table and initializes it with the provided bindings.
    pub fn make_bind_table(&mut self, info: &BindTableInfo) -> Result<Handle<BindTable>, GPUError> {
        let layout = self.bind_table_layouts.get_ref(info.layout).unwrap();

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(layout.pool)
            .set_layouts(&[layout.layout])
            .build();

        let descriptor_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? };
        let descriptor_set = descriptor_sets[0];

        self.set_name(
            descriptor_set,
            info.debug_name,
            vk::ObjectType::DESCRIPTOR_SET,
        );

        let bind_table = BindTable {
            set: descriptor_set,
            set_id: info.set,
            layout: info.layout,
            buffer_states: Vec::new(),
        };

        let table = self.bind_tables.insert(bind_table).unwrap();
        let resolved = self.resolve_bind_table_bindings(info.bindings);

        let bindings = resolved.bindings();

        self.write_bind_table_bindings(table, &bindings)?;

        Ok(table)
    }

    fn create_pipeline_layout(
        &self,
        bind_table_layout_handle: &[Option<Handle<BindTableLayout>>],
    ) -> Result<vk::PipelineLayout> {
        let max_index = bind_table_layout_handle
            .iter()
            .enumerate()
            .filter_map(|(idx, h)| h.map(|_| idx))
            .max();

        let mut layouts = Vec::new();
        if let Some(max_index) = max_index {
            layouts.reserve(max_index + 1);
            for index in 0..=max_index {
                match bind_table_layout_handle.get(index) {
                    Some(Some(t)) => {
                        layouts.push(self.bind_table_layouts.get_ref(*t).unwrap().layout)
                    }
                    _ => layouts.push(self.empty_set_layout),
                }
            }
        }

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&layouts)
            .push_constant_ranges(&[]) // Add push constant ranges if needed
            .build();

        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None)? };

        Ok(pipeline_layout)
    }

    fn create_empty_set_layout(
        device: &ash::Device,
        supports_update_after_bind: bool,
    ) -> Result<vk::DescriptorSetLayout> {
        let mut layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[]);
        if supports_update_after_bind {
            layout_info =
                layout_info.flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL);
        }
        let layout_info = layout_info.build();

        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None)? };
        Ok(layout)
    }

    fn create_shader_module(&self, spirv_code: &[u32]) -> Result<vk::ShaderModule> {
        // Step 1: Create Shader Module Info
        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(spirv_code) // The SPIR-V bytecode
            .build();

        // Step 2: Create Shader Module using Vulkan
        let shader_module = unsafe { self.device.create_shader_module(&create_info, None)? };

        // Step 3: Return the shader module
        Ok(shader_module)
    }

    #[cfg(feature = "dashi-serde")]
    pub fn make_render_pass_from_yaml(&mut self, yaml_str: &str) -> Result<RenderPassWithImages> {
        let rp_cfg = cfg::RenderPassCfg::from_yaml(&yaml_str)?; // parse YAML
        let mut color_storage: Vec<Vec<AttachmentDescription>> =
            Vec::with_capacity(rp_cfg.subpasses.len());
        let mut depth_storage: Vec<Option<AttachmentDescription>> =
            Vec::with_capacity(rp_cfg.subpasses.len());
        for subpass in &rp_cfg.subpasses {
            color_storage.push(
                subpass
                    .color_attachments
                    .iter()
                    .map(|att| att.description)
                    .collect(),
            );
            depth_storage.push(
                subpass
                    .depth_stencil_attachment
                    .as_ref()
                    .map(|att| att.description),
            );
        }

        let mut subpass_descriptions: Vec<SubpassDescription<'_>> =
            Vec::with_capacity(rp_cfg.subpasses.len());
        for (idx, subpass_cfg) in rp_cfg.subpasses.iter().enumerate() {
            subpass_descriptions.push(SubpassDescription {
                color_attachments: &color_storage[idx],
                depth_stencil_attachment: depth_storage[idx].as_ref(),
                subpass_dependencies: &subpass_cfg.subpass_dependencies,
            });
        }

        let rp_info = RenderPassInfo {
            debug_name: &rp_cfg.debug_name,
            viewport: rp_cfg.viewport,
            subpasses: &subpass_descriptions,
        };
        let render_pass = self.make_render_pass(&rp_info)?;

        let width = rp_cfg.viewport.scissor.w;
        let height = rp_cfg.viewport.scissor.h;

        let mut subpasses: Vec<RenderPassSubpassTargets> =
            Vec::with_capacity(rp_cfg.subpasses.len());

        for (subpass_idx, subpass_cfg) in rp_cfg.subpasses.iter().enumerate() {
            let mut targets = RenderPassSubpassTargets::default();

            for (attachment_idx, attachment_cfg) in subpass_cfg.color_attachments.iter().enumerate()
            {
                if attachment_idx >= targets.color_attachments.len() {
                    return Err(GPUError::Unimplemented(
                        "render pass YAML supports up to 4 color attachments per subpass",
                    ));
                }

                let debug_name = attachment_cfg.debug_name.clone().unwrap_or_else(|| {
                    format!(
                        "{}.subpass{}/color{}",
                        rp_cfg.debug_name, subpass_idx, attachment_idx
                    )
                });

                let image_info = ImageInfo {
                    debug_name: &debug_name,
                    dim: [width, height, 1],
                    layers: 1,
                    format: attachment_cfg.description.format,
                    mip_levels: 1,
                    initial_data: None,
                    samples: attachment_cfg.description.samples,
                };

                let image = self.make_image(&image_info)?;
                let aspect = if attachment_cfg.description.format == Format::D24S8 {
                    AspectMask::DepthStencil
                } else {
                    AspectMask::Color
                };

                targets.color_attachments[attachment_idx] = Some(RenderPassAttachmentInfo {
                    name: Some(debug_name),
                    view: ImageView {
                        img: image,
                        range: Default::default(),
                        aspect,
                    },
                    clear_value: attachment_cfg.clear_value,
                    description: attachment_cfg.description,
                });
            }

            if let Some(depth_cfg) = &subpass_cfg.depth_stencil_attachment {
                let debug_name = depth_cfg.debug_name.clone().unwrap_or_else(|| {
                    format!("{}.subpass{}/depth", rp_cfg.debug_name, subpass_idx)
                });

                let image_info = ImageInfo {
                    debug_name: &debug_name,
                    dim: [width, height, 1],
                    layers: 1,
                    format: depth_cfg.description.format,
                    mip_levels: 1,
                    initial_data: None,
                    samples: depth_cfg.description.samples,
                };

                let image = self.make_image(&image_info)?;
                targets.depth_attachment = Some(RenderPassAttachmentInfo {
                    name: Some(debug_name),
                    view: ImageView {
                        img: image,
                        range: Default::default(),
                        aspect: AspectMask::DepthStencil,
                    },
                    clear_value: depth_cfg.clear_value,
                    description: depth_cfg.description,
                });
            }

            subpasses.push(targets);
        }

        Ok(RenderPassWithImages {
            render_pass,
            subpasses,
        })
    }

    #[cfg(feature = "dashi-serde")]
    pub fn make_render_pass_from_yaml_file(&mut self, path: &str) -> Result<RenderPassWithImages> {
        let s = cfg::load_text(path)?;
        self.make_render_pass_from_yaml(&s)
    }

    #[cfg(feature = "dashi-serde")]
    pub fn make_bind_table_layout_from_yaml(
        &mut self,
        yaml_str: &str,
    ) -> Result<Handle<BindTableLayout>> {
        let cfg = cfg::BindTableLayoutCfg::from_yaml(yaml_str)?;
        let borrowed = cfg.borrow();
        let info = borrowed.info();
        self.make_bind_table_layout(&info)
    }

    #[cfg(feature = "dashi-serde")]
    pub fn make_bind_table_layouts_from_yaml(
        &mut self,
        yaml_str: &str,
    ) -> Result<Vec<Handle<BindTableLayout>>> {
        let cfgs = cfg::BindTableLayoutCfg::vec_from_yaml(yaml_str)?;
        cfgs.iter()
            .map(|cfg| {
                let borrowed = cfg.borrow();
                let info = borrowed.info();
                self.make_bind_table_layout(&info)
            })
            .collect()
    }

    #[cfg(feature = "dashi-serde")]
    pub fn make_bind_table_layout_from_yaml_file(
        &mut self,
        path: &str,
    ) -> Result<Handle<BindTableLayout>> {
        let s = cfg::load_text(path)?;
        self.make_bind_table_layout_from_yaml(&s)
    }

    #[cfg(feature = "dashi-serde")]
    pub fn make_bind_table_layouts_from_yaml_file(
        &mut self,
        path: &str,
    ) -> Result<Vec<Handle<BindTableLayout>>> {
        let s = cfg::load_text(path)?;
        self.make_bind_table_layouts_from_yaml(&s)
    }

    fn create_imageless_framebuffer(
        &self,
        render_pass: vk::RenderPass,
        attachment_formats: &[Format],
        width: u32,
        height: u32,
    ) -> Result<vk::Framebuffer, GPUError> {
        let mut view_formats_vk = Vec::with_capacity(attachment_formats.len());
        let mut attachment_image_infos = Vec::with_capacity(attachment_formats.len());
        let width = width.max(1);
        let height = height.max(1);

        for fmt in attachment_formats {
            let mut usage = vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED;

            if matches!(fmt, Format::D24S8) {
                usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            } else {
                usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
            }

            let vk_fmt = lib_to_vk_image_format(fmt);
            view_formats_vk.push(vk_fmt);
            let info = vk::FramebufferAttachmentImageInfo::builder()
                .usage(usage)
                .width(width)
                .height(height)
                .layer_count(1)
                .view_formats(std::slice::from_ref(view_formats_vk.last().unwrap()))
                .build();
            attachment_image_infos.push(info);
        }

        let mut attachments_info = vk::FramebufferAttachmentsCreateInfo::builder()
            .attachment_image_infos(&attachment_image_infos);

        let fb_info = vk::FramebufferCreateInfo::builder()
            .flags(vk::FramebufferCreateFlags::IMAGELESS_KHR)
            .render_pass(render_pass)
            .width(width)
            .height(height)
            .layers(1)
            .attachment_count(attachment_formats.len() as u32)
            .push_next(&mut attachments_info);

        let fb = unsafe { self.device.create_framebuffer(&fb_info, None) }?;

        Ok(fb)
    }

    /// Builds a render pass with the supplied subpass configuration.
    pub fn make_render_pass(
        &mut self,
        info: &RenderPassInfo,
    ) -> Result<Handle<RenderPass>, GPUError> {
        let mut attachments = Vec::with_capacity(256);
        let mut color_attachment_refs = Vec::with_capacity(256);
        let mut subpasses = Vec::with_capacity(256);
        let mut deps = Vec::with_capacity(256);
        let mut attachment_formats = Vec::with_capacity(256);
        let mut subpass_samples = Vec::with_capacity(info.subpasses.len());
        let mut subpass_formats = Vec::with_capacity(info.subpasses.len());
        for subpass in info.subpasses {
            let mut depth_stencil_attachment_ref = None;
            let attachment_offset = attachments.len();
            let color_offset = color_attachment_refs.len();
            let mut subpass_sample_info = SubpassSampleInfo::default();
            let mut subpass_format_info = SubpassAttachmentFormats::default();

            for (index, color_attachment) in subpass.color_attachments.iter().enumerate() {
                let attachment_desc = vk::AttachmentDescription {
                    format: lib_to_vk_image_format(&color_attachment.format),
                    samples: convert_sample_count(color_attachment.samples),
                    load_op: convert_load_op(color_attachment.load_op),
                    store_op: convert_store_op(color_attachment.store_op),
                    stencil_load_op: convert_load_op(color_attachment.stencil_load_op),
                    stencil_store_op: convert_store_op(color_attachment.stencil_store_op),
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    ..Default::default()
                };

                let attachment_ref = vk::AttachmentReference {
                    attachment: attachment_offset as u32 + index as u32,
                    layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                };

                attachments.push(attachment_desc);
                color_attachment_refs.push(attachment_ref);
                attachment_formats.push(color_attachment.format);
                subpass_sample_info
                    .color_samples
                    .push(color_attachment.samples);
                subpass_format_info
                    .color_formats
                    .push(color_attachment.format);
            }

            // Process depth-stencil attachment
            if let Some(depth_stencil_attachment) = subpass.depth_stencil_attachment {
                let depth_attachment_desc = vk::AttachmentDescription {
                    format: lib_to_vk_image_format(&depth_stencil_attachment.format),
                    samples: convert_sample_count(depth_stencil_attachment.samples),
                    load_op: convert_load_op(depth_stencil_attachment.load_op),
                    store_op: convert_store_op(depth_stencil_attachment.store_op),
                    stencil_load_op: convert_load_op(depth_stencil_attachment.stencil_load_op),
                    stencil_store_op: convert_store_op(depth_stencil_attachment.stencil_store_op),
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ..Default::default()
                };
                attachments.push(depth_attachment_desc);
                depth_stencil_attachment_ref = Some(vk::AttachmentReference {
                    attachment: (attachments.len() - 1) as u32,
                    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                });
                attachment_formats.push(depth_stencil_attachment.format);
                subpass_sample_info.depth_sample = Some(depth_stencil_attachment.samples);
                subpass_format_info.depth_format = Some(depth_stencil_attachment.format);
            }

            let colors = if color_attachment_refs.is_empty() {
                &[]
            } else {
                &color_attachment_refs[color_offset..]
            };

            // Create subpass description
            subpasses.push(match depth_stencil_attachment_ref.as_ref() {
                Some(d) => vk::SubpassDescription::builder()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(colors)
                    .depth_stencil_attachment(d)
                    .build(),
                None => vk::SubpassDescription::builder()
                    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                    .color_attachments(colors)
                    .build(),
            });

            for dep in subpass.subpass_dependencies {
                deps.push(vk::SubpassDependency {
                    src_subpass: dep.subpass_id,
                    dst_subpass: subpasses.len() as u32,
                    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: vk::PipelineStageFlags::VERTEX_SHADER,
                    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    dependency_flags: vk::DependencyFlags::empty(),
                });
            }

            subpass_samples.push(subpass_sample_info);
            subpass_formats.push(subpass_format_info);
        }
        // Create the render pass info
        let render_pass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ptr(),
            ..Default::default()
        };

        // Create render pass
        let render_pass = unsafe { self.device.create_render_pass(&render_pass_info, None) }?;
        self.set_name(render_pass, info.debug_name, vk::ObjectType::RENDER_PASS);

        let width = info.viewport.scissor.w.max(1);
        let height = info.viewport.scissor.h.max(1);

        let fb =
            self.create_imageless_framebuffer(render_pass, &attachment_formats, width, height)?;

        return Ok(self
            .render_passes
            .insert(RenderPass {
                raw: render_pass,
                fb,
                width,
                height,
                attachment_formats,
                subpass_samples,
                subpass_formats,
            })
            .unwrap());
    }

    pub fn render_pass_subpass_info(
        &self,
        render_pass: Handle<RenderPass>,
        subpass_id: u8,
    ) -> Option<RenderPassSubpassInfo> {
        let rp = self.render_passes.get_ref(render_pass)?;
        let index = subpass_id as usize;
        let formats = rp.subpass_formats.get(index)?.clone();
        let samples = rp.subpass_samples.get(index)?.clone();

        Some(RenderPassSubpassInfo {
            color_formats: formats.color_formats,
            depth_format: formats.depth_format,
            samples,
        })
    }

    /// Creates a compute pipeline layout from shader and bind table layouts.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn make_compute_pipeline_layout(
        &mut self,
        info: &ComputePipelineLayoutInfo,
    ) -> Result<Handle<ComputePipelineLayout>, GPUError> {
        let shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE) // HAS to be compute.
            .module(self.create_shader_module(info.shader.spirv).unwrap())
            .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap()) // Entry point is usually "main"
            .build();

        let layout = self.create_pipeline_layout(&info.bt_layouts)?;

        return Ok(self
            .compute_pipeline_layouts
            .insert(ComputePipelineLayout {
                layout,
                shader_stage,
            })
            .unwrap());
    }

    /// Creates a graphics pipeline layout including shader stages and vertex setup.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn make_graphics_pipeline_layout(
        &mut self,
        info: &GraphicsPipelineLayoutInfo,
    ) -> Result<Handle<GraphicsPipelineLayout>, GPUError> {
        let mut shader_stages: Vec<vk::PipelineShaderStageCreateInfo> =
            Vec::with_capacity(info.shaders.len());
        for shader_info in info.shaders {
            let stage_flags = match shader_info.stage {
                ShaderType::Vertex => vk::ShaderStageFlags::VERTEX,
                ShaderType::Fragment => vk::ShaderStageFlags::FRAGMENT,
                ShaderType::All => vk::ShaderStageFlags::ALL,
                other => {
                    return Err(GPUError::UnsupportedShaderStage(other));
                }
            };

            shader_stages.push(
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(stage_flags)
                    .module(self.create_shader_module(shader_info.spirv).unwrap())
                    .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap()) // Entry point is usually "main"
                    .build(),
            );
        }

        // Step 2: Create Vertex Input State
        let vertex_binding_description = vk::VertexInputBindingDescription::builder()
            .binding(0) // Binding 0 for now
            .stride(info.vertex_info.stride as u32)
            .input_rate(match info.vertex_info.rate {
                VertexRate::Vertex => vk::VertexInputRate::VERTEX,
            })
            .build();

        let vertex_attribute_descriptions: Vec<vk::VertexInputAttributeDescription> = info
            .vertex_info
            .entries
            .iter()
            .map(|entry| {
                vk::VertexInputAttributeDescription::builder()
                    .location(entry.location as u32)
                    .binding(0) // Binding 0 for now
                    .format(match entry.format {
                        ShaderPrimitiveType::Vec4 => vk::Format::R32G32B32A32_SFLOAT,
                        ShaderPrimitiveType::Vec3 => vk::Format::R32G32B32_SFLOAT,
                        ShaderPrimitiveType::Vec2 => vk::Format::R32G32_SFLOAT,
                        ShaderPrimitiveType::IVec4 => vk::Format::R32G32B32A32_SINT,
                        ShaderPrimitiveType::UVec4 => vk::Format::R32G32B32A32_UINT,
                    })
                    .offset(entry.offset as u32)
                    .build()
            })
            .collect();

        // Step 3: Create Input Assembly State
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(match info.details.topology {
                Topology::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            })
            .primitive_restart_enable(false)
            .build();

        // Step 5: Rasterization State
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(match info.details.culling {
                CullMode::Back => vk::CullModeFlags::BACK,
                CullMode::None => vk::CullModeFlags::NONE,
            })
            .front_face(match info.details.front_face {
                VertexOrdering::Clockwise => vk::FrontFace::CLOCKWISE,
                VertexOrdering::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
            })
            .depth_bias_enable(false)
            .line_width(1.0)
            .build();

        // Step 6: Multisampling
        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(info.details.min_sample_shading > 0.0)
            .rasterization_samples(info.details.sample_count.into())
            .min_sample_shading(info.details.min_sample_shading)
            .build();

        // Step 7: Depth and Stencil State (depth testing)
        let depth_stencil_state = match info.details.depth_test {
            Some(depth) => Some(
                vk::PipelineDepthStencilStateCreateInfo::builder()
                    .depth_test_enable(depth.should_test)
                    .depth_write_enable(depth.should_write)
                    .min_depth_bounds(0.0)
                    .max_depth_bounds(1.0)
                    .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                    .build(),
            ),
            None => None,
        };

        // Step 9: Create Pipeline Layout (assume we have a layout creation function)
        let layout = self.create_pipeline_layout(&info.bt_layouts)?;

        self.set_name(layout, info.debug_name, vk::ObjectType::PIPELINE_LAYOUT);

        let color_blends: Vec<vk::PipelineColorBlendAttachmentState> = info
            .details
            .color_blend_states
            .iter()
            .map(|c| c.clone().into())
            .collect();

        let mut dynamic_states = info.details.dynamic_states.clone();
        for required in [DynamicState::Viewport, DynamicState::Scissor] {
            if !dynamic_states.contains(&required) {
                dynamic_states.push(required);
            }
        }

        let dynamic_states: Vec<vk::DynamicState> =
            dynamic_states.iter().map(|s| (*s).into()).collect();

        return Ok(self
            .gfx_pipeline_layouts
            .insert(GraphicsPipelineLayout {
                layout,
                shader_stages,
                vertex_input: vertex_binding_description,
                vertex_attribs: vertex_attribute_descriptions,
                rasterizer,
                multisample: multisampling,
                depth_stencil: depth_stencil_state,
                input_assembly,
                color_blend_states: color_blends,
                dynamic_states,
                sample_count: info.details.sample_count,
                min_sample_shading: info.details.min_sample_shading,
            })
            .unwrap());
    }

    /// Builds a compute pipeline from an existing layout.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn make_compute_pipeline(
        &mut self,
        info: &ComputePipelineInfo,
    ) -> Result<Handle<ComputePipeline>, GPUError> {
        let layout = self.compute_pipeline_layouts.get_ref(info.layout).unwrap();

        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .layout(layout.layout)
            .stage(layout.shader_stage)
            .build();

        let compute_pipelines = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()
        };

        self.set_name(
            compute_pipelines[0],
            info.debug_name,
            vk::ObjectType::PIPELINE,
        );

        return Ok(self
            .compute_pipelines
            .insert(ComputePipeline {
                raw: compute_pipelines[0],
                layout: info.layout,
            })
            .unwrap());
    }

    /// Builds a graphics pipeline referencing a render pass and layout.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn make_graphics_pipeline(
        &mut self,
        info: &GraphicsPipelineInfo,
    ) -> Result<Handle<GraphicsPipeline>, GPUError> {
        let layout = self.gfx_pipeline_layouts.get_ref(info.layout).unwrap();
        let subpass_index = info.subpass_id as usize;
        let expected_subpass = &info.subpass_samples;

        for (attachment_idx, expected) in expected_subpass.color_samples.iter().enumerate() {
            if layout.sample_count != *expected {
                return Err(GPUError::MismatchedSampleCount {
                    context: format!(
                        "render pass subpass {} color attachment {}",
                        subpass_index, attachment_idx
                    ),
                    expected: *expected,
                    actual: layout.sample_count,
                });
            }
        }

        if let Some(depth_sample) = expected_subpass.depth_sample {
            if layout.sample_count != depth_sample {
                return Err(GPUError::MismatchedSampleCount {
                    context: format!("render pass subpass {} depth attachment", subpass_index),
                    expected: depth_sample,
                    actual: layout.sample_count,
                });
            }
        }

        if info.attachment_formats.len() != info.subpass_samples.color_samples.len() {
            return Err(GPUError::LibraryError(format!(
                "attachment formats differ between info and subpass_samples. ({} vs {})",
                info.attachment_formats.len(),
                info.subpass_samples.color_samples.len()
            )));
        }

        if info.depth_format.is_some() != info.subpass_samples.depth_sample.is_some() {
            return Err(GPUError::LibraryError(format!(
                "depth format mismatch between pipeline and subpass ({} vs {})",
                info.depth_format.is_some(),
                info.subpass_samples.depth_sample.is_some()
            )));
        }

        let mut attachments = Vec::with_capacity(info.attachment_formats.len() + 1);
        let mut attachment_formats = Vec::with_capacity(info.attachment_formats.len() + 1);
        let mut color_attachment_refs = Vec::with_capacity(info.attachment_formats.len());

        for (index, (format, samples)) in info
            .attachment_formats
            .iter()
            .zip(info.subpass_samples.color_samples.iter())
            .enumerate()
        {
            let attachment_desc = vk::AttachmentDescription {
                format: lib_to_vk_image_format(format),
                samples: convert_sample_count(*samples),
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::DONT_CARE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ..Default::default()
            };

            let attachment_ref = vk::AttachmentReference {
                attachment: index as u32,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };

            attachments.push(attachment_desc);
            color_attachment_refs.push(attachment_ref);
            attachment_formats.push(*format);
        }

        let mut depth_attachment_ref = None;
        if let Some(depth_format) = info.depth_format {
            let depth_samples = info
                .subpass_samples
                .depth_sample
                .ok_or(GPUError::LibraryError(format!("Depth format active, but did not provide any depth sample!")))?;
            let depth_attachment = vk::AttachmentDescription {
                format: lib_to_vk_image_format(&depth_format),
                samples: convert_sample_count(depth_samples),
                load_op: vk::AttachmentLoadOp::DONT_CARE,
                store_op: vk::AttachmentStoreOp::DONT_CARE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ..Default::default()
            };
            depth_attachment_ref = Some(vk::AttachmentReference {
                attachment: attachments.len() as u32,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            });
            attachments.push(depth_attachment);
            attachment_formats.push(depth_format);
        }

        let target_subpass = info.subpass_id as usize;
        let mut subpasses = Vec::with_capacity(target_subpass + 1);
        for idx in 0..=target_subpass {
            if idx == target_subpass {
                subpasses.push(match depth_attachment_ref.as_ref() {
                    Some(depth) => vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&color_attachment_refs)
                        .depth_stencil_attachment(depth)
                        .build(),
                    None => vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .color_attachments(&color_attachment_refs)
                        .build(),
                });
            } else {
                subpasses.push(
                    vk::SubpassDescription::builder()
                        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                        .build(),
                );
            }
        }

        let dummy_rp_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .build();

        let dummy_render_pass = unsafe { self.device.create_render_pass(&dummy_rp_info, None) }?;
        self.set_name(
            dummy_render_pass,
            info.debug_name,
            vk::ObjectType::RENDER_PASS,
        );

        let width = 1;
        let height = 1;

        let fb = self.create_imageless_framebuffer(
            dummy_render_pass,
            &attachment_formats,
            width,
            height,
        )?;

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&[layout.vertex_input])
            .vertex_attribute_descriptions(&layout.vertex_attribs)
            .build();

        // Step 4: Viewport and Scissor State
        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(1.0)
            .height(1.0)
            .min_depth(0.0)
            .max_depth(1.0)
            .build();

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(vk::Extent2D {
                width: 1,
                height: 1,
            })
            .build();

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&[viewport])
            .scissors(&[scissor])
            .build();

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&layout.color_blend_states)
            .build();

        // Step 10: Create Graphics Pipeline
        let dynamic_state_info = if !layout.dynamic_states.is_empty() {
            Some(
                vk::PipelineDynamicStateCreateInfo::builder()
                    .dynamic_states(&layout.dynamic_states)
                    .build(),
            )
        } else {
            None
        };

        let mut pipeline_builder = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&layout.shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&layout.input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&layout.rasterizer)
            .multisample_state(&layout.multisample)
            .color_blend_state(&color_blend_state)
            .layout(layout.layout)
            .render_pass(dummy_render_pass)
            .subpass(info.subpass_id as u32);

        if let Some(d) = layout.depth_stencil.as_ref() {
            pipeline_builder = pipeline_builder.depth_stencil_state(d);
        }

        if let Some(ref dyn_info) = dynamic_state_info {
            pipeline_builder = pipeline_builder.dynamic_state(dyn_info);
        }

        let pipeline_info = pipeline_builder.build();

        let graphics_pipelines_result = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        };

        let mut subpass_samples = vec![SubpassSampleInfo::default(); target_subpass + 1];
        subpass_samples[target_subpass] = info.subpass_samples.clone();

        let mut subpass_formats = vec![SubpassAttachmentFormats::default(); target_subpass + 1];
        subpass_formats[target_subpass] = SubpassAttachmentFormats {
            color_formats: info.attachment_formats.clone(),
            depth_format: info.depth_format,
        };

        let render_pass_handle = self
            .render_passes
            .insert(RenderPass {
                raw: dummy_render_pass,
                fb,
                width,
                height,
                attachment_formats,
                subpass_samples,
                subpass_formats,
            })
            .unwrap();

        let graphics_pipelines = graphics_pipelines_result?;

        self.set_name(
            graphics_pipelines[0],
            info.debug_name,
            vk::ObjectType::PIPELINE,
        );

        return Ok(self
            .gfx_pipelines
            .insert(GraphicsPipeline {
                raw: graphics_pipelines[0],
                render_pass: render_pass_handle,
                layout: info.layout,
                subpass: info.subpass_id,
                subpass_formats: SubpassAttachmentFormats {
                    color_formats: info.attachment_formats.clone(),
                    depth_format: info.depth_format,
                },
                attachment_formats: info.attachment_formats.clone(),
                depth_format: info.depth_format,
                sample_count: layout.sample_count,
                subpass_samples: info.subpass_samples.clone(),
            })
            .unwrap());
    }
}
