mod error;
use crate::{
    cmd::{CommandStream, Executable},
    driver::{
        command::{CopyBuffer, CopyBufferImage},
        state::SubresourceRange,
    },
    utils::{Handle, Pool},
    CommandRing,
};
use ash::*;
pub use error::*;
use std::{
    collections::HashMap,
    ffi::{c_char, c_void, CStr, CString},
    mem::ManuallyDrop,
};
use vk_mem::Alloc;

pub mod device_selector;
pub use device_selector::*;
pub mod structs;
pub use structs::*;
pub mod builders;
pub mod commands;
pub use commands::*;
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
use command_pool::CommandPool;


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

#[derive(Clone, Default)]
pub struct RenderPass {
    pub(super) raw: vk::RenderPass,
    pub(super) fb: vk::Framebuffer,
    pub(super) viewport: Viewport,
    pub(super) attachment_formats: Vec<Format>,
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
    ctx: *mut Context,
    pool: *mut CommandPool,
    curr_rp: Option<Handle<RenderPass>>,
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

pub struct Context {
    pub(super) entry: ash::Entry,
    pub(super) instance: ash::Instance,
    pub(super) pdevice: vk::PhysicalDevice,
    pub(super) device: ash::Device,
    pub(super) properties: ash::vk::PhysicalDeviceProperties,
    pub(super) gfx_pool: CommandPool,
    pub(super) compute_pool: Option<CommandPool>,
    pub(super) transfer_pool: Option<CommandPool>,
    pub(super) allocator: ManuallyDrop<vk_mem::Allocator>,
    pub(super) gfx_queue: Queue,
    pub(super) compute_queue: Option<Queue>,
    pub(super) transfer_queue: Option<Queue>,
    pub(super) buffers: Pool<Buffer>,
    pub(super) render_passes: Pool<RenderPass>,
    pub(super) semaphores: Pool<Semaphore>,
    pub(super) fences: Pool<Fence>,
    pub(super) images: Pool<Image>,
    pub(super) image_views: Pool<VkImageView>,
    pub(super) image_view_cache: HashMap<ImageView, Handle<VkImageView>>,
    pub(super) samplers: Pool<Sampler>,
    pub(super) bind_group_layouts: Pool<BindGroupLayout>,
    pub(super) bind_groups: Pool<BindGroup>,
    pub(super) bind_table_layouts: Pool<BindTableLayout>,
    pub(super) bind_tables: Pool<BindTable>,
    pub(super) gfx_pipeline_layouts: Pool<GraphicsPipelineLayout>,
    pub(super) gfx_pipelines: Pool<GraphicsPipeline>,
    pub(super) compute_pipeline_layouts: Pool<ComputePipelineLayout>,
    pub(super) compute_pipelines: Pool<ComputePipeline>,

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

impl std::panic::UnwindSafe for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        if let (Some(utils), Some(messenger)) = (&self.debug_utils, self.debug_messenger) {
            unsafe {
                utils.destroy_debug_utils_messenger(messenger, None);
            }
        }
    }
}

impl Context {
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

        let mut descriptor_indexing =
            vk::PhysicalDeviceDescriptorIndexingFeatures::builder().build();
        let features = vk::PhysicalDeviceFeatures::builder()
            .shader_clip_distance(true)
            .build();

        let mut features2 = vk::PhysicalDeviceFeatures2::builder()
            .features(features)
            .push_next(&mut descriptor_indexing)
            .build();

        unsafe { instance.get_physical_device_features2(pdevice, &mut features2) };

        let mut features16bit = vk::PhysicalDevice16BitStorageFeatures::builder()
            .uniform_and_storage_buffer16_bit_access(true)
            .build();

        // Bind table enabled
        if descriptor_indexing.shader_sampled_image_array_non_uniform_indexing <= 0
            && descriptor_indexing.descriptor_binding_sampled_image_update_after_bind <= 0
            && descriptor_indexing.shader_uniform_buffer_array_non_uniform_indexing <= 0
            && descriptor_indexing.descriptor_binding_uniform_buffer_update_after_bind <= 0
            && descriptor_indexing.shader_storage_buffer_array_non_uniform_indexing <= 0
            && descriptor_indexing.descriptor_binding_storage_buffer_update_after_bind <= 0
        {
            features2 = Default::default();
        }

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

        let device = unsafe {
            instance.create_device(
                pdevice,
                &vk::DeviceCreateInfo::builder()
                    .enabled_extension_names(&extensions_to_enable)
                    .queue_create_infos(&queue_infos)
                    .push_next(&mut features2)
                    .push_next(&mut features16bit)
                    .push_next(&mut imageless)
                    .build(),
                None,
            )
        }?;

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
    pub fn headless(info: &ContextInfo) -> Result<Box<Self>> {
        let enable_validation = std::env::var("DASHI_VALIDATION")
            .map(|v| v == "1")
            .unwrap_or(false);
        let (
            entry,
            instance,
            pdevice,
            device,
            properties,
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

        let mut ctx = Box::new(Context {
            entry,
            instance,
            pdevice,
            device,
            properties,
            gfx_pool: CommandPool::default(),
            compute_pool: None,
            transfer_pool: None,
            allocator: ManuallyDrop::new(allocator),
            gfx_queue,
            compute_queue,
            transfer_queue,

            buffers: Default::default(),
            render_passes: Default::default(),
            semaphores: Default::default(),
            fences: Default::default(),
            images: Default::default(),
            image_views: Default::default(),
            image_view_cache: HashMap::new(),
            samplers: Default::default(),
            bind_group_layouts: Default::default(),
            bind_groups: Default::default(),
            bind_table_layouts: Default::default(),
            bind_tables: Default::default(),
            gfx_pipeline_layouts: Default::default(),
            gfx_pipelines: Default::default(),
            compute_pipeline_layouts: Default::default(),
            compute_pipelines: Default::default(),
            gpu_timers: Vec::new(),
            timestamp_period: properties.limits.timestamp_period,
            headless: true,

            #[cfg(feature = "dashi-sdl2")]
            sdl_context: sdl2::init().unwrap(),
            #[cfg(feature = "dashi-sdl2")]
            sdl_video: None,
            debug_utils,
            debug_messenger,
        });
        let ctx_ptr = &mut *ctx as *mut _;
        ctx.gfx_pool = CommandPool::new(ctx_ptr, QueueType::Graphics)?;
        if ctx.compute_queue.is_some() {
            ctx.compute_pool = Some(CommandPool::new(ctx_ptr, QueueType::Compute)?);
        }
        if ctx.transfer_queue.is_some() {
            ctx.transfer_pool = Some(CommandPool::new(ctx_ptr, QueueType::Transfer)?);
        }
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
    pub fn new(info: &ContextInfo) -> Result<Box<Self>> {
        let enable_validation = std::env::var("DASHI_VALIDATION")
            .map(|v| v == "1")
            .unwrap_or(false);
        let (
            entry,
            instance,
            pdevice,
            device,
            properties,
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

        let mut ctx = Box::new(Context {
            entry,
            instance,
            pdevice,
            device,
            properties,
            gfx_pool: CommandPool::default(),
            compute_pool: None,
            transfer_pool: None,
            allocator: ManuallyDrop::new(allocator),
            gfx_queue,
            compute_queue,
            transfer_queue,

            buffers: Default::default(),
            render_passes: Default::default(),
            semaphores: Default::default(),
            fences: Default::default(),
            images: Default::default(),
            image_views: Default::default(),
            image_view_cache: HashMap::new(),
            samplers: Default::default(),
            bind_group_layouts: Default::default(),
            bind_groups: Default::default(),
            bind_table_layouts: Default::default(),
            bind_tables: Default::default(),
            gfx_pipeline_layouts: Default::default(),
            gfx_pipelines: Default::default(),
            compute_pipeline_layouts: Default::default(),
            compute_pipelines: Default::default(),
            gpu_timers: Vec::new(),
            timestamp_period: properties.limits.timestamp_period,
            headless: false,

            #[cfg(feature = "dashi-sdl2")]
            sdl_context,
            #[cfg(feature = "dashi-sdl2")]
            sdl_video: Some(sdl_video),
            debug_utils,
            debug_messenger,
        });
        let ctx_ptr = &mut *ctx as *mut _;
        ctx.gfx_pool = CommandPool::new(ctx_ptr, QueueType::Graphics)?;
        if ctx.compute_queue.is_some() {
            ctx.compute_pool = Some(CommandPool::new(ctx_ptr, QueueType::Compute)?);
        }
        if ctx.transfer_queue.is_some() {
            ctx.transfer_pool = Some(CommandPool::new(ctx_ptr, QueueType::Transfer)?);
        }
        ctx.init_gpu_timers(1)?;
        Ok(ctx)
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
    pub fn make_command_pool(&mut self, ty: QueueType) -> Result<CommandPool> {
        CommandPool::new(self as *mut _, ty)
    }

    /// Creates a ringbuffer of CommandQueues
    pub fn make_command_ring(&mut self, info: &CommandQueueInfo2) -> Result<CommandRing> {
        Ok(CommandRing::new(self, info.debug_name, 3, info.queue_type)?)
    }

    /// Allocate and begin recording a new command list.
    ///
    /// The returned [`CommandQueue`] starts in the recording state and may be
    /// submitted once. [`Self::submit`] will end the list automatically if it
    /// is still recording.
    pub fn make_command_queue(&mut self, info: &CommandQueueInfo2) -> Result<CommandQueue> {
        let pool_ptr: *mut CommandPool = match info.queue_type {
            QueueType::Graphics => &mut self.gfx_pool as *mut _,
            QueueType::Compute => self
                .compute_pool
                .as_mut()
                .unwrap_or(&mut self.gfx_pool) as *mut _,
            QueueType::Transfer => self
                .transfer_pool
                .as_mut()
                .unwrap_or(self.compute_pool.as_mut().unwrap_or(&mut self.gfx_pool))
                as *mut _,
        };

        unsafe { (*pool_ptr).begin(info.debug_name, info.parent.is_some()) }
    }

    /// Allocate and begin recording a new command list.
    ///
    /// The returned [`CommandQueue`] starts in the recording state and may be
    /// submitted once. [`Self::submit`] will end the list automatically if it
    /// is still recording.
    pub fn begin_command_queue(&mut self, info: &CommandQueueInfo) -> Result<CommandQueue> {
        self.make_command_queue(&CommandQueueInfo2 {
            debug_name: info.debug_name,
            parent: None,
            queue_type: info.queue_type,
        })
    }
    fn oneshot_transition_image(&mut self, img: ImageView, layout: vk::ImageLayout) {
        let view_handle = self.get_or_create_image_view(&img).unwrap();
        let mut list = self
            .begin_command_queue(&CommandQueueInfo {
                debug_name: "",
                ..Default::default()
            })
            .unwrap();
        self.transition_image(list.cmd_buf, view_handle, layout);
        let fence = self.submit(&mut list, &Default::default()).unwrap();
        self.wait(fence).unwrap();
        self.destroy_cmd_queue(list);
    }

    fn oneshot_transition_image_noview(&mut self, img: Handle<Image>, layout: vk::ImageLayout) {
        let tmp_view = ImageView {
            img,
            layer: 0,
            mip_level: 0,
            aspect: Default::default(),
        };
        let view_handle = self.get_or_create_image_view(&tmp_view).unwrap();

        let mut list = self
            .begin_command_queue(&CommandQueueInfo {
                debug_name: "oneshot_transition",
                ..Default::default()
            })
            .unwrap();
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
        self.transition_image_stages(
            cmd,
            img,
            layout,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            Default::default(),
            Default::default(),
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
        let aspect: vk::ImageAspectFlags = info.aspect.into();
        let sub_range = vk::ImageSubresourceRange::builder()
            .base_array_layer(info.layer)
            .layer_count(1)
            .base_mip_level(info.mip_level)
            .level_count(1)
            .aspect_mask(aspect)
            .build();

        let view = unsafe {
            self.device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(img.img)
                    .format(lib_to_vk_image_format(&img.format))
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

        let mut aspect = vk::ImageAspectFlags::COLOR;
        if info.format == Format::D24S8 {
            aspect = vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL;
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
                    .samples(vk::SampleCountFlags::TYPE_1)
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

        match self.images.insert(Image {
            img: image,
            alloc: allocation,
            layouts: vec![vk::ImageLayout::UNDEFINED; info.mip_levels as usize],
            sub_layers: vk::ImageSubresourceLayers::builder()
                .layer_count(info.dim[2] as u32)
                .mip_level(0)
                .base_array_layer(0)
                .aspect_mask(aspect)
                .build(),
            dim: info.dim,
            format: info.format,
        }) {
            Some(h) => {
                self.init_image(h, &info)?;
                return Ok(h);
            }
            None => Err(GPUError::SlotError()),
        }
    }

    pub fn map_buffer_mut<T>(&mut self, buf: Handle<Buffer>) -> Result<&mut [T]> {
        let buf = match self.buffers.get_ref(buf) {
            Some(it) => it,
            None => return Err(GPUError::SlotError()),
        };

        let mut alloc: vk_mem::Allocation = unsafe { std::mem::transmute_copy(&buf.alloc) };
        let mapped = unsafe { self.allocator.map_memory(&mut alloc) }?;
        let mut typed_map: *mut T = unsafe { std::mem::transmute(mapped) };
        typed_map = unsafe { typed_map.offset(buf.offset as isize) };
        return Ok(unsafe {
            std::slice::from_raw_parts_mut(typed_map, buf.size as usize / std::mem::size_of::<T>())
        });
    }

    pub fn map_buffer<T>(&mut self, buf: Handle<Buffer>) -> Result<&[T]> {
        let buf = match self.buffers.get_ref(buf) {
            Some(it) => it,
            None => return Err(GPUError::SlotError()),
        };

        let mut alloc: vk_mem::Allocation = unsafe { std::mem::transmute_copy(&buf.alloc) };
        let mapped = unsafe { self.allocator.map_memory(&mut alloc) }?;
        let mut typed_map: *mut T = unsafe { std::mem::transmute(mapped) };
        typed_map = unsafe { typed_map.offset(buf.offset as isize) };
        return Ok(unsafe {
            std::slice::from_raw_parts(typed_map, buf.size as usize / std::mem::size_of::<T>())
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

                let mut list = self.begin_command_queue(&Default::default())?;
                let mut stream = CommandStream::new().begin();
                stream.copy_buffers(&CopyBuffer {
                    src: staging,
                    dst: buf,
                    src_offset: 0,
                    dst_offset: 0,
                    amount: unsafe { info.initial_data.unwrap_unchecked().len() } as u32,
                });

                stream.end().append(&mut list);

                let fence = self.submit(&mut list, &Default::default())?;
                self.wait(fence.clone())?;
                self.destroy_cmd_queue(list);
                self.destroy_buffer(staging);
            }
            MemoryVisibility::CpuAndGpu => {
                let mapped: &mut [u8] = self.map_buffer_mut(buf)?;
                mapped.copy_from_slice(unsafe { info.initial_data.unwrap_unchecked() });
                self.unmap_buffer(buf)?;
            }
        }

        return Ok(());
    }

    fn init_image(&mut self, image: Handle<Image>, info: &ImageInfo) -> Result<()> {
        let mut list = self.begin_command_queue(&Default::default())?;
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

        let mut list = self.begin_command_queue(&Default::default())?;
        let mut cmd = CommandStream::new().begin();
        cmd.copy_buffer_to_image(&CopyBufferImage {
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
                cmd.copy_buffer_to_image(&CopyBufferImage {
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

        cmd.end().append(&mut list);
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
            ptr: self.map_buffer_mut(buffer)?.as_mut_ptr(),
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
        match self.buffers.insert(cpy) {
            Some(handle) => {
                return Some(handle);
            }
            None => return None,
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
            match self.buffers.insert(Buffer {
                buf: buffer,
                alloc: allocation,
                size: info.byte_size,
                offset: 0,
                suballocated: false,
            }) {
                Some(handle) => {
                    self.init_buffer(handle, info)?;
                    return Ok(handle);
                }
                None => return Err(GPUError::SlotError()),
            }
        }
    }

    /// Completes any outstanding resource cleanup on the context.
    ///
    /// # Prerequisites
    /// - The context must still be alive.
    /// - Ensure the GPU has finished using any resources scheduled for cleanup.
    ///
    /// Currently this method is a no-op and exists for API completeness.
    pub fn clean_up(&mut self) {}

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

        // Bind group layouts
        self.bind_group_layouts
            .for_each_occupied_mut(|layout| unsafe {
                self.device
                    .destroy_descriptor_set_layout(layout.layout, None);
                self.device.destroy_descriptor_pool(layout.pool, None);
            });

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

        // Buffers
        self.buffers.for_each_occupied_mut(|buf| {
            if !buf.suballocated {
                unsafe { self.allocator.destroy_buffer(buf.buf, &mut buf.alloc) };
            }
        });

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

        let mut flags = Vec::new();
        let mut bindings = Vec::new();
        for shader_info in info.shaders.iter() {
            for variable in shader_info.variables.iter() {
                let descriptor_type = match variable.var_type {
                    BindGroupVariableType::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
                    BindGroupVariableType::DynamicUniform => {
                        vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    }
                    BindGroupVariableType::Storage => vk::DescriptorType::STORAGE_BUFFER,
                    BindGroupVariableType::SampledImage => {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    }
                    BindGroupVariableType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
                    BindGroupVariableType::DynamicStorage => {
                        vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                    }
                };

                let stage_flags = match shader_info.shader_type {
                    ShaderType::Vertex => vk::ShaderStageFlags::VERTEX,
                    ShaderType::Fragment => vk::ShaderStageFlags::FRAGMENT,
                    ShaderType::Compute => vk::ShaderStageFlags::COMPUTE,
                    ShaderType::All => vk::ShaderStageFlags::ALL,
                };

                flags.push(
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                        | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                );
                let layout_binding = vk::DescriptorSetLayoutBinding::builder()
                    .binding(variable.binding)
                    .descriptor_type(descriptor_type)
                    .descriptor_count(variable.count) // Assuming one per binding
                    .stage_flags(stage_flags)
                    .build();

                bindings.push(layout_binding);
            }
        }
        let mut layout_binding_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
            .binding_flags(&flags)
            .build();

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .push_next(&mut layout_binding_info)
            .build();

        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)?
        };

        let pool_sizes = bindings
            .iter()
            .map(|binding| {
                vk::DescriptorPoolSize::builder()
                    .ty(binding.descriptor_type)
                    .descriptor_count(1) // Assuming one descriptor per binding
                    .build()
            })
            .collect::<Vec<_>>();

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(MAX_DESCRIPTOR_SETS);

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
            })
            .unwrap());
    }

    /// Creates a bind group layout for standard resources.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn make_bind_group_layout(
        &mut self,
        info: &BindGroupLayoutInfo,
    ) -> Result<Handle<BindGroupLayout>, GPUError> {
        let max_descriptor_sets: u32 = 2048;
        let mut bindings = Vec::new();
        for shader_info in info.shaders.iter() {
            for variable in shader_info.variables.iter() {
                let descriptor_type = match variable.var_type {
                    BindGroupVariableType::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
                    BindGroupVariableType::DynamicUniform => {
                        vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
                    }
                    BindGroupVariableType::Storage => vk::DescriptorType::STORAGE_BUFFER,
                    BindGroupVariableType::SampledImage => {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    }
                    BindGroupVariableType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
                    BindGroupVariableType::DynamicStorage => {
                        vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                    }
                };

                let stage_flags = match shader_info.shader_type {
                    ShaderType::Vertex => vk::ShaderStageFlags::VERTEX,
                    ShaderType::Fragment => vk::ShaderStageFlags::FRAGMENT,
                    ShaderType::Compute => vk::ShaderStageFlags::COMPUTE,
                    ShaderType::All => vk::ShaderStageFlags::ALL,
                };

                let layout_binding = vk::DescriptorSetLayoutBinding::builder()
                    .binding(variable.binding)
                    .descriptor_type(descriptor_type)
                    .descriptor_count(variable.count) // Assuming one per binding
                    .stage_flags(stage_flags)
                    .build();

                bindings.push(layout_binding);
            }
        }

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .build();

        let descriptor_set_layout = unsafe {
            self.device
                .create_descriptor_set_layout(&layout_info, None)?
        };

        let pool_sizes = bindings
            .iter()
            .map(|binding| {
                vk::DescriptorPoolSize::builder()
                    .ty(binding.descriptor_type)
                    .descriptor_count(max_descriptor_sets) // Assuming one per binding
                    .build()
            })
            .collect::<Vec<_>>();

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(max_descriptor_sets);

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

        // Step 4: Return the BindGroupLayout
        return Ok(self
            .bind_group_layouts
            .insert(BindGroupLayout {
                pool: descriptor_pool,
                layout: descriptor_set_layout,
                variables: info
                    .shaders
                    .iter()
                    .flat_map(|shader| shader.variables.iter().cloned())
                    .collect(),
            })
            .unwrap());
    }

    /// Updates an existing bind group with new resource bindings.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid. (If using OpenXR)
    /// - Synchronization primitives are handled during presentation.
    pub fn update_bind_group(&mut self, info: &BindGroupUpdateInfo) -> Result<()> {
        let bg = self.bind_groups.get_ref(info.bg).unwrap();
        let descriptor_set = bg.set;

        // Step 2: Prepare the write operations for the descriptor set
        let mut write_descriptor_sets = Vec::new();
        let mut buffer_infos = Vec::new();
        let mut image_infos = Vec::new();

        for binding_info in info.bindings.iter() {
            for res in binding_info.resources {
                match &res.resource {
                    ShaderResource::Buffer(handle) => {
                        let buffer = self.buffers.get_ref(*handle).unwrap();
                        let buffer_info = vk::DescriptorBufferInfo::builder()
                            .buffer(buffer.buf)
                            .offset(buffer.offset as u64)
                            .range(buffer.size as u64)
                            .build();

                        buffer_infos.push(buffer_info);

                        let write_descriptor_set = vk::WriteDescriptorSet::builder()
                            .dst_set(descriptor_set)
                            .dst_binding(binding_info.binding)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER) // Assuming a uniform buffer for now
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
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER) // Assuming a sampled image
                            .dst_array_element(res.slot)
                            .image_info(&image_infos[image_infos.len() - 1..])
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::Dynamic(alloc) => {
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
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC) // Assuming a uniform buffer for now
                            .buffer_info(&buffer_infos[buffer_infos.len() - 1..])
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::DynamicStorage(alloc) => {
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
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC) // Assuming a uniform buffer for now
                            .buffer_info(&buffer_infos[buffer_infos.len() - 1..])
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::StorageBuffer(handle) => {
                        let buffer = self.buffers.get_ref(*handle).unwrap();

                        let buffer_info = vk::DescriptorBufferInfo::builder()
                            .buffer(buffer.buf)
                            .offset(buffer.offset as u64)
                            .range(buffer.size as u64)
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
                    ShaderResource::ConstBuffer(_) => {
                        return Err(GPUError::Unimplemented(
                            "Constant buffers are not supported",
                        ))
                    }
                }
            }
        }

        unsafe {
            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        Ok(())
    }

    /// Updates an existing bind table with new resource bindings.
    pub fn update_bind_table(&mut self, info: &BindTableUpdateInfo) -> Result<()> {
        let table = self.bind_tables.get_ref(info.table).unwrap();
        let descriptor_set = table.set;

        let mut write_descriptor_sets = Vec::new();
        let mut buffer_infos = Vec::new();
        let mut image_infos = Vec::new();

        for binding_info in info.bindings.iter() {
            for res in binding_info.resources {
                match &res.resource {
                    ShaderResource::Buffer(handle) => {
                        let buffer = self.buffers.get_ref(*handle).unwrap();
                        let buffer_info = vk::DescriptorBufferInfo::builder()
                            .buffer(buffer.buf)
                            .offset(buffer.offset as u64)
                            .range(buffer.size as u64)
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
                    ShaderResource::Dynamic(alloc) => {
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
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::DynamicStorage(alloc) => {
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
                            .buffer_info(&buffer_infos[buffer_infos.len() - 1..])
                            .build();

                        write_descriptor_sets.push(write_descriptor_set);
                    }
                    ShaderResource::StorageBuffer(handle) => {
                        let buffer = self.buffers.get_ref(*handle).unwrap();

                        let buffer_info = vk::DescriptorBufferInfo::builder()
                            .buffer(buffer.buf)
                            .offset(buffer.offset as u64)
                            .range(buffer.size as u64)
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
                    ShaderResource::ConstBuffer(_) => {
                        return Err(GPUError::Unimplemented(
                            "Constant buffers are not supported",
                        ))
                    }
                }
            }
        }

        unsafe {
            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        Ok(())
    }

    /// Creates a bind group using indexed resources.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn make_indexed_bind_group(
        &mut self,
        info: &IndexedBindGroupInfo,
    ) -> Result<Handle<BindGroup>, GPUError> {
        // Retrieve the BindGroupLayout from the handle
        let layout = self.bind_group_layouts.get_ref(info.layout).unwrap();

        // Step 1: Allocate Descriptor Set
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(layout.pool)
            .set_layouts(&[layout.layout])
            .build();

        let descriptor_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? };

        let descriptor_set = descriptor_sets[0]; // We are allocating one descriptor set

        self.set_name(
            descriptor_set,
            info.debug_name,
            vk::ObjectType::DESCRIPTOR_SET,
        );

        // Step 4: Create the BindGroup and return a handle
        let bind_group = BindGroup {
            set: descriptor_set,
            set_id: info.set,
        };

        let bg = self.bind_groups.insert(bind_group).unwrap();
        self.update_bind_group(&BindGroupUpdateInfo {
            bg,
            bindings: info.bindings,
        })?;

        Ok(bg)
    }

    /// Creates a bind group from the provided bindings.
    ///
    /// # Prerequisites
    /// - Correct attachment formats.
    /// - Matching pipeline layouts.
    /// - Swapchain acquisition order is respected.
    /// - XR session state is valid.
    /// - Synchronization primitives are handled during presentation.
    pub fn make_bind_group(&mut self, info: &BindGroupInfo) -> Result<Handle<BindGroup>, GPUError> {
        // Retrieve the BindGroupLayout from the handle
        let layout = self.bind_group_layouts.get_ref(info.layout).unwrap();

        // Step 1: Allocate Descriptor Set
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(layout.pool)
            .set_layouts(&[layout.layout])
            .build();

        let descriptor_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info)? };

        let descriptor_set = descriptor_sets[0]; // We are allocating one descriptor set

        self.set_name(
            descriptor_set,
            info.debug_name,
            vk::ObjectType::DESCRIPTOR_SET,
        );

        // Step 2: Prepare the write operations for the descriptor set
        let mut write_descriptor_sets = Vec::new();
        let mut buffer_infos = Vec::new();
        let mut image_infos = Vec::new();

        for binding_info in info.bindings.iter() {
            match &binding_info.resource {
                ShaderResource::Buffer(handle) => {
                    let buffer = self.buffers.get_ref(*handle).unwrap();

                    let buffer_info = vk::DescriptorBufferInfo::builder()
                        .buffer(buffer.buf)
                        .offset(buffer.offset as u64)
                        .range(buffer.size as u64)
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
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER) // Assuming a sampled image
                        .image_info(&image_infos[image_infos.len() - 1..])
                        .build();

                    write_descriptor_sets.push(write_descriptor_set);
                }
                ShaderResource::Dynamic(alloc) => {
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
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC) // Assuming a uniform buffer for now
                        .buffer_info(&buffer_infos[buffer_infos.len() - 1..])
                        .build();

                    write_descriptor_sets.push(write_descriptor_set);
                }
                ShaderResource::DynamicStorage(alloc) => {
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
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER_DYNAMIC) // Assuming a uniform buffer for now
                        .buffer_info(&buffer_infos[buffer_infos.len() - 1..])
                        .build();

                    write_descriptor_sets.push(write_descriptor_set);
                }
                ShaderResource::StorageBuffer(buffer_handle) => {
                    let buffer = self.buffers.get_ref(*buffer_handle).unwrap();

                    let buffer_info = vk::DescriptorBufferInfo::builder()
                        .buffer(buffer.buf)
                        .offset(buffer.offset as u64)
                        .range(buffer.size as u64)
                        .build();

                    buffer_infos.push(buffer_info);

                    let write_descriptor_set = vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_set)
                        .dst_binding(binding_info.binding)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&buffer_infos[buffer_infos.len() - 1..])
                        .build();

                    write_descriptor_sets.push(write_descriptor_set);
                }
                ShaderResource::ConstBuffer(_) => {
                    return Err(GPUError::Unimplemented(
                        "Constant buffers are not supported",
                    ))
                }
            }
        }

        unsafe {
            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }

        // Step 4: Create the BindGroup and return a handle
        let bind_group = BindGroup {
            set: descriptor_set,
            set_id: info.set,
        };

        Ok(self.bind_groups.insert(bind_group).unwrap())
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
        };

        let table = self.bind_tables.insert(bind_table).unwrap();
        self.update_bind_table(&BindTableUpdateInfo {
            table,
            bindings: info.bindings,
        })?;

        Ok(table)
    }

    fn create_pipeline_layout(
        &self,
        bind_group_layout_handle: &[Option<Handle<BindGroupLayout>>],
        bind_table_layout_handle: &[Option<Handle<BindTableLayout>>],
    ) -> Result<vk::PipelineLayout> {
        let mut layouts = Vec::new();
        for (bg, bt) in bind_group_layout_handle
            .iter()
            .zip(bind_table_layout_handle.iter())
        {
            if let Some(b) = bg {
                layouts.push(self.bind_group_layouts.get_ref(*b).unwrap().layout);
            } else if let Some(t) = bt {
                layouts.push(self.bind_table_layouts.get_ref(*t).unwrap().layout);
            }
        }

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&layouts)
            .push_constant_ranges(&[]) // Add push constant ranges if needed
            .build();

        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None)? };

        Ok(pipeline_layout)
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
        for subpass in info.subpasses {
            let mut depth_stencil_attachment_ref = None;
            let attachment_offset = attachments.len();
            let color_offset = color_attachment_refs.len();

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

        let width = info.viewport.scissor.w;
        let height = info.viewport.scissor.h;

        let mut view_formats_vk = Vec::with_capacity(attachment_formats.len());
        let mut attachment_image_infos = Vec::with_capacity(attachment_formats.len());
        for fmt in &attachment_formats {
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

        let mut attachments_info =
            vk::FramebufferAttachmentsCreateInfo::builder().attachment_image_infos(&attachment_image_infos);

        let fb_info = vk::FramebufferCreateInfo::builder()
            .flags(vk::FramebufferCreateFlags::IMAGELESS_KHR)
            .render_pass(render_pass)
            .width(width)
            .height(height)
            .layers(1)
            .attachment_count(attachment_formats.len() as u32)
            .push_next(&mut attachments_info);

        let fb = unsafe { self.device.create_framebuffer(&fb_info, None) }?;

        return Ok(self
            .render_passes
            .insert(RenderPass {
                raw: render_pass,
                fb,
                viewport: info.viewport,
                attachment_formats,
            })
            .unwrap());
    }

    /// Creates a compute pipeline layout from shader and bind group layouts.
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

        let layout = self.create_pipeline_layout(&info.bg_layouts, &info.bt_layouts)?;

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

        // Step 6: Multisampling (we'll disable multisampling for now)
        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
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
        let layout = self.create_pipeline_layout(&info.bg_layouts, &info.bt_layouts)?;

        self.set_name(layout, info.debug_name, vk::ObjectType::PIPELINE_LAYOUT);

        let color_blends: Vec<vk::PipelineColorBlendAttachmentState> = info
            .details
            .color_blend_states
            .iter()
            .map(|c| c.clone().into())
            .collect();

        let dynamic_states: Vec<vk::DynamicState> = info
            .details
            .dynamic_states
            .iter()
            .map(|s| (*s).into())
            .collect();

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
        let rp_ref = self.render_passes.get_ref(info.render_pass).unwrap();
        let rp = rp_ref.raw;

        let rp_viewport = self
            .render_passes
            .get_ref(info.render_pass)
            .unwrap()
            .viewport
            .clone();

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&[layout.vertex_input])
            .vertex_attribute_descriptions(&layout.vertex_attribs)
            .build();

        // Step 4: Viewport and Scissor State
        let viewport = vk::Viewport::builder()
            .x(rp_viewport.area.x)
            .y(rp_viewport.area.y)
            .width(rp_viewport.area.w)
            .height(rp_viewport.area.h)
            .min_depth(0.0)
            .max_depth(1.0)
            .build();

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D {
                x: rp_viewport.scissor.x as i32,
                y: rp_viewport.scissor.y as i32,
            })
            .extent(vk::Extent2D {
                width: rp_viewport.scissor.w,
                height: rp_viewport.scissor.h,
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
            .render_pass(rp)
            .subpass(info.subpass_id as u32);

        if let Some(d) = layout.depth_stencil.as_ref() {
            pipeline_builder = pipeline_builder.depth_stencil_state(d);
        }

        if let Some(ref dyn_info) = dynamic_state_info {
            pipeline_builder = pipeline_builder.dynamic_state(dyn_info);
        }

        let pipeline_info = pipeline_builder.build();

        let graphics_pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()
        };

        self.set_name(
            graphics_pipelines[0],
            info.debug_name,
            vk::ObjectType::PIPELINE,
        );

        return Ok(self
            .gfx_pipelines
            .insert(GraphicsPipeline {
                render_pass: info.render_pass,
                raw: graphics_pipelines[0],
                layout: info.layout,
            })
            .unwrap());
    }
}
