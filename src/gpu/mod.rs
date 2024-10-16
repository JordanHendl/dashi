use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
mod error;
use crate::utils::{offset_alloc, Handle, Pool};
use ash::*;
pub use error::*;
use std::collections::HashMap;
use std::ffi::CString;
use vk_mem::Alloc;

pub mod structs;
pub use structs::*;
pub mod commands;
pub use commands::*;

// Convert Filter enum to VkFilter
impl From<Filter> for vk::Filter {
    fn from(filter: Filter) -> Self {
        match filter {
            Filter::Nearest => vk::Filter::NEAREST,
            Filter::Linear => vk::Filter::LINEAR,
        }
    }
}

// Convert SamplerAddressMode enum to VkSamplerAddressMode
impl From<SamplerAddressMode> for vk::SamplerAddressMode {
    fn from(address_mode: SamplerAddressMode) -> Self {
        match address_mode {
            SamplerAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            SamplerAddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            SamplerAddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            SamplerAddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        }
    }
}

// Convert SamplerMipmapMode enum to VkSamplerMipmapMode
impl From<SamplerMipmapMode> for vk::SamplerMipmapMode {
    fn from(mipmap_mode: SamplerMipmapMode) -> Self {
        match mipmap_mode {
            SamplerMipmapMode::Nearest => vk::SamplerMipmapMode::NEAREST,
            SamplerMipmapMode::Linear => vk::SamplerMipmapMode::LINEAR,
        }
    }
}

// Convert BorderColor enum to VkBorderColor
impl From<BorderColor> for vk::BorderColor {
    fn from(border_color: BorderColor) -> Self {
        match border_color {
            BorderColor::OpaqueBlack => vk::BorderColor::INT_OPAQUE_BLACK,
            BorderColor::OpaqueWhite => vk::BorderColor::INT_OPAQUE_WHITE,
            BorderColor::TransparentBlack => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
        }
    }
}

// Function to convert SamplerCreateInfo to VkSamplerCreateInfo
impl From<SamplerInfo> for vk::SamplerCreateInfo {
    fn from(info: SamplerInfo) -> Self {
        vk::SamplerCreateInfo {
            mag_filter: info.mag_filter.into(),
            min_filter: info.min_filter.into(),
            address_mode_u: info.address_mode_u.into(),
            address_mode_v: info.address_mode_v.into(),
            address_mode_w: info.address_mode_w.into(),
            anisotropy_enable: if info.anisotropy_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
            max_anisotropy: info.max_anisotropy,
            border_color: info.border_color.into(),
            unnormalized_coordinates: if info.unnormalized_coordinates {
                vk::TRUE
            } else {
                vk::FALSE
            },
            compare_enable: if info.compare_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
            mipmap_mode: info.mipmap_mode.into(),
            ..Default::default() // Other default Vulkan fields can be set here as needed
        }
    }
}

pub(super) fn convert_rect2d_to_vulkan(rect: Rect2D) -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D {
            x: rect.x as i32, // Vulkan uses signed integers for the offset
            y: rect.y as i32,
        },
        extent: vk::Extent2D {
            width: rect.w,
            height: rect.h,
        },
    }
}

pub(super) fn convert_barrier_point_vk(pt: BarrierPoint) -> vk::PipelineStageFlags {
    match pt {
        BarrierPoint::DrawEnd => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        BarrierPoint::BlitRead => vk::PipelineStageFlags::TRANSFER,
        BarrierPoint::BlitWrite => vk::PipelineStageFlags::TRANSFER,
        BarrierPoint::Present => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
    }
}

fn lib_to_vk_image_format(fmt: &Format) -> vk::Format {
    match fmt {
        Format::RGB8 => return vk::Format::R8G8B8_SRGB,
        Format::RGBA32F => return vk::Format::R32G32B32A32_SFLOAT,
        Format::RGBA8 => return vk::Format::R8G8B8A8_SRGB,
        Format::BGRA8 => return vk::Format::B8G8R8A8_SRGB,
        Format::BGRA8Unorm => return vk::Format::B8G8R8A8_SNORM,
        Format::D24S8 => vk::Format::D24_UNORM_S8_UINT,
    }
}

fn vk_to_lib_image_format(fmt: vk::Format) -> Format {
    match fmt {
        vk::Format::R8G8B8_SRGB => return Format::RGB8,
        vk::Format::R32G32B32A32_SFLOAT => return Format::RGBA32F,
        vk::Format::R8G8B8A8_SRGB => return Format::RGBA8,
        vk::Format::B8G8R8A8_SRGB => return Format::BGRA8,
        vk::Format::B8G8R8A8_SNORM => return Format::BGRA8Unorm,
        _ => todo!(),
    }
}

fn channel_count(fmt: &Format) -> u32 {
    match fmt {
        Format::RGB8 => 3,
        Format::BGRA8 | Format::BGRA8Unorm | Format::RGBA8 | Format::RGBA32F => 4,
        Format::D24S8 => 4,
    }
}

fn bytes_per_channel(fmt: &Format) -> u32 {
    match fmt {
        Format::RGB8 | Format::BGRA8 | Format::BGRA8Unorm | Format::RGBA8 => 1,
        Format::RGBA32F => 4,
        Format::D24S8 => 3,
    }
}

fn convert_load_op(load_op: LoadOp) -> vk::AttachmentLoadOp {
    match load_op {
        LoadOp::Load => vk::AttachmentLoadOp::LOAD,
        LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
        LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
    }
}

fn convert_store_op(store_op: StoreOp) -> vk::AttachmentStoreOp {
    match store_op {
        StoreOp::Store => vk::AttachmentStoreOp::STORE,
        StoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
    }
}

fn convert_sample_count(sample_count: SampleCount) -> vk::SampleCountFlags {
    match sample_count {
        SampleCount::S1 => vk::SampleCountFlags::TYPE_1,
        SampleCount::S2 => vk::SampleCountFlags::TYPE_2,
    }
}

#[derive(Clone)]
pub(super) struct CommandBufferQueue {
    queue: VecDeque<(
        vk::CommandBuffer,
        vk::Fence,
        vk::Semaphore,
        Arc<Mutex<bool>>,
    )>,
}

#[derive(Debug)]
pub struct Buffer {
    buf: vk::Buffer,
    alloc: vk_mem::Allocation,
    size: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct DynamicBuffer {
    pub(crate) handle: Handle<Buffer>,
    pub(crate) alloc: offset_alloc::Allocation,
    pub(crate) ptr: *mut u8,
    pub(crate) size: u16,
}

impl DynamicBuffer {
    pub fn slice<T>(&mut self) -> &mut [T] {
        let typed_map: *mut T = unsafe { std::mem::transmute(self.ptr) };
        return unsafe {
            std::slice::from_raw_parts_mut(typed_map, self.size as usize / std::mem::size_of::<T>())
        };
    }
}

pub struct DynamicAllocator {
    allocator: offset_alloc::Allocator,
    ptr: *mut u8,
    min_alloc_size: u32,
    pool: Handle<Buffer>,
}

impl DynamicAllocator {
    pub fn reset(&mut self) {
        self.allocator.reset();
    }

    pub fn bump(&mut self) -> Option<DynamicBuffer> {
        let alloc = self.allocator.allocate(self.min_alloc_size)?;
        return Some(DynamicBuffer {
            handle: self.pool,
            alloc,
            ptr: unsafe { self.ptr.offset(alloc.offset as isize) },
            size: self.min_alloc_size as u16,
        });
    }
}
#[derive(Debug)]
pub struct Image {
    img: vk::Image,
    alloc: vk_mem::Allocation,
    dim: [u32; 3],
    format: Format,
    layout: vk::ImageLayout,
    sub_layers: vk::ImageSubresourceLayers,
    extent: vk::Extent3D,
}

#[derive(Debug)]
pub struct Sampler {
    sampler: vk::Sampler,
}

#[derive(Debug)]
pub struct ImageView {
    img: Handle<Image>,
    range: vk::ImageSubresourceRange,
    view: vk::ImageView,
}

#[derive(Clone, Default)]
pub struct RenderPass {
    pub(super) raw: vk::RenderPass,
    pub(super) viewport: Viewport,
    pub(super) fb: Vec<(vk::Framebuffer, Vec<Handle<Image>>)>,
    pub(super) clear_values: Vec<vk::ClearValue>,
}

#[derive(Debug)]
pub struct BindGroupLayout {
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    variables: Vec<BindGroupVariable>,
}

pub struct BindGroup {
    set: vk::DescriptorSet,
    set_id: u32,
}

pub struct Display {
    window: std::cell::Cell<sdl2::video::Window>,
    swapchain: ash::vk::SwapchainKHR,
    surface: ash::vk::SurfaceKHR,
    images: Vec<Handle<Image>>,
    views: Vec<Handle<ImageView>>,
    loader: ash::extensions::khr::Surface,
    sc_loader: ash::extensions::khr::Swapchain,
    semaphores: Vec<Semaphore>,
    fences: Vec<Fence>,
    frame_idx: u32,
}

#[derive(Clone)]
pub struct Fence {
    raw: vk::Fence,
    device: ash::Device,
}

impl Default for Fence {
    fn default() -> Self {
        Self {
            raw: Default::default(),
            device: unsafe { std::mem::MaybeUninit::zeroed().assume_init() },
        }
    }
}

impl Fence {
    fn new(raw: vk::Fence, device: ash::Device) -> Self {
        Self { raw, device }
    }

    pub fn wait(&mut self) -> Result<(), GPUError> {
        let _res = unsafe {
            self.device
                .wait_for_fences(&[self.raw], true, std::u64::MAX)
        }?;

        Ok(())
    }

    fn reset(&mut self) -> Result<(), GPUError> {
        unsafe { self.device.reset_fences(&[self.raw]) }?;
        Ok(())
    }
}

#[derive(Copy, Clone, Default)]
pub struct Semaphore {
    raw: vk::Semaphore,
}

#[derive(Clone, Default, Debug)]
pub struct GraphicsPipelineLayout {
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    depth_stencil: Option<vk::PipelineDepthStencilStateCreateInfo>,
    multisample: vk::PipelineMultisampleStateCreateInfo,
    rasterizer: vk::PipelineRasterizationStateCreateInfo,
    vertex_input: vk::VertexInputBindingDescription,
    vertex_attribs: Vec<vk::VertexInputAttributeDescription>,
    layout: vk::PipelineLayout,
}

#[derive(Clone, Default)]
pub struct GraphicsPipeline {
    raw: vk::Pipeline,
    render_pass: Handle<RenderPass>,
    layout: Handle<GraphicsPipelineLayout>,
}

pub struct CommandList {
    cmd_buf: vk::CommandBuffer,
    fence: Fence,
    semaphore: vk::Semaphore,
    dirty: bool,
    ctx: *mut Context,
    curr_rp: Option<Handle<RenderPass>>,
    curr_pipeline: Option<Handle<GraphicsPipeline>>,
    last_op_access: vk::AccessFlags,
    last_op_stage: vk::PipelineStageFlags,
}

impl Default for CommandList {
    fn default() -> Self {
        Self {
            cmd_buf: Default::default(),
            fence: Default::default(),
            semaphore: Default::default(),
            dirty: false,
            ctx: std::ptr::null_mut(),
            curr_rp: None,
            curr_pipeline: None,
            last_op_access: vk::AccessFlags::TRANSFER_READ,
            last_op_stage: vk::PipelineStageFlags::ALL_COMMANDS,
        }
    }
}

#[derive(Default)]
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
    pub(super) pool: vk::CommandPool,
    pub(super) allocator: vk_mem::Allocator,
    pub(super) gfx_queue: Queue,
    pub(super) buffers: Pool<Buffer>,
    pub(super) render_passes: Pool<RenderPass>,
    pub(super) images: Pool<Image>,
    pub(super) image_views: Pool<ImageView>,
    pub(super) samplers: Pool<Sampler>,
    pub(super) bind_group_layouts: Pool<BindGroupLayout>,
    pub(super) bind_groups: Pool<BindGroup>,
    pub(super) gfx_pipeline_layouts: Pool<GraphicsPipelineLayout>,
    pub(super) gfx_pipelines: Pool<GraphicsPipeline>,
    pub(super) rps: HashMap<u64, Handle<RenderPass>>,
    pub(super) cmds_to_release: Vec<(CommandList, Fence)>,

    #[cfg(feature = "dashi-sdl2")]
    pub(super) sdl_context: sdl2::Sdl,
    #[cfg(feature = "dashi-sdl2")]
    pub(super) sdl_video: sdl2::VideoSubsystem,
    pub(super) cmd_destroy_queue: Vec<(vk::CommandBuffer, Arc<Mutex<bool>>, vk::Semaphore)>,
    #[cfg(debug_assertions)]
    pub(super) debug_utils: ash::extensions::ext::DebugUtils,
}

impl Context {
    pub fn new(_info: &ContextInfo) -> Result<Self, GPUError> {
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 2, 0),
            ..Default::default()
        };

        // Create instance
        let entry = unsafe { Entry::load() }?;
        let requested_instance_extensions = [
            #[cfg(debug_assertions)]
            ash::extensions::ext::DebugUtils::name().as_ptr(),
            ash::extensions::khr::Surface::name().as_ptr(),
            #[cfg(target_os = "linux")]
            ash::extensions::khr::XlibSurface::name().as_ptr(),
        ];

        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::builder()
                    .application_info(&app_info)
                    .enabled_extension_names(&requested_instance_extensions)
                    .build(),
                None,
            )
        }?;

        let pdevice = unsafe { instance.enumerate_physical_devices()?[0] };
        let device_prop = unsafe { instance.get_physical_device_properties(pdevice) };

        let queue_prop = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
        let features = unsafe { instance.get_physical_device_features(pdevice) };

        let mut queue_family: u32 = 0;

        let mut gfx_queue = Queue::default();
        for prop in queue_prop.iter() {
            if prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                gfx_queue.family = queue_family;
            }
            queue_family += 1;
        }

        let priorities = [1.0];
        let device = unsafe {
            instance.create_device(
                pdevice,
                &vk::DeviceCreateInfo::builder()
                    .enabled_features(
                        &vk::PhysicalDeviceFeatures::builder()
                            .shader_clip_distance(true)
                            .build(),
                    )
                    .enabled_extension_names(
                        [
                            ash::extensions::khr::Swapchain::name().as_ptr(),
                            #[cfg(any(target_os = "macos", target_os = "ios"))]
                            KhrPortabilitySubsetFn::name().as_ptr(),
                        ]
                        .as_ref(),
                    )
                    .queue_create_infos(&[vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(gfx_queue.family)
                        .queue_priorities(&priorities)
                        .build()])
                    .build(),
                None,
            )
        }?;

        gfx_queue.queue = unsafe { device.get_device_queue(gfx_queue.family, 0) };

        let allocator = vk_mem::Allocator::new(vk_mem::AllocatorCreateInfo::new(
            &instance, &device, pdevice,
        ))?;

        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(gfx_queue.family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .build(),
                None,
            )
        }?;

        #[cfg(debug_assertions)]
        let debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);

        let sdl_context = sdl2::init().unwrap();
        let sdl_video = sdl_context.video().unwrap();
        return Ok(Context {
            entry,
            instance,
            pdevice,
            device,
            pool,
            allocator,
            gfx_queue,
            properties: device_prop,

            #[cfg(feature = "dashi-sdl2")]
            sdl_context,
            #[cfg(feature = "dashi-sdl2")]
            sdl_video,

            cmds_to_release: Default::default(),
            buffers: Default::default(),
            render_passes: Default::default(),
            images: Default::default(),
            image_views: Default::default(),
            bind_group_layouts: Default::default(),
            bind_groups: Default::default(),
            gfx_pipeline_layouts: Default::default(),
            gfx_pipelines: Default::default(),
            rps: Default::default(),
            cmd_destroy_queue: Default::default(),
            samplers: Default::default(),

            #[cfg(debug_assertions)]
            debug_utils,
        });
    }

    #[cfg(feature = "dashi-sdl2")]
    pub fn get_sdl_event(&mut self) -> sdl2::EventPump {
        return self.sdl_context.event_pump().unwrap();
    }

    fn set_name<T>(&self, obj: T, name: &str, t: vk::ObjectType)
    where
        T: ash::vk::Handle,
    {
        #[cfg(debug_assertions)]
        {
            unsafe {
                let name: CString = CString::new(name.to_string()).unwrap();
                self.debug_utils
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

    pub fn begin_command_list(&mut self, info: &CommandListInfo) -> Result<CommandList, GPUError> {
        let cmd = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(self.pool)
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .build(),
            )
        }?;

        self.set_name(cmd[0], info.debug_name, vk::ObjectType::COMMAND_BUFFER);

        let f = unsafe {
            self.device.create_fence(
                &vk::FenceCreateInfo::builder()
                    .flags(vk::FenceCreateFlags::SIGNALED)
                    .build(),
                None,
            )
        }?;

        self.set_name(
            f,
            format!("{}.fence", info.debug_name).as_str(),
            vk::ObjectType::FENCE,
        );

        let s = unsafe {
            self.device
                .create_semaphore(&vk::SemaphoreCreateInfo::builder().build(), None)
        }?;

        self.set_name(
            s,
            format!("{}.semaphore", info.debug_name).as_str(),
            vk::ObjectType::SEMAPHORE,
        );

        unsafe {
            self.device
                .begin_command_buffer(cmd[0], &vk::CommandBufferBeginInfo::builder().build())?
        };

        return Ok(CommandList {
            cmd_buf: cmd[0],
            fence: Fence::new(f, self.device.clone()),
            semaphore: s,
            dirty: true,
            ctx: self,
            ..Default::default()
        });
    }
    fn oneshot_transition_image(&mut self, img: Handle<ImageView>, layout: vk::ImageLayout) {
        let mut list = self
            .begin_command_list(&CommandListInfo {
                debug_name: "",
                ..Default::default()
            })
            .unwrap();
        self.transition_image(list.cmd_buf, img, layout);
        let (sem, mut fence) = self.submit(&mut list, None).unwrap();
        fence.wait().unwrap();
    }

    fn oneshot_transition_image_noview(&mut self, img: Handle<Image>, layout: vk::ImageLayout) {
        let tmp_view = self
            .make_image_view(&ImageViewInfo {
                debug_name: "oneshot_view",
                img: img,
                layer: 0,
                mip_level: 0,
            })
            .unwrap();

        let mut list = self
            .begin_command_list(&CommandListInfo {
                debug_name: "oneshot_transition",
                ..Default::default()
            })
            .unwrap();
        self.transition_image(list.cmd_buf, tmp_view, layout);
        let (sem, fence) = self.submit(&mut list, None).unwrap();
    }

    fn transition_image_stages(
        &mut self,
        cmd: vk::CommandBuffer,
        img: Handle<ImageView>,
        layout: vk::ImageLayout,
        src: vk::PipelineStageFlags,
        dst: vk::PipelineStageFlags,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
    ) {
        let view = self.image_views.get_mut_ref(img).unwrap();
        let img = self.images.get_mut_ref(view.img).unwrap();
        let old_layout = img.layout;
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

        img.layout = layout;
    }

    fn transition_image(
        &mut self,
        cmd: vk::CommandBuffer,
        img: Handle<ImageView>,
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

    pub fn submit(
        &mut self,
        cmd: &mut CommandList,
        wait_sems: Option<&[Semaphore]>,
    ) -> Result<(Semaphore, Fence), GPUError> {
        if cmd.dirty {
            unsafe { self.device.end_command_buffer(cmd.cmd_buf)? };
            cmd.dirty = false;
        }

        cmd.fence.wait()?;
        cmd.fence.reset()?;

        let mut raw_wait_sems: Vec<vk::Semaphore> = Vec::with_capacity(32);
        match wait_sems {
            Some(sems) => {
                for sem in sems {
                    raw_wait_sems.push(sem.raw)
                }
            }
            None => {}
        }

        let stage_masks = &[vk::PipelineStageFlags::ALL_COMMANDS];
        unsafe {
            self.device.queue_submit(
                self.gfx_queue.queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[cmd.cmd_buf])
                    .signal_semaphores(&[cmd.semaphore])
                    .wait_dst_stage_mask(stage_masks)
                    .wait_semaphores(&raw_wait_sems)
                    .build()],
                cmd.fence.raw,
            )?
        };

        for (cmd, fence) in &mut self.cmds_to_release {
            fence.wait()?;
            unsafe {
                self.device.destroy_fence(fence.raw, None);
                self.device.destroy_semaphore(cmd.semaphore, None);
                self.device.free_command_buffers(self.pool, &[cmd.cmd_buf]);
            }
        }

        self.cmds_to_release.clear();

        return Ok((Semaphore { raw: cmd.semaphore }, cmd.fence.clone()));
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

    pub fn make_image_view(&mut self, info: &ImageViewInfo) -> Result<Handle<ImageView>, GPUError> {
        let img = self.images.get_ref(info.img).unwrap();
        let sub_range = vk::ImageSubresourceRange::builder()
            .base_array_layer(0)
            .layer_count(1)
            .base_mip_level(info.mip_level)
            .level_count(1)
            .aspect_mask(img.sub_layers.aspect_mask)
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

        self.set_name(view, info.debug_name, vk::ObjectType::IMAGE_VIEW);

        match self.image_views.insert(ImageView {
            view,
            range: sub_range,
            img: info.img,
        }) {
            Some(h) => return Ok(h),
            None => return Err(GPUError::SlotError()),
        }
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
                    .array_layers(info.dim[2] as u32)
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
            layout: vk::ImageLayout::UNDEFINED,
            sub_layers: vk::ImageSubresourceLayers::builder()
                .layer_count(info.dim[2] as u32)
                .mip_level(0)
                .base_array_layer(0)
                .aspect_mask(aspect)
                .build(),
            extent: vk::Extent3D::builder()
                .width(info.dim[0] as u32)
                .height(info.dim[1] as u32)
                .depth(info.dim[2] as u32)
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

    pub fn map_buffer_mut<T>(&mut self, buf: Handle<Buffer>) -> Result<&mut [T], GPUError> {
        let buf = match self.buffers.get_ref(buf) {
            Some(it) => it,
            None => return Err(GPUError::SlotError()),
        };

        let mut alloc: vk_mem::Allocation = unsafe { std::mem::transmute_copy(&buf.alloc) };
        let mapped = unsafe { self.allocator.map_memory(&mut alloc) }?;
        let typed_map: *mut T = unsafe { std::mem::transmute(mapped) };
        return Ok(unsafe {
            std::slice::from_raw_parts_mut(typed_map, buf.size as usize / std::mem::size_of::<T>())
        });
    }

    pub fn map_buffer<T>(&mut self, buf: Handle<Buffer>) -> Result<&[T], GPUError> {
        let buf = match self.buffers.get_ref(buf) {
            Some(it) => it,
            None => return Err(GPUError::SlotError()),
        };

        let mut alloc: vk_mem::Allocation = unsafe { std::mem::transmute_copy(&buf.alloc) };
        let mapped = unsafe { self.allocator.map_memory(&mut alloc) }?;
        let typed_map: *mut T = unsafe { std::mem::transmute(mapped) };
        return Ok(unsafe {
            std::slice::from_raw_parts(typed_map, buf.size as usize / std::mem::size_of::<T>())
        });
    }

    pub fn unmap_buffer(&self, buf: Handle<Buffer>) -> Result<(), GPUError> {
        let buf = match self.buffers.get_ref(buf) {
            Some(it) => it,
            None => return Err(GPUError::SlotError()),
        };

        let mut alloc: vk_mem::Allocation = unsafe { std::mem::transmute_copy(&buf.alloc) };
        unsafe { self.allocator.unmap_memory(&mut alloc) };

        return Ok(());
    }
    fn init_buffer(&mut self, buf: Handle<Buffer>, info: &BufferInfo) -> Result<(), GPUError> {
        if info.initial_data.is_none() {
            return Ok(());
        }

        match info.visibility {
            MemoryVisibility::Gpu => {
                let mut new_info = info.clone();
                new_info.visibility = MemoryVisibility::CpuAndGpu;

                let staging = self.make_buffer(&new_info)?;
                let mut list = self.begin_command_list(&Default::default())?;
                list.append(Command::BufferCopyCommand(BufferCopy {
                    src: staging,
                    dst: buf,
                    src_offset: 0,
                    dst_offset: 0,
                    size: unsafe { info.initial_data.unwrap_unchecked().len() },
                }));

                let (_sem, mut fence) = self.submit(&mut list, None)?;
                fence.wait()?;
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

    fn init_image(&mut self, image: Handle<Image>, info: &ImageInfo) -> Result<(), GPUError> {
        let img_ref = self.images.get_ref(image).unwrap();
        let tmp_view = self.make_image_view(&ImageViewInfo {
            debug_name: "view",
            img: image,
            layer: 0,
            mip_level: 0,
        })?;

        let mut list = self.begin_command_list(&Default::default())?;
        if info.initial_data.is_none() {
            self.transition_image(list.cmd_buf, tmp_view, vk::ImageLayout::GENERAL);
            let (_sem, mut fence) = self.submit(&mut list, None)?;
            fence.wait()?;
            self.destroy_image_view(tmp_view);
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

        let mut list = self.begin_command_list(&Default::default())?;
        list.append(Command::BufferImageCopyCommand(BufferImageCopy {
            src: staging,
            dst: tmp_view,
            src_offset: 0,
        }));

        let (_sem, mut fence) = self.submit(&mut list, None)?;
        fence.wait()?;
        self.destroy_buffer(staging);
        self.destroy_image_view(tmp_view);
        return Ok(());
    }

    pub fn make_dynamic_allocator(
        &mut self,
        info: &DynamicAllocatorInfo,
    ) -> Result<DynamicAllocator, GPUError> {
        let buffer = self.make_buffer(&BufferInfo {
            debug_name: info.debug_name,
            byte_size: info.byte_size,
            visibility: MemoryVisibility::CpuAndGpu,
            ..Default::default()
        })?;

        return Ok(DynamicAllocator {
            allocator: offset_alloc::Allocator::new(
                info.byte_size,
                self.properties.limits.min_uniform_buffer_offset_alignment as u32,
            ),
            pool: buffer,
            ptr: self.map_buffer_mut(buffer)?.as_mut_ptr(),
            min_alloc_size: self.properties.limits.min_uniform_buffer_offset_alignment as u32,
        });
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
            }) {
                Some(handle) => {
                    self.init_buffer(handle, info)?;
                    return Ok(handle);
                }
                None => return Err(GPUError::SlotError()),
            }
        }
    }

    pub fn destroy_buffer(&mut self, handle: Handle<Buffer>) {
        let buf = self.buffers.get_mut_ref(handle).unwrap();
        unsafe { self.allocator.destroy_buffer(buf.buf, &mut buf.alloc) };
        self.buffers.release(handle);
    }

    pub fn destroy_image_view(&mut self, handle: Handle<ImageView>) {
        let img = self.image_views.get_mut_ref(handle).unwrap();
        unsafe { self.device.destroy_image_view(img.view, None) };
    }
    pub fn destroy_image(&mut self, handle: Handle<Image>) {
        let img = self.images.get_mut_ref(handle).unwrap();
        unsafe { self.allocator.destroy_image(img.img, &mut img.alloc) };
        self.images.release(handle);
    }

    pub fn make_bind_group_layout(
        &mut self,
        info: &BindGroupLayoutInfo,
    ) -> Result<Handle<BindGroupLayout>, GPUError> {
        const MAX_DESCRIPTOR_SETS: u32 = 2048;

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
                };

                let stage_flags = match shader_info.shader_type {
                    ShaderType::Vertex => vk::ShaderStageFlags::VERTEX,
                    ShaderType::Fragment => vk::ShaderStageFlags::FRAGMENT,
                    ShaderType::Compute => vk::ShaderStageFlags::COMPUTE,
                };

                let layout_binding = vk::DescriptorSetLayoutBinding::builder()
                    .binding(variable.binding)
                    .descriptor_type(descriptor_type)
                    .descriptor_count(1) // Assuming one per binding
                    .stage_flags(stage_flags)
                    .build();

                bindings.push(layout_binding);
            }
        }

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
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
            .max_sets(MAX_DESCRIPTOR_SETS);

        let descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None)? };

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

        // Step 2: Prepare the write operations for the descriptor set
        let mut write_descriptor_sets = Vec::new();
        let mut buffer_infos = Vec::new();
        let mut image_infos = Vec::new();

        for binding_info in info.bindings.iter() {
            match &binding_info.resource {
                ShaderResource::Buffer(buffer_handle) => {
                    let buffer = self.buffers.get_ref(*buffer_handle).unwrap();

                    let buffer_info = vk::DescriptorBufferInfo::builder()
                        .buffer(buffer.buf)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
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
                ShaderResource::SampledImage(image_handle, sampler) => {
                    let image = self.image_views.get_ref(*image_handle).unwrap();
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
                        .range(vk::WHOLE_SIZE)
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

    fn create_pipeline_layout(
        &self,
        bind_group_layout_handle: Handle<BindGroupLayout>,
    ) -> Result<vk::PipelineLayout, GPUError> {
        let bg = self
            .bind_group_layouts
            .get_ref(bind_group_layout_handle)
            .unwrap();
        let bind_group_layouts = vec![bg.layout]; // Assuming value is the Vulkan handle

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&bind_group_layouts)
            .push_constant_ranges(&[]) // Add push constant ranges if needed
            .build();

        let pipeline_layout = unsafe { self.device.create_pipeline_layout(&layout_info, None)? };

        Ok(pipeline_layout)
    }

    fn create_shader_module(&self, spirv_code: &[u32]) -> Result<vk::ShaderModule, GPUError> {
        // Step 1: Create Shader Module Info
        let create_info = vk::ShaderModuleCreateInfo::builder()
            .code(spirv_code) // The SPIR-V bytecode
            .build();

        // Step 2: Create Shader Module using Vulkan
        let shader_module = unsafe { self.device.create_shader_module(&create_info, None)? };

        // Step 3: Return the shader module
        Ok(shader_module)
    }

    pub fn make_render_pass(
        &mut self,
        info: &RenderPassInfo,
    ) -> Result<Handle<RenderPass>, GPUError> {
        let mut attachments = Vec::new();
        let mut color_attachment_refs = Vec::new();
        let mut depth_stencil_attachment_ref = None;
        let mut clear_values = Vec::new();
        for (index, color_attachment) in info.color_attachments.iter().enumerate() {
            let view = self.image_views.get_ref(color_attachment.view).unwrap();
            let img = self.images.get_ref(view.img).unwrap();
            let attachment_desc = vk::AttachmentDescription {
                format: lib_to_vk_image_format(&img.format),
                samples: convert_sample_count(color_attachment.samples),
                load_op: convert_load_op(color_attachment.load_op),
                store_op: convert_store_op(color_attachment.store_op),
                stencil_load_op: convert_load_op(color_attachment.stencil_load_op),
                stencil_store_op: convert_store_op(color_attachment.stencil_store_op),
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ..Default::default()
            };
            attachments.push(attachment_desc);

            let c = color_attachment.clear_color;
            clear_values.push(vk::ClearValue {
                color: vk::ClearColorValue { float32: c },
            });

            let attachment_ref = vk::AttachmentReference {
                attachment: index as u32,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };
            color_attachment_refs.push(attachment_ref);
        }

        // Process depth-stencil attachment
        if let Some(depth_stencil_attachment) = info.depth_stencil_attachment {
            let view = self
                .image_views
                .get_ref(depth_stencil_attachment.view)
                .unwrap();
            let img = self.images.get_ref(view.img).unwrap();
            let depth_attachment_desc = vk::AttachmentDescription {
                format: lib_to_vk_image_format(&img.format),
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
            let c = depth_stencil_attachment.clear_color;
            clear_values.push(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: c[0],
                    stencil: c[1] as u32,
                },
            });
            depth_stencil_attachment_ref = Some(vk::AttachmentReference {
                attachment: (attachments.len() - 1) as u32,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            });
        }

        // Create subpass description
        let subpass_desc = match depth_stencil_attachment_ref.as_ref() {
            Some(d) => vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_attachment_refs)
                .depth_stencil_attachment(d)
                .build(),
            None => vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_attachment_refs)
                .build(),
        };

        // Create the render pass info
        let render_pass_info = vk::RenderPassCreateInfo {
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: 1,
            p_subpasses: &subpass_desc,
            ..Default::default()
        };

        // Create render pass
        let render_pass = unsafe { self.device.create_render_pass(&render_pass_info, None) }?;

        let fbs = self.create_framebuffers(render_pass, info)?;
        return Ok(self
            .render_passes
            .insert(RenderPass {
                raw: render_pass,
                fb: fbs,
                clear_values,
                viewport: info.viewport,
            })
            .unwrap());
    }

    pub fn make_graphics_pipeline_layout(
        &mut self,
        info: &GraphicsPipelineLayoutInfo,
    ) -> Result<Handle<GraphicsPipelineLayout>, GPUError> {
        let shader_stages: Vec<vk::PipelineShaderStageCreateInfo> = info
            .shaders
            .iter()
            .map(|shader_info| {
                let stage_flags = match shader_info.stage {
                    ShaderType::Vertex => vk::ShaderStageFlags::VERTEX,
                    ShaderType::Fragment => vk::ShaderStageFlags::FRAGMENT,
                    _ => todo!(),
                };

                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(stage_flags)
                    .module(unsafe { self.create_shader_module(shader_info.spirv).unwrap() })
                    .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap()) // Entry point is usually "main"
                    //                    .specialization_info(None) // Handle specialization constants if needed
                    .build()
            })
            .collect();

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
                        ShaderPrimitiveType::Vec2 => vk::Format::R32G32_SFLOAT,
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
            })
            .front_face(match info.details.front_face {
                VertexOrdering::Clockwise => vk::FrontFace::CLOCKWISE,
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
        let depth_stencil_state = if info.details.depth_test {
            Some(
                vk::PipelineDepthStencilStateCreateInfo::builder()
                    .depth_test_enable(true)
                    .depth_write_enable(true)
                    .min_depth_bounds(0.0)
                    .max_depth_bounds(1.0)
                    .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                    .build(),
            )
        } else {
            None
        };

        // Step 9: Create Pipeline Layout (assume we have a layout creation function)
        let layout = self.create_pipeline_layout(info.bg_layout)?;

        return Ok(self
            .gfx_pipeline_layouts
            .insert(GraphicsPipelineLayout {
                layout,
                shader_stages: shader_stages,
                vertex_input: vertex_binding_description,
                vertex_attribs: vertex_attribute_descriptions,
                rasterizer: rasterizer,
                multisample: multisampling,
                depth_stencil: depth_stencil_state,
                input_assembly: input_assembly,
            })
            .unwrap());
    }

    pub fn release_list_on_next_submit(&mut self, fence: Fence, list: CommandList) {
        self.cmds_to_release.push((list, fence));
    }

    pub fn make_graphics_pipeline(
        &mut self,
        info: &GraphicsPipelineInfo,
    ) -> Result<Handle<GraphicsPipeline>, GPUError> {
        let layout = self.gfx_pipeline_layouts.get_ref(info.layout).unwrap();
        let rp = self.render_passes.get_ref(info.render_pass).unwrap().raw;
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

        // Step 8: Color Blend State
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false)
            .build();

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .attachments(&[color_blend_attachment])
            .build();

        // Step 10: Create Graphics Pipeline
        let pipeline_info = match layout.depth_stencil.as_ref() {
            Some(d) => vk::GraphicsPipelineCreateInfo::builder()
                .stages(&layout.shader_stages)
                .vertex_input_state(&vertex_input_info)
                .input_assembly_state(&layout.input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&layout.rasterizer)
                .multisample_state(&layout.multisample)
                .depth_stencil_state(d)
                .color_blend_state(&color_blend_state)
                .layout(layout.layout)
                .render_pass(rp)
                .subpass(0)
                .build(),
            None => vk::GraphicsPipelineCreateInfo::builder()
                .stages(&layout.shader_stages)
                .vertex_input_state(&vertex_input_info)
                .input_assembly_state(&layout.input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&layout.rasterizer)
                .multisample_state(&layout.multisample)
                .color_blend_state(&color_blend_state)
                .layout(layout.layout)
                .render_pass(rp)
                .subpass(0)
                .build(),
        };

        let graphics_pipelines = unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .unwrap()
        };

        return Ok(self
            .gfx_pipelines
            .insert(GraphicsPipeline {
                render_pass: info.render_pass,
                raw: graphics_pipelines[0],
                layout: info.layout,
            })
            .unwrap());
    }

    pub(super) fn create_framebuffers(
        &mut self,
        render_pass: vk::RenderPass,
        render_pass_begin: &RenderPassInfo,
    ) -> Result<Vec<(vk::Framebuffer, Vec<Handle<Image>>)>, GPUError> {
        let mut framebuffers_with_images = Vec::new();
        let mut width = std::u32::MAX;
        let mut height = std::u32::MAX;
        // Loop through each subpass and create a framebuffer
        let mut attachments = Vec::new();
        let mut created_images = Vec::new();

        // Collect the image views for color attachments
        for color_attachment in render_pass_begin.color_attachments.iter() {
            let created_image_handle = color_attachment.view;
            self.oneshot_transition_image(
                color_attachment.view,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
            let view = self.image_views.get_ref(created_image_handle).unwrap();
            let image = self.images.get_ref(view.img).unwrap();
            let color_view = view.view;

            width = std::cmp::min(image.dim[0], width);
            height = std::cmp::min(image.dim[1], height);
            attachments.push(color_view);
            created_images.push(view.img);
        }

        // Collect the image view for depth/stencil attachment if present
        if let Some(depth_stencil_attachment) = render_pass_begin.depth_stencil_attachment {
            let view = depth_stencil_attachment.view;
            self.oneshot_transition_image(view, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            let depth_view_info = self.image_views.get_ref(view).unwrap();
            let depth_img = self.images.get_ref(depth_view_info.img).unwrap();
            let depth_view = depth_view_info.view;
            width = std::cmp::min(depth_img.dim[0], width);
            height = std::cmp::min(depth_img.dim[1], height);

            attachments.push(depth_view);
            created_images.push(depth_view_info.img);
        }

        // Create framebuffer
        let framebuffer_info = vk::FramebufferCreateInfo {
            render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width,
            height,
            layers: 1,
            ..Default::default()
        };

        let framebuffer = unsafe {
            self.device
                .create_framebuffer(&framebuffer_info, None)
                .expect("Failed to create framebuffer")
        };

        // Store the framebuffer and the associated images
        framebuffers_with_images.push((framebuffer, created_images));

        Ok(framebuffers_with_images)
    }

    pub fn destroy_display(&mut self, dsp: Display) {
        unsafe { dsp.sc_loader.destroy_swapchain(dsp.swapchain, None) };
        unsafe { dsp.loader.destroy_surface(dsp.surface, None) };
        for sem in dsp.semaphores {
            unsafe { self.device.destroy_semaphore(sem.raw, None) };
        }

        for fence in &dsp.fences {
            unsafe { self.device.destroy_fence(fence.raw, None) };
        }
    }

    #[cfg(feature = "dashi-sdl2")]
    fn make_window(
        &mut self,
        info: &WindowInfo,
    ) -> (std::cell::Cell<sdl2::video::Window>, vk::SurfaceKHR) {
        let mut window = std::cell::Cell::new(
            self.sdl_video
                .window(&info.title, info.size[0], info.size[1])
                .vulkan()
                .build()
                .expect("Unable to create SDL2 Window!"),
        );

        let surface = window
            .get_mut()
            .vulkan_create_surface(vk::Handle::as_raw(self.instance.handle()) as usize)
            .expect("Unable to create vulkan surface!");

        (window, vk::Handle::from_raw(surface))
    }

    pub fn make_display(&mut self, info: &DisplayInfo) -> Result<Display, GPUError> {
        let (window, surface) = self.make_window(&info.window);

        let loader = ash::extensions::khr::Surface::new(&self.entry, &self.instance);
        let capabilities =
            unsafe { loader.get_physical_device_surface_capabilities(self.pdevice, surface)? };
        let _formats =
            unsafe { loader.get_physical_device_surface_formats(self.pdevice, surface)? };
        let _present_modes =
            unsafe { loader.get_physical_device_surface_present_modes(self.pdevice, surface)? };

        // Choose extent
        let size = info.window.size;
        let mut chosen_extent = vk::Extent2D {
            width: size[0],
            height: size[1],
        };
        if capabilities.current_extent.width != std::u32::MAX {
            chosen_extent = capabilities.current_extent.clone();
        } else {
            chosen_extent.width = std::cmp::max(
                capabilities.min_image_extent.width,
                std::cmp::min(capabilities.max_image_extent.width, chosen_extent.width),
            );
            chosen_extent.height = std::cmp::max(
                capabilities.min_image_extent.height,
                std::cmp::min(capabilities.max_image_extent.height, chosen_extent.height),
            );
        }

        // Select a present mode.
        let present_mode = if info.vsync {
            vk::PresentModeKHR::FIFO
        } else {
            vk::PresentModeKHR::IMMEDIATE
        };

        let wanted_format = vk::Format::B8G8R8A8_SRGB;
        let wanted_color_space = vk::ColorSpaceKHR::SRGB_NONLINEAR;

        let image_usage = vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::COLOR_ATTACHMENT;

        let num_framebuffers = match info.buffering {
            WindowBuffering::Double => 2,
            WindowBuffering::Triple => 3,
        };
        let swap_loader = ash::extensions::khr::Swapchain::new(&self.instance, &self.device);
        let swapchain = unsafe {
            swap_loader.create_swapchain(
                &vk::SwapchainCreateInfoKHR::builder()
                    .surface(surface)
                    .present_mode(present_mode)
                    .image_format(wanted_format)
                    .image_array_layers(1)
                    .image_color_space(wanted_color_space)
                    .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .image_extent(chosen_extent)
                    .image_usage(image_usage)
                    .min_image_count(std::cmp::max(
                        num_framebuffers,
                        capabilities.min_image_count,
                    ))
                    .pre_transform(capabilities.current_transform)
                    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                    .build(),
                Default::default(),
            )?
        };

        // Now, we need to make the images!
        let images = unsafe { swap_loader.get_swapchain_images(swapchain)? };
        let mut handles: Vec<Handle<Image>> = Vec::with_capacity(images.len() as usize);
        let mut view_handles: Vec<Handle<ImageView>> = Vec::with_capacity(images.len() as usize);
        for img in images {
            let sub_range = vk::ImageSubresourceRange::builder()
                .base_array_layer(0)
                .base_mip_level(0)
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1)
                .level_count(1)
                .build();

            match self.images.insert(Image {
                img,
                alloc: unsafe { std::mem::MaybeUninit::zeroed().assume_init() },
                layout: vk::ImageLayout::UNDEFINED,
                sub_layers: vk::ImageSubresourceLayers::builder()
                    .layer_count(1)
                    .mip_level(0)
                    .base_array_layer(0)
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .build(),
                extent: vk::Extent3D::builder()
                    .width(size[0])
                    .height(size[1])
                    .depth(1)
                    .build(),
                dim: [chosen_extent.width, chosen_extent.height, 1],
                format: vk_to_lib_image_format(wanted_format),
            }) {
                Some(handle) => {
                    self.oneshot_transition_image_noview(handle, vk::ImageLayout::PRESENT_SRC_KHR);
                    let h = self.make_image_view(&ImageViewInfo {
                        debug_name: "",
                        img: handle,
                        layer: 0,
                        mip_level: 0,
                    })?;

                    view_handles.push(h);
                    handles.push(handle)
                }
                None => todo!(),
            };
        }

        let mut sems = Vec::with_capacity(handles.len() as usize);
        let mut fences = Vec::with_capacity(handles.len() as usize);
        for idx in 0..handles.len() {
            sems.push(Semaphore {
                raw: unsafe {
                    self.device
                        .create_semaphore(&vk::SemaphoreCreateInfo::builder().build(), None)?
                },
            });
            let f = unsafe {
                self.device.create_fence(
                    &vk::FenceCreateInfo::builder()
                        .flags(vk::FenceCreateFlags::SIGNALED)
                        .build(),
                    None,
                )
            }?;

            fences.push(Fence::new(f, self.device.clone()));
        }
        return Ok(Display {
            window,
            swapchain,
            surface,
            images: handles,
            loader,
            sc_loader: swap_loader,
            frame_idx: 0,
            semaphores: sems,
            fences,
            views: view_handles,
        });
    }

    pub fn acquire_new_image(
        &self,
        dsp: &mut Display,
    ) -> Result<(Handle<ImageView>, Semaphore, u32, bool), GPUError> {
        let signal_sem = dsp.semaphores[dsp.frame_idx as usize];
        let fence = &mut dsp.fences[dsp.frame_idx as usize];

        fence.wait()?;
        fence.reset()?;
        let res = unsafe {
            dsp.sc_loader.acquire_next_image2(
                &vk::AcquireNextImageInfoKHR::builder()
                    .swapchain(dsp.swapchain)
                    .semaphore(signal_sem.raw)
                    .fence(dsp.fences[dsp.frame_idx as usize].raw)
                    .timeout(std::u64::MAX)
                    .device_mask(0x1)
                    .build(),
            )
        }?;

        dsp.frame_idx = res.0;
        return Ok((dsp.views[res.0 as usize], signal_sem, res.0, res.1));
    }

    pub fn present_display(
        &mut self,
        dsp: &Display,
        wait_sems: &[Semaphore],
    ) -> Result<(), GPUError> {
        let mut raw_wait_sems: Vec<vk::Semaphore> = Vec::with_capacity(32);
        for sem in wait_sems {
            raw_wait_sems.push(sem.raw);
        }

        unsafe {
            dsp.sc_loader.queue_present(
                self.gfx_queue.queue,
                &vk::PresentInfoKHR::builder()
                    .image_indices(&[dsp.frame_idx])
                    .swapchains(&[dsp.swapchain])
                    .wait_semaphores(&raw_wait_sems)
                    .build(),
            )
        }?;

        return Ok(());
    }
}

#[test]
fn test_context() {
    let ctx = Context::new(&ContextInfo {});
    assert!(ctx.is_ok());
}

#[test]
fn test_buffer() {
    let c_buffer_size = 1280;
    let c_test_val = 8 as u8;
    let mut ctx = Context::new(&Default::default()).unwrap();

    let initial_data = vec![c_test_val as u8; c_buffer_size as usize];
    let buffer_res = ctx.make_buffer(&BufferInfo {
        debug_name: "Test Buffer",
        byte_size: c_buffer_size,
        visibility: MemoryVisibility::CpuAndGpu,
        initial_data: Some(&initial_data),
        ..Default::default()
    });

    assert!(buffer_res.is_ok());

    let buffer = buffer_res.unwrap();

    let mapped_res = ctx.map_buffer::<u8>(buffer);
    assert!(mapped_res.is_ok());

    let mapped = mapped_res.unwrap();
    for byte in mapped {
        assert_eq!(*byte, c_test_val);
    }

    let res = ctx.unmap_buffer(buffer);
    assert!(res.is_ok());

    ctx.destroy_buffer(buffer);
}

#[test]
fn test_image() {
    let c_test_dim: [u32; 3] = [1280, 1024, 1];
    let c_format = Format::RGBA8;
    let c_mip_levels = 1;
    let c_test_val = 8 as u8;
    let initial_data =
        vec![c_test_val as u8; (c_test_dim[0] * c_test_dim[1] * c_test_dim[2] * 4) as usize];
    let mut ctx = Context::new(&Default::default()).unwrap();
    let image_res = ctx.make_image(&ImageInfo {
        debug_name: "Test Image",
        dim: c_test_dim,
        format: c_format,
        mip_levels: c_mip_levels,
        initial_data: Some(&initial_data),
    });

    assert!(image_res.is_ok());
    let image = image_res.unwrap();
    ctx.destroy_image(image);
}
