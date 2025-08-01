mod error;
use crate::utils::{
    offset_alloc::{self},
    Handle, Pool,
};
use ash::*;
#[cfg(feature = "dashi-minifb")]
use minifb;
#[cfg(feature = "dashi-winit")]
use winit;
#[cfg(feature = "dashi-openxr")]
use openxr as xr;
pub use error::*;
use std::{
    collections::HashMap,
    ffi::{c_char, CStr, CString},
};
use vk_mem::Alloc;

pub mod device_selector;
pub use device_selector::*;
pub mod structs;
pub use structs::*;
pub mod builders;
pub mod commands;
pub use commands::*;
pub mod framed_cmd_list;
pub use framed_cmd_list::*;
pub mod timing;
pub use timing::*;
#[cfg(feature = "dashi-minifb")]
pub mod minifb_window;
#[cfg(feature = "dashi-winit")]
pub mod winit_window;
#[cfg(feature = "dashi-openxr")]
pub mod openxr_window;

// Convert Filter enum to VkFilter
impl From<Filter> for vk::Filter {
    fn from(filter: Filter) -> Self {
        match filter {
            Filter::Nearest => vk::Filter::NEAREST,
            Filter::Linear => vk::Filter::LINEAR,
        }
    }
}

impl From<BlendFactor> for vk::BlendFactor {
    fn from(op: BlendFactor) -> Self {
        match op {
            BlendFactor::One => vk::BlendFactor::ONE,
            BlendFactor::Zero => vk::BlendFactor::ZERO,
            BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
            BlendFactor::InvSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
            BlendFactor::InvSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
            BlendFactor::InvDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
            BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
            BlendFactor::InvDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            BlendFactor::BlendFactor => vk::BlendFactor::CONSTANT_ALPHA,
        }
    }
}

impl From<AspectMask> for vk::ImageAspectFlags {
    fn from(value: AspectMask) -> Self {
        match value {
            AspectMask::Color => vk::ImageAspectFlags::COLOR,
            AspectMask::Depth => vk::ImageAspectFlags::DEPTH,
            AspectMask::Stencil => vk::ImageAspectFlags::STENCIL,
            AspectMask::DepthStencil => vk::ImageAspectFlags::STENCIL | vk::ImageAspectFlags::DEPTH,
        }
    }
}

impl From<BlendOp> for vk::BlendOp {
    fn from(op: BlendOp) -> Self {
        match op {
            BlendOp::Add => vk::BlendOp::ADD,
            BlendOp::Subtract => vk::BlendOp::SUBTRACT,
            BlendOp::InvSubtract => vk::BlendOp::REVERSE_SUBTRACT,
            BlendOp::Min => vk::BlendOp::MIN,
            BlendOp::Max => vk::BlendOp::MAX,
        }
    }
}

impl From<WriteMask> for vk::ColorComponentFlags {
    fn from(op: WriteMask) -> Self {
        let mut flags = vk::ColorComponentFlags::empty();

        if op.r {
            flags |= vk::ColorComponentFlags::R;
        }
        if op.g {
            flags |= vk::ColorComponentFlags::G;
        }
        if op.b {
            flags |= vk::ColorComponentFlags::B;
        }
        if op.a {
            flags |= vk::ColorComponentFlags::A;
        }

        flags
    }
}

impl From<ColorBlendState> for vk::PipelineColorBlendAttachmentState {
    fn from(state: ColorBlendState) -> Self {
        // Step 8: Color Blend State
        vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(state.write_mask.into())
            .src_color_blend_factor(state.src_blend.into())
            .dst_color_blend_factor(state.dst_blend.into())
            .src_alpha_blend_factor(state.src_alpha_blend.into())
            .dst_alpha_blend_factor(state.dst_alpha_blend.into())
            .color_blend_op(state.blend_op.into())
            .alpha_blend_op(state.alpha_blend_op.into())
            .blend_enable(state.enable)
            .build()
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

impl From<DynamicState> for vk::DynamicState {
    fn from(state: DynamicState) -> Self {
        match state {
            DynamicState::Viewport => vk::DynamicState::VIEWPORT,
            DynamicState::Scissor => vk::DynamicState::SCISSOR,
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
        Format::R8Uint => vk::Format::R8_UINT,
        Format::R8Sint => vk::Format::R8_SINT,
        Format::RGBA8Unorm => vk::Format::R8G8B8A8_UNORM,
    }
}

fn vk_to_lib_image_format(fmt: vk::Format) -> Format {
    match fmt {
        vk::Format::R8G8B8_SRGB => return Format::RGB8,
        vk::Format::R32G32B32A32_SFLOAT => return Format::RGBA32F,
        vk::Format::R8G8B8A8_SRGB => return Format::RGBA8,
        vk::Format::B8G8R8A8_SRGB => return Format::BGRA8,
        vk::Format::B8G8R8A8_SNORM => return Format::BGRA8Unorm,
        vk::Format::R8_SINT => return Format::R8Sint,
        vk::Format::R8_UINT => return Format::R8Uint,
        _ => todo!(),
    }
}

pub fn channel_count(fmt: &Format) -> u32 {
    match fmt {
        Format::RGB8 => 3,
        Format::BGRA8 | Format::BGRA8Unorm | Format::RGBA8 | Format::RGBA8Unorm | Format::RGBA32F => 4,
        Format::D24S8 => 4,
        Format::R8Sint | Format::R8Uint => 1,
    }
}

pub fn bytes_per_channel(fmt: &Format) -> u32 {
    match fmt {
        Format::RGB8
        | Format::BGRA8
        | Format::BGRA8Unorm
        | Format::RGBA8 
        | Format::RGBA8Unorm
        | Format::R8Sint 
        | Format::R8Uint => 1,
        Format::RGBA32F => 4,
        Format::D24S8 => 3,
    }
}

pub fn mip_dimensions(dim: [u32; 3], level: u32) -> [u32; 3] {
    [
        std::cmp::max(1, dim[0] >> level),
        std::cmp::max(1, dim[1] >> level),
        std::cmp::max(1, dim[2] >> level),
    ]
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

#[derive(Debug)]
pub struct Buffer {
    buf: vk::Buffer,
    alloc: vk_mem::Allocation,
    offset: u32,
    size: u32,
    suballocated: bool,
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
            alloc: unsafe { std::mem::transmute_copy(&self.alloc) },
            offset: self.offset.clone(),
            size: self.size.clone(),
            suballocated: self.suballocated.clone(),
        }
    }
}
impl Handle<Buffer> {
    pub fn to_unmapped_dynamic(&self, byte_offset: u32) -> DynamicBuffer {
        return DynamicBuffer {
            handle: self.clone(),
            alloc: offset_alloc::Allocation {
                offset: byte_offset,
                metadata: 0,
            },
            ptr: std::ptr::null_mut(),
            size: 0,
        };
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DynamicBuffer {
    pub(crate) handle: Handle<Buffer>,
    pub(crate) alloc: offset_alloc::Allocation,
    pub(crate) ptr: *mut u8,
    pub(crate) size: u16,
}

impl Default for DynamicBuffer {
    fn default() -> Self {
        Self {
            handle: Default::default(),
            alloc: offset_alloc::Allocation {
                offset: 0,
                metadata: 0,
            },
            ptr: std::ptr::null_mut(),
            size: Default::default(),
        }
    }
}
impl DynamicBuffer {
    pub fn handle(&self) -> Handle<Buffer> {
        self.handle
    }

    pub fn offset(&self) -> u32 {
        self.alloc.offset
    }

    pub fn slice<T>(&mut self) -> &mut [T] {
        let typed_map: *mut T = unsafe { std::mem::transmute(self.ptr) };
        return unsafe {
            std::slice::from_raw_parts_mut(typed_map, self.size as usize / std::mem::size_of::<T>())
        };
    }
}

#[derive(Clone)]
pub struct DynamicAllocator {
    allocator: offset_alloc::Allocator,
    ptr: *mut u8,
    min_alloc_size: u32,
    pool: Handle<Buffer>,
}

impl Default for DynamicAllocator {
    fn default() -> Self {
        Self {
            allocator: Default::default(),
            ptr: std::ptr::null_mut(),
            min_alloc_size: Default::default(),
            pool: Default::default(),
        }
    }
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
    layouts: Vec<vk::ImageLayout>,
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
pub(super) struct SubpassContainer {
    pub(super) fb: vk::Framebuffer,
    pub(super) clear_values: Vec<vk::ClearValue>,
}

#[derive(Clone, Default)]
pub struct RenderPass {
    pub(super) raw: vk::RenderPass,
    pub(super) viewport: Viewport,
    pub(super) subpasses: HashMap<u64, SubpassContainer>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct BindGroupLayout {
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    variables: Vec<BindGroupVariable>,
    bindless: bool,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct BindGroup {
    set: vk::DescriptorSet,
    set_id: u32,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct IndexedBindGroup {
    set: vk::DescriptorSet,
    set_id: u32,
}

#[allow(dead_code)]
pub struct Display {
    #[cfg(all(feature = "dashi-sdl2", not(feature = "dashi-openxr")))]
    window: std::cell::Cell<sdl2::video::Window>,
    #[cfg(all(feature = "dashi-minifb", not(feature = "dashi-openxr")))]
    window: minifb::Window,
    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    window: winit::window::Window,
    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    event_loop: winit::event_loop::EventLoop<()>,
    #[cfg(not(feature = "dashi-openxr"))]
    swapchain: ash::vk::SwapchainKHR,
    #[cfg(not(feature = "dashi-openxr"))]
    surface: ash::vk::SurfaceKHR,
    #[cfg(not(feature = "dashi-openxr"))]
    images: Vec<Handle<Image>>,
    #[cfg(not(feature = "dashi-openxr"))]
    views: Vec<Handle<ImageView>>,
    #[cfg(not(feature = "dashi-openxr"))]
    loader: ash::extensions::khr::Surface,
    #[cfg(not(feature = "dashi-openxr"))]
    sc_loader: ash::extensions::khr::Swapchain,
    #[cfg(not(feature = "dashi-openxr"))]
    semaphores: Vec<Handle<Semaphore>>,
    #[cfg(not(feature = "dashi-openxr"))]
    fences: Vec<Handle<Fence>>,
    #[cfg(not(feature = "dashi-openxr"))]
    frame_idx: u32,

    #[cfg(feature = "dashi-openxr")]
    xr_instance: xr::Instance,
    #[cfg(feature = "dashi-openxr")]
    xr_session: xr::Session<xr::Vulkan>,
    #[cfg(feature = "dashi-openxr")]
    xr_waiter: xr::FrameWaiter,
    #[cfg(feature = "dashi-openxr")]
    xr_stream: xr::FrameStream<xr::Vulkan>,
    #[cfg(feature = "dashi-openxr")]
    xr_swapchain: xr::Swapchain<xr::Vulkan>,
    #[cfg(feature = "dashi-openxr")]
    xr_images: Vec<vk::Image>,
    #[cfg(feature = "dashi-openxr")]
    xr_view_config: Vec<xr::ViewConfigurationView>,
    #[cfg(feature = "dashi-openxr")]
    xr_input: openxr_window::XrInput,
}

impl Display {
    #[cfg(all(feature = "dashi-minifb", not(feature = "dashi-openxr")))]
    pub fn minifb_window(&mut self) -> &mut minifb::Window {
        &mut self.window
    }
    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub fn winit_window(&self) -> &winit::window::Window {
        &self.window
    }
    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    pub fn winit_event_loop(&mut self) -> &mut winit::event_loop::EventLoop<()> {
        &mut self.event_loop
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_instance(&self) -> &xr::Instance {
        &self.xr_instance
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_session(&self) -> &xr::Session<xr::Vulkan> {
        &self.xr_session
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_frame_waiter(&mut self) -> &mut xr::FrameWaiter {
        &mut self.xr_waiter
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_frame_stream(&mut self) -> &mut xr::FrameStream<xr::Vulkan> {
        &mut self.xr_stream
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_swapchain(&self) -> &xr::Swapchain<xr::Vulkan> {
        &self.xr_swapchain
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_swapchain_images(&self) -> &[vk::Image] {
        &self.xr_images
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_view_configuration(&self) -> &[xr::ViewConfigurationView] {
        &self.xr_view_config
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn poll_inputs(&mut self) -> xr::Result<openxr_window::XrInputState> {
        self.xr_input.poll_inputs()
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn xr_input(&self) -> &openxr_window::XrInput {
        &self.xr_input
    }
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

#[derive(Clone, Default, Debug)]
pub struct ComputePipelineLayout {
    shader_stage: vk::PipelineShaderStageCreateInfo,
    layout: vk::PipelineLayout,
}

#[derive(Clone, Default, Debug)]
pub struct GraphicsPipelineLayout {
    input_assembly: vk::PipelineInputAssemblyStateCreateInfo,
    shader_stages: Vec<vk::PipelineShaderStageCreateInfo>,
    depth_stencil: Option<vk::PipelineDepthStencilStateCreateInfo>,
    multisample: vk::PipelineMultisampleStateCreateInfo,
    rasterizer: vk::PipelineRasterizationStateCreateInfo,
    vertex_input: vk::VertexInputBindingDescription,
    color_blend_states: Vec<vk::PipelineColorBlendAttachmentState>,
    vertex_attribs: Vec<vk::VertexInputAttributeDescription>,
    dynamic_states: Vec<vk::DynamicState>,
    layout: vk::PipelineLayout,
}

#[derive(Clone, Default)]
pub struct ComputePipeline {
    raw: vk::Pipeline,
    layout: Handle<ComputePipelineLayout>,
}

#[derive(Clone, Default)]
pub struct GraphicsPipeline {
    raw: vk::Pipeline,
    render_pass: Handle<RenderPass>,
    layout: Handle<GraphicsPipelineLayout>,
}

#[derive(Clone)]
pub struct CommandList {
    cmd_buf: vk::CommandBuffer,
    fence: Handle<Fence>,
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
    pub(super) semaphores: Pool<Semaphore>,
    pub(super) fences: Pool<Fence>,
    pub(super) images: Pool<Image>,
    pub(super) image_views: Pool<ImageView>,
    pub(super) samplers: Pool<Sampler>,
    pub(super) bind_group_layouts: Pool<BindGroupLayout>,
    pub(super) bind_groups: Pool<BindGroup>,
    pub(super) gfx_pipeline_layouts: Pool<GraphicsPipelineLayout>,
    pub(super) gfx_pipelines: Pool<GraphicsPipeline>,
    pub(super) compute_pipeline_layouts: Pool<ComputePipelineLayout>,
    pub(super) compute_pipelines: Pool<ComputePipeline>,
    pub(super) cmds_to_release: Vec<(CommandList, Handle<Fence>)>,

    pub(super) gpu_timers: Vec<GpuTimer>,
    pub(super) timestamp_period: f32,

    /// Indicates whether the context was created in headless mode
    headless: bool,

    #[cfg(feature = "dashi-sdl2")]
    pub(super) sdl_context: sdl2::Sdl,
    #[cfg(feature = "dashi-sdl2")]
    pub(super) sdl_video: Option<sdl2::VideoSubsystem>,
    #[cfg(debug_assertions)]
    pub(super) debug_utils: ash::extensions::ext::DebugUtils,
}

impl Default for Context {
    fn default() -> Self {
        Self::new(&ContextInfo::default()).unwrap()
    }
}

impl std::panic::UnwindSafe for Context {}

impl Context {
    fn init_core(
        info: &ContextInfo,
        windowed: bool,
    ) -> Result<
        (
            ash::Entry,
            ash::Instance,
            vk::PhysicalDevice,
            ash::Device,
            ash::vk::PhysicalDeviceProperties,
            vk::CommandPool,
            vk_mem::Allocator,
            Queue,
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
        #[cfg(debug_assertions)]
        inst_exts.push(ash::extensions::ext::DebugUtils::name().as_ptr());

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
                    .build(),
                None,
            )
        }?;

        let pdevice = unsafe { instance.enumerate_physical_devices()?[info.device.device_id] };
        let device_prop = unsafe { instance.get_physical_device_properties(pdevice) };

        let queue_prop = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
        let _features = unsafe { instance.get_physical_device_features(pdevice) };

        let mut queue_family: u32 = 0;

        let mut gfx_queue = Queue::default();
        for prop in queue_prop.iter() {
            if prop.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                gfx_queue.family = queue_family;
            }
            queue_family += 1;
        }

        let priorities = [1.0];

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

        // Bindless enabled
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
                unsafe {
                    std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_GOOGLE_user_type\0").as_ptr()
                },
                vk::GoogleHlslFunctionality1Fn::name().as_ptr(),
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                KhrPortabilitySubsetFn::name().as_ptr(),
            ]
        } else {
            vec![unsafe {
                std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_GOOGLE_user_type\0").as_ptr()
            }]
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

        let device = unsafe {
            instance.create_device(
                pdevice,
                &vk::DeviceCreateInfo::builder()
                    .enabled_extension_names(&extensions_to_enable)
                    .queue_create_infos(&[vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(gfx_queue.family)
                        .queue_priorities(&priorities)
                        .build()])
                    .push_next(&mut features2)
                    .push_next(&mut features16bit)
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

        return Ok((
            entry,
            instance,
            pdevice,
            device,
            device_prop,
            pool,
            allocator,
            gfx_queue,
        ));
    }

    pub fn headless(info: &ContextInfo) -> Result<Self, GPUError> {
        let (entry, instance, pdevice, device, properties, pool, allocator, gfx_queue) =
            Self::init_core(info, false)?;

        #[cfg(debug_assertions)]
        let debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);

        let ctx = Context {
            entry,
            instance,
            pdevice,
            device,
            properties,
            pool,
            allocator,
            gfx_queue,

            buffers: Default::default(),
            render_passes: Default::default(),
            semaphores: Default::default(),
            fences: Default::default(),
            images: Default::default(),
            image_views: Default::default(),
            samplers: Default::default(),
            bind_group_layouts: Default::default(),
            bind_groups: Default::default(),
            gfx_pipeline_layouts: Default::default(),
            gfx_pipelines: Default::default(),
            compute_pipeline_layouts: Default::default(),
            compute_pipelines: Default::default(),
            cmds_to_release: Default::default(),
            gpu_timers: Vec::new(),
            timestamp_period: properties.limits.timestamp_period,
            headless: true,

            #[cfg(feature = "dashi-sdl2")]
            sdl_context: sdl2::init().unwrap(),
            #[cfg(feature = "dashi-sdl2")]
            sdl_video: None,
            #[cfg(debug_assertions)]
            debug_utils,
        };
        let mut ctx = ctx;
        ctx.init_gpu_timers(1)?;
        Ok(ctx)
    }

    pub fn new(info: &ContextInfo) -> Result<Self, GPUError> {
        let (entry, instance, pdevice, device, properties, pool, allocator, gfx_queue) =
            Self::init_core(info, true)?;

        #[cfg(debug_assertions)]
        let debug_utils = ash::extensions::ext::DebugUtils::new(&entry, &instance);

        #[cfg(feature = "dashi-sdl2")]
        let sdl_context = sdl2::init().unwrap();
        #[cfg(feature = "dashi-sdl2")]
        let sdl_video = sdl_context.video().unwrap();

        let ctx = Context {
            entry,
            instance,
            pdevice,
            device,
            properties,
            pool,
            allocator,
            gfx_queue,

            buffers: Default::default(),
            render_passes: Default::default(),
            semaphores: Default::default(),
            fences: Default::default(),
            images: Default::default(),
            image_views: Default::default(),
            samplers: Default::default(),
            bind_group_layouts: Default::default(),
            bind_groups: Default::default(),
            gfx_pipeline_layouts: Default::default(),
            gfx_pipelines: Default::default(),
            compute_pipeline_layouts: Default::default(),
            compute_pipelines: Default::default(),
            cmds_to_release: Default::default(),
            gpu_timers: Vec::new(),
            timestamp_period: properties.limits.timestamp_period,
            headless: false,

            #[cfg(feature = "dashi-sdl2")]
            sdl_context,
            #[cfg(feature = "dashi-sdl2")]
            sdl_video: Some(sdl_video),
            #[cfg(debug_assertions)]
            debug_utils,
        };
        let mut ctx = ctx;
        ctx.init_gpu_timers(1)?;
        Ok(ctx)
    }

    #[cfg(feature = "dashi-sdl2")]
    pub fn get_sdl_ctx(&mut self) -> &mut sdl2::Sdl {
        return &mut self.sdl_context;
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

    pub fn wait(&mut self, fence: Handle<Fence>) -> Result<(), GPUError> {
        let fence = self.fences.get_ref(fence).unwrap();
        let _res = unsafe {
            self.device
                .wait_for_fences(&[fence.raw], true, std::u64::MAX)
        }?;

        unsafe { self.device.reset_fences(&[fence.raw]) }?;

        Ok(())
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
                    .flags(vk::FenceCreateFlags::empty())
                    .build(),
                None,
            )
        }?;

        self.set_name(
            f,
            format!("{}.fence", info.debug_name).as_str(),
            vk::ObjectType::FENCE,
        );

        unsafe {
            self.device
                .begin_command_buffer(cmd[0], &vk::CommandBufferBeginInfo::builder().build())?
        };

        return Ok(CommandList {
            cmd_buf: cmd[0],
            fence: self.fences.insert(Fence::new(f)).unwrap(),
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
        let fence = self.submit(&mut list, &Default::default()).unwrap();
        self.wait(fence).unwrap();
        self.destroy_cmd_list(list);
    }

    fn oneshot_transition_image_noview(&mut self, img: Handle<Image>, layout: vk::ImageLayout) {
        let tmp_view = self
            .make_image_view(&ImageViewInfo {
                debug_name: "oneshot_view",
                img: img,
                layer: 0,
                mip_level: 0,
                ..Default::default()
            })
            .unwrap();

        let mut list = self
            .begin_command_list(&CommandListInfo {
                debug_name: "oneshot_transition",
                ..Default::default()
            })
            .unwrap();
        self.transition_image(list.cmd_buf, tmp_view, layout);
        let fence = self.submit(&mut list, &Default::default()).unwrap();
        self.wait(fence).unwrap();
        self.destroy_cmd_list(list);
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
        unsafe {
            self.device.queue_submit(
                self.gfx_queue.queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&[cmd.cmd_buf])
                    .signal_semaphores(&raw_signal_sems)
                    .wait_dst_stage_mask(&stage_masks)
                    .wait_semaphores(&raw_wait_sems)
                    .build()],
                self.fences.get_ref(cmd.fence).unwrap().raw.clone(),
            )?
        };

        for (cmd, fence) in self.cmds_to_release.clone() {
            self.wait(fence.clone())?;
            self.destroy_cmd_list(cmd);
        }

        self.cmds_to_release.clear();

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

    pub fn make_image_view(&mut self, info: &ImageViewInfo) -> Result<Handle<ImageView>, GPUError> {
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
        let mut typed_map: *mut T = unsafe { std::mem::transmute(mapped) };
        typed_map = unsafe { typed_map.offset(buf.offset as isize) };
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
        let mut typed_map: *mut T = unsafe { std::mem::transmute(mapped) };
        typed_map = unsafe { typed_map.offset(buf.offset as isize) };
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
                list.append(Command::BufferCopy(BufferCopy {
                    src: staging,
                    dst: buf,
                    src_offset: 0,
                    dst_offset: 0,
                    size: unsafe { info.initial_data.unwrap_unchecked().len() },
                }));

                let fence = self.submit(&mut list, &Default::default())?;
                self.wait(fence.clone())?;
                self.destroy_cmd_list(list);
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
        let tmp_view = self.make_image_view(&ImageViewInfo {
            debug_name: "view",
            img: image,
            layer: 0,
            mip_level: 0,
            aspect: match info.format {
                Format::R8Sint
                | Format::R8Uint
                | Format::RGB8
                | Format::BGRA8
                | Format::BGRA8Unorm
                | Format::RGBA8
                | Format::RGBA32F => AspectMask::Color,
                Format::D24S8 => AspectMask::DepthStencil,
                _ => AspectMask::Color,
            },
        })?;

        let mut list = self.begin_command_list(&Default::default())?;
        if info.initial_data.is_none() {
            self.transition_image(list.cmd_buf, tmp_view, vk::ImageLayout::GENERAL);
            let fence = self.submit(&mut list, &Default::default())?;
            self.wait(fence)?;
            self.destroy_cmd_list(list);
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

        // 1) barrier: UNDEFINED -> TRANSFER_DST_OPTIMAL
        self.transition_image_stages(
            list.cmd_buf,
            tmp_view,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            Default::default(),
            vk::AccessFlags::TRANSFER_WRITE,
        );

        list.append(Command::BufferImageCopy(BufferImageCopy {
            src: staging,
            dst: tmp_view,
            src_offset: 0,
        }));

        // 3) barrier: TRANSFER_DST_OPTIMAL -> GENERAL
        self.transition_image_stages(
            list.cmd_buf,
            tmp_view,
            vk::ImageLayout::GENERAL,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );

        if info.mip_levels > 1 {
            for i in 0..info.mip_levels - 1 {
                let src_view = self
                    .make_image_view(&ImageViewInfo {
                        debug_name: "mip_src",
                        img: image,
                        layer: 0,
                        mip_level: i,
                        aspect: if info.format == Format::D24S8 {
                            AspectMask::DepthStencil
                        } else {
                            AspectMask::Color
                        },
                    })?;
                let dst_view = self
                    .make_image_view(&ImageViewInfo {
                        debug_name: "mip_dst",
                        img: image,
                        layer: 0,
                        mip_level: i + 1,
                        aspect: if info.format == Format::D24S8 {
                            AspectMask::DepthStencil
                        } else {
                            AspectMask::Color
                        },
                    })?;

                self.transition_image(list.cmd_buf, src_view, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                self.transition_image(list.cmd_buf, dst_view, vk::ImageLayout::TRANSFER_DST_OPTIMAL);

                list.append(Command::Blit(ImageBlit {
                    src: src_view,
                    dst: dst_view,
                    filter: Filter::Linear,
                    ..Default::default()
                }));

                self.transition_image(list.cmd_buf, src_view, vk::ImageLayout::GENERAL);
                if i + 1 < info.mip_levels - 1 {
                    self.transition_image(list.cmd_buf, dst_view, vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
                } else {
                    self.transition_image(list.cmd_buf, dst_view, vk::ImageLayout::GENERAL);
                }

                self.destroy_image_view(src_view);
                self.destroy_image_view(dst_view);
            }
        }

        let fence = self.submit(&mut list, &Default::default())?;
        self.wait(fence)?;
        self.destroy_cmd_list(list);
        self.destroy_buffer(staging);
        self.destroy_image_view(tmp_view);
        Ok(())
    }

    pub fn sync_current_device(&mut self) {
        unsafe { self.device.device_wait_idle().unwrap() };
    }

    pub fn init_gpu_timers(&mut self, count: usize) -> Result<(), GPUError> {
        for timer in self.gpu_timers.drain(..) {
            unsafe { self.device.destroy_query_pool(timer.pool, None) };
        }
        let mut timers = Vec::with_capacity(count);
        for _ in 0..count {
            timers.push(GpuTimer::new(&self.device)?);
        }
        self.gpu_timers = timers;
        Ok(())
    }

    pub fn gpu_timer_begin(&mut self, list: &mut CommandList, frame: usize) {
        if let Some(t) = self.gpu_timers.get(frame) {
            unsafe { t.begin(&self.device, list.cmd_buf) };
        }
    }

    pub fn gpu_timer_end(&mut self, list: &mut CommandList, frame: usize) {
        if let Some(t) = self.gpu_timers.get(frame) {
            unsafe { t.end(&self.device, list.cmd_buf) };
        }
    }

    pub fn get_elapsed_gpu_time_ms(&mut self, frame: usize) -> Option<f32> {
        self.gpu_timers
            .get(frame)
            .and_then(|t| t.resolve(&self.device, self.timestamp_period).ok())
    }

    pub fn make_dynamic_allocator(
        &mut self,
        info: &DynamicAllocatorInfo,
    ) -> Result<DynamicAllocator, GPUError> {
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
            allocator: offset_alloc::Allocator::new(info.byte_size, info.num_allocations),
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

    pub fn clean_up(&mut self) {}

    pub fn destroy(mut self) {
        // Drain and destroy any pending command lists
        //        for (cmd, fence) in self.cmds_to_release.drain(..) {
        //            let _ = self.wait(fence.clone());
        //            unsafe { self.device.free_command_buffers(self.pool, &[cmd.cmd_buf]) };
        //            let fence = self.fences.get_mut_ref(fence).unwrap();
        //            unsafe { self.device.destroy_fence(fence.raw, None) };
        //        }

        //        // Bind groups
        //        self.bind_groups.for_each_occupied_mut(|bg| unsafe {
        //            self.device.free_descriptor_sets(bg.set, &[bg.set]).ok();
        //        });

        // Bind group layouts
        self.bind_group_layouts
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
            for (_, subpass) in &rp.subpasses {
                unsafe { self.device.destroy_framebuffer(subpass.fb, None) };
            }
            unsafe { self.device.destroy_render_pass(rp.raw, None) };
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
            unsafe { self.device.destroy_query_pool(timer.pool, None) };
        }

        // Command pool
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
        }

        // Allocator cleanup handled separately
        //        self.allocator.cleanup().ok(); // optional: wrap in Result if needed
        //

        drop(self.allocator);

        // Device and instance
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }

    pub fn destroy_dynamic_allocator(&mut self, alloc: DynamicAllocator) {
        self.unmap_buffer(alloc.pool).unwrap();
        self.destroy_buffer(alloc.pool);
    }

    pub fn destroy_buffer(&mut self, handle: Handle<Buffer>) {
        let buf = self.buffers.get_mut_ref(handle).unwrap();
        if !buf.suballocated {
            unsafe { self.allocator.destroy_buffer(buf.buf, &mut buf.alloc) };
        }
        self.buffers.release(handle);
    }

    pub fn destroy_semaphore(&mut self, handle: Handle<Semaphore>) {
        let sem = self.semaphores.get_mut_ref(handle).unwrap();
        unsafe { self.device.destroy_semaphore(sem.raw, None) };
        self.semaphores.release(handle);
    }

    pub fn destroy_fence(&mut self, handle: Handle<Fence>) {
        let fence = self.fences.get_mut_ref(handle).unwrap();
        unsafe { self.device.destroy_fence(fence.raw, None) };
        self.fences.release(handle);
    }

    pub fn destroy_image_view(&mut self, handle: Handle<ImageView>) {
        let img = self.image_views.get_mut_ref(handle).unwrap();
        unsafe { self.device.destroy_image_view(img.view, None) };
        self.image_views.release(handle);
    }
    pub fn destroy_image(&mut self, handle: Handle<Image>) {
        let img = self.images.get_mut_ref(handle).unwrap();
        unsafe { self.allocator.destroy_image(img.img, &mut img.alloc) };
        self.images.release(handle);
    }

    pub fn destroy_render_pass(&mut self, handle: Handle<RenderPass>) {
        let rp = self.render_passes.get_ref(handle).unwrap();
        for (_id, sb) in &rp.subpasses {
            unsafe { self.device.destroy_framebuffer(sb.fb, None) };
        }

        unsafe { self.device.destroy_render_pass(rp.raw, None) };

        self.render_passes.release(handle);
    }

    pub fn destroy_cmd_list(&mut self, list: CommandList) {
        unsafe { self.device.free_command_buffers(self.pool, &[list.cmd_buf]) };
        self.destroy_fence(list.fence);
    }

    pub fn make_bindless_bind_group_layout(
        &mut self,
        info: &BindGroupLayoutInfo,
    ) -> Result<Handle<BindGroupLayout>, GPUError> {
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

        // Step 4: Return the BindlessBindGroupLayout
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
                bindless: true,
            })
            .unwrap());
    }

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
                bindless: false,
            })
            .unwrap());
    }

    pub fn update_bind_group(&mut self, info: &BindGroupUpdateInfo) {
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
                }
            }
        }

        unsafe {
            self.device
                .update_descriptor_sets(&write_descriptor_sets, &[]);
        }
    }

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
        });

        Ok(bg)
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
        bind_group_layout_handle: &[Option<Handle<BindGroupLayout>>],
    ) -> Result<vk::PipelineLayout, GPUError> {
        let mut bgs = Vec::new();
        for bg in bind_group_layout_handle {
            if let Some(b) = bg {
                bgs.push(self.bind_group_layouts.get_ref(*b).unwrap().layout);
            }
        }

        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&bgs)
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
        let mut attachments = Vec::with_capacity(256);
        let mut color_attachment_refs = Vec::with_capacity(256);
        let mut subpasses = Vec::with_capacity(256);
        let mut deps = Vec::with_capacity(256);
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

        //        let fbs = self.create_framebuffers(render_pass, info)?;
        return Ok(self
            .render_passes
            .insert(RenderPass {
                raw: render_pass,
                viewport: info.viewport,
                subpasses: Default::default(),
            })
            .unwrap());
    }
    pub fn make_compute_pipeline_layout(
        &mut self,
        info: &ComputePipelineLayoutInfo,
    ) -> Result<Handle<ComputePipelineLayout>, GPUError> {
        let shader_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE) // HAS to be compute.
            .module(self.create_shader_module(info.shader.spirv).unwrap())
            .name(std::ffi::CStr::from_bytes_with_nul(b"main\0").unwrap()) // Entry point is usually "main"
            .build();

        let layout = self.create_pipeline_layout(&info.bg_layouts)?;

        return Ok(self
            .compute_pipeline_layouts
            .insert(ComputePipelineLayout {
                layout,
                shader_stage,
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
                    ShaderType::All => vk::ShaderStageFlags::ALL,
                    _ => todo!(),
                };

                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(stage_flags)
                    .module(self.create_shader_module(shader_info.spirv).unwrap())
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
        let layout = self.create_pipeline_layout(&info.bg_layouts)?;

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

    pub fn release_list_on_next_submit(&mut self, fence: Handle<Fence>, list: CommandList) {
        self.cmds_to_release.push((list, fence));
    }

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

    pub(super) fn make_framebuffer(
        &mut self,
        info: &RenderPassBegin,
    ) -> Result<vk::Framebuffer, GPUError> {
        //    ) -> Result<Vec<(vk::Framebuffer, Vec<Handle<Image>>)>, GPUError> {
        let mut width = std::u32::MAX;
        let mut height = std::u32::MAX;
        // Loop through each subpass and create a framebuffer
        let mut attachments = Vec::new();

        let mut created_images = Vec::new();
        // Collect the image views for color attachments
        for attachment in info.attachments.iter() {
            match attachment.clear {
                ClearValue::Color(_) | ClearValue::IntColor(_) | ClearValue::UintColor(_) => {
                    let created_image_handle = attachment.img;
                    self.oneshot_transition_image(
                        attachment.img,
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
                ClearValue::DepthStencil {
                    depth: _,
                    stencil: _,
                } => {
                    let view = attachment.img;
                    self.oneshot_transition_image(
                        view,
                        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    );
                    let depth_view_info = self.image_views.get_ref(view).unwrap();
                    let depth_img = self.images.get_ref(depth_view_info.img).unwrap();
                    let depth_view = depth_view_info.view;
                    width = std::cmp::min(depth_img.dim[0], width);
                    height = std::cmp::min(depth_img.dim[1], height);

                    attachments.push(depth_view);
                    created_images.push(depth_view_info.img);
                }
            }
        }

        let rp = self.render_passes.get_ref(info.render_pass).unwrap();
        // Create framebuffer
        let framebuffer_info = vk::FramebufferCreateInfo {
            render_pass: rp.raw,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width,
            height,
            layers: 1,
            ..Default::default()
        };

        return Ok(unsafe {
            self.device
                .create_framebuffer(&framebuffer_info, None)
                .expect("Failed to create framebuffer")
        });
    }

    #[cfg(not(feature = "dashi-openxr"))]
    pub fn destroy_display(&mut self, dsp: Display) {
        for img in &dsp.images {
            self.images.release(*img);
        }

        for img in &dsp.views {
            self.destroy_image_view(*img);
        }
        unsafe { dsp.sc_loader.destroy_swapchain(dsp.swapchain, None) };
        unsafe { dsp.loader.destroy_surface(dsp.surface, None) };
        for sem in dsp.semaphores {
            self.destroy_semaphore(sem);
        }

        for fence in &dsp.fences {
            self.destroy_fence(fence.clone());
        }
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn destroy_display(&mut self, _dsp: Display) {
        // OpenXR resources are cleaned up by Drop implementations
    }

    #[cfg(all(feature = "dashi-sdl2", not(feature = "dashi-openxr")))]
    fn make_window(
        &mut self,
        info: &WindowInfo,
    ) -> (std::cell::Cell<sdl2::video::Window>, vk::SurfaceKHR) {
        let mut window = std::cell::Cell::new(
            self.sdl_video
                .as_ref()
                .unwrap()
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

    #[cfg(all(feature = "dashi-minifb", not(feature = "dashi-openxr")))]
    fn make_window(
        &mut self,
        info: &WindowInfo,
    ) -> (minifb::Window, vk::SurfaceKHR) {
        minifb_window::create_window(&self.entry, &self.instance, info).unwrap()
    }

    #[cfg(all(feature = "dashi-winit", not(feature = "dashi-openxr")))]
    fn make_window(
        &mut self,
        info: &WindowInfo,
    ) -> (winit::event_loop::EventLoop<()>, winit::window::Window, vk::SurfaceKHR) {
        winit_window::create_window(&self.entry, &self.instance, info).unwrap()
    }

    #[cfg(not(feature = "dashi-openxr"))]
    pub fn make_display(&mut self, info: &DisplayInfo) -> Result<Display, GPUError> {
        if self.headless {
            return Err(GPUError::HeadlessDisplayNotSupported);
        }
        #[cfg(feature = "dashi-winit")]
        let (event_loop, window, surface) = self.make_window(&info.window);
        #[cfg(not(feature = "dashi-winit"))]
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
            match self.images.insert(Image {
                img,
                alloc: unsafe { std::mem::MaybeUninit::zeroed().assume_init() },
                layouts: vec![vk::ImageLayout::UNDEFINED],
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
                        debug_name: &info.window.title,
                        img: handle,
                        layer: 0,
                        mip_level: 0,
                        ..Default::default()
                    })?;

                    view_handles.push(h);
                    handles.push(handle)
                }
                None => todo!(),
            };
        }

        let sems = self.make_semaphores(handles.len()).unwrap();
        let mut fences = Vec::with_capacity(handles.len() as usize);
        for _idx in 0..handles.len() {
            fences.push(self.make_fence()?);
        }
        self.init_gpu_timers(handles.len())?;
        return Ok(Display {
            window,
            #[cfg(feature = "dashi-winit")]
            event_loop,
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

    #[cfg(feature = "dashi-openxr")]
    pub fn make_xr_display(&mut self, _info: &XrDisplayInfo) -> Result<Display, GPUError> {
        let (xr_instance, session, waiter, stream, swapchain, images, views) =
            openxr_window::create_xr_session(
                &self.instance,
                self.pdevice,
                &self.device,
                self.gfx_queue.family,
            )
            .map_err(|_| GPUError::LibraryError())?;

        let xr_input = openxr_window::XrInput::new(&xr_instance, &session)
            .map_err(|_| GPUError::LibraryError())?;

        Ok(Display {
            xr_instance,
            xr_session: session,
            xr_waiter: waiter,
            xr_stream: stream,
            xr_swapchain: swapchain,
            xr_images: images,
            xr_view_config: views,
            xr_input,
        })
    }

    #[cfg(not(feature = "dashi-openxr"))]
    pub fn acquire_new_image(
        &mut self,
        dsp: &mut Display,
    ) -> Result<(Handle<ImageView>, Handle<Semaphore>, u32, bool), GPUError> {
        let signal_sem_handle = dsp.semaphores[dsp.frame_idx as usize].clone();
        let fence = dsp.fences[dsp.frame_idx as usize];

        self.wait(fence.clone())?;

        let signal_sem = self.semaphores.get_ref(signal_sem_handle).unwrap();
        let res = unsafe {
            dsp.sc_loader.acquire_next_image2(
                &vk::AcquireNextImageInfoKHR::builder()
                    .swapchain(dsp.swapchain)
                    .semaphore(signal_sem.raw)
                    .fence(
                        self.fences
                            .get_ref(dsp.fences[dsp.frame_idx as usize])
                            .unwrap()
                            .raw,
                    )
                    .timeout(std::u64::MAX)
                    .device_mask(0x1)
                    .build(),
            )
        }?;

        dsp.frame_idx = res.0;
        return Ok((dsp.views[res.0 as usize], signal_sem_handle, res.0, res.1));
    }

    #[cfg(not(feature = "dashi-openxr"))]
    pub fn present_display(
        &mut self,
        dsp: &Display,
        wait_sems: &[Handle<Semaphore>],
    ) -> Result<(), GPUError> {
        let mut raw_wait_sems: Vec<vk::Semaphore> = Vec::with_capacity(32);
        for sem in wait_sems {
            raw_wait_sems.push(self.semaphores.get_ref(sem.clone()).unwrap().raw);
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

    #[cfg(feature = "dashi-openxr")]
    pub fn acquire_xr_image(
        &mut self,
        dsp: &mut Display,
    ) -> Result<(u32, xr::FrameState), GPUError> {
        let state = dsp
            .xr_waiter
            .wait()
            .map_err(|_| GPUError::LibraryError())?;
        dsp.xr_stream
            .begin()
            .map_err(|_| GPUError::LibraryError())?;
        let idx = dsp
            .xr_swapchain
            .acquire_image()
            .map_err(|_| GPUError::LibraryError())?;
        dsp.xr_swapchain
            .wait_image(xr::Duration::INFINITE)
            .map_err(|_| GPUError::LibraryError())?;
        Ok((idx, state))
    }

    #[cfg(feature = "dashi-openxr")]
    pub fn present_xr_display(
        &mut self,
        dsp: &mut Display,
        state: xr::FrameState,
    ) -> Result<(), GPUError> {
        dsp.xr_swapchain
            .release_image()
            .map_err(|_| GPUError::LibraryError())?;
        dsp.xr_stream
            .end(state.predicted_display_time, xr::EnvironmentBlendMode::OPAQUE, &[])
            .map_err(|_| GPUError::LibraryError())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    #[test]
    #[serial]
    fn test_context() {
        let ctx = Context::headless(&Default::default());
        assert!(ctx.is_ok());
        ctx.unwrap().destroy();
    }

    #[test]
    #[serial]
    fn test_buffer() {
        let c_buffer_size = 1280;
        let c_test_val = 8 as u8;
        let mut ctx = Context::headless(&Default::default()).unwrap();

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
        ctx.destroy();
    }

    #[test]
    #[serial]
    fn test_image() {
        let c_test_dim: [u32; 3] = [1280, 1024, 1];
        let c_format = Format::RGBA8;
        let c_mip_levels = 1;
        let c_test_val = 8 as u8;
        let initial_data =
            vec![c_test_val as u8; (c_test_dim[0] * c_test_dim[1] * c_test_dim[2] * 4) as usize];
        let mut ctx = Context::headless(&Default::default()).unwrap();
        let image_res = ctx.make_image(&ImageInfo {
            debug_name: "Test Image",
            dim: c_test_dim,
            format: c_format,
            mip_levels: c_mip_levels,
            initial_data: Some(&initial_data),
            ..Default::default()
        });

        assert!(image_res.is_ok());
        let image = image_res.unwrap();
        ctx.destroy_image(image);
        ctx.destroy();
    }

    #[test]
    #[serial]
    fn test_headless_context_creation() {
        // headless() should succeed...
        let ctx = Context::headless(&ContextInfo::default());
        assert!(ctx.is_ok(), "Context::headless() failed to create");
        let mut ctx = ctx.unwrap();

        // ...and never initialize any SDL bits in windowed builds
        #[cfg(feature = "dashi-sdl2")]
        {
            assert!(
                ctx.sdl_video.is_none(),
                "SDL video subsystem must be None in headless"
            );
        }

        // Core Vulkan ops still work:
        let buf = ctx
            .make_buffer(&BufferInfo {
                debug_name: "headless-buffer",
                byte_size: 128,
                visibility: MemoryVisibility::CpuAndGpu,
                ..Default::default()
            })
            .expect("make_buffer failed in headless mode");
        ctx.destroy_buffer(buf);

        // And we can clean up without panicking
        ctx.destroy();
    }

    #[test]
    #[serial]
    #[cfg(not(feature = "dashi-openxr"))]
    fn test_headless_rejects_display() {
        let mut ctx =
            Context::headless(&ContextInfo::default()).expect("headless() should succeed");

        // Try to call the windowed API -- should panic or error
        let info = DisplayInfo {
            window: WindowInfo {
                title: "nope".to_string(),
                size: [64, 64],
                resizable: false,
            },
            vsync: false,
            buffering: WindowBuffering::Double,
        };

        let err = ctx.make_display(&info).err().unwrap();

        assert!(
            matches!(err, GPUError::HeadlessDisplayNotSupported),
            "expected HeadlessDisplayNotSupported, got {:?}",
            err
        );

        ctx.destroy();
    }

    #[test]
    #[serial]
    fn compute_test() {
        // The GPU context that holds all the data.
        let mut ctx = Context::headless(&Default::default()).unwrap();

        // Make the bind group layout. This describes the bindings into a shader.
        let bg_layout = ctx
            .make_bind_group_layout(&BindGroupLayoutInfo {
                debug_name: "Hello Compute BG Layout",
                shaders: &[ShaderInfo {
                    shader_type: ShaderType::Compute,
                    variables: &[
                        BindGroupVariable {
                            var_type: BindGroupVariableType::Storage,
                            binding: 0,
                            count: 2048,
                        },
                        BindGroupVariable {
                            var_type: BindGroupVariableType::Storage,
                            binding: 1,
                            count: 2048,
                        },
                        BindGroupVariable {
                            var_type: BindGroupVariableType::DynamicUniform,
                            binding: 2,
                            count: 1,
                        },
                    ],
                }],
            })
            .unwrap();

        // Make a pipeline layout. This describes a graphics pipeline's state.
        let pipeline_layout = ctx
            .make_compute_pipeline_layout(&ComputePipelineLayoutInfo {
                bg_layouts: [Some(bg_layout), None, None, None],
                shader: &PipelineShaderInfo {
                    stage: ShaderType::Compute,
                    spirv: inline_spirv::inline_spirv!(
                        r#"
#version 450

layout(local_size_x = 1) in;

layout(binding = 0) buffer InputBuffer {
    float inputData[];
};

layout(binding = 1) buffer OutputBuffer {
    float outputData[];
};

layout(binding = 2) uniform OutputBuffer {
    float num_to_add;
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    outputData[index] = inputData[index] + num_to_add;
}
"#,
                        comp
                    ),
                    specialization: &[],
                },
            })
            .expect("Unable to create Compute Pipeline Layout!");

        // Make a compute pipeline. This describes a compute pass.
        let pipeline = ctx
            .make_compute_pipeline(&ComputePipelineInfo {
                debug_name: "Compute",
                layout: pipeline_layout,
            })
            .unwrap();

        // Make dynamic allocator to use for dynamic buffers.
        let mut allocator = ctx.make_dynamic_allocator(&Default::default()).unwrap();
        const BUFF_SIZE: u32 = 2048 * std::mem::size_of::<f32>() as u32;
        let initial_data = vec![0; BUFF_SIZE as usize];
        let input = ctx
            .make_buffer(&BufferInfo {
                debug_name: "input_test",
                byte_size: BUFF_SIZE,
                visibility: MemoryVisibility::Gpu,
                usage: BufferUsage::STORAGE,
                initial_data: Some(&initial_data),
            })
            .unwrap();

        let output = ctx
            .make_buffer(&BufferInfo {
                debug_name: "output_test",
                byte_size: BUFF_SIZE,
                visibility: MemoryVisibility::CpuAndGpu,
                usage: BufferUsage::STORAGE,
                initial_data: Some(&initial_data),
            })
            .unwrap();

        // Make bind group what we want to bind to what was described in the Bind Group Layout.
        let bind_group = ctx
            .make_bind_group(&BindGroupInfo {
                debug_name: "Hello Compute BG",
                layout: bg_layout,
                bindings: &[
                    BindingInfo {
                        resource: ShaderResource::StorageBuffer(input),
                        binding: 0,
                    },
                    BindingInfo {
                        resource: ShaderResource::StorageBuffer(output),
                        binding: 1,
                    },
                    BindingInfo {
                        resource: ShaderResource::Dynamic(&allocator),
                        binding: 2,
                    },
                ],
                ..Default::default()
            })
            .unwrap();

        // Reset the allocator
        allocator.reset();

        // Begin recording commands
        let mut list = ctx.begin_command_list(&Default::default()).unwrap();

        // Bump alloc some data to write the triangle position to.
        let mut buf = allocator.bump().unwrap();
        buf.slice::<f32>()[0] = 5.0;

        list.dispatch_compute(Dispatch {
            compute: pipeline,
            workgroup_size: [BUFF_SIZE / std::mem::size_of::<f32>() as u32, 1, 1],
            bind_groups: [Some(bind_group), None, None, None],
            dynamic_buffers: [Some(buf), None, None, None],
            ..Default::default()
        });

        // Submit our recorded commands
        let fence = ctx.submit(&mut list, &Default::default()).unwrap();

        ctx.wait(fence).unwrap();

        let data = ctx.map_buffer::<f32>(output).unwrap();
        for entry in data {
            assert!(*entry == 5.0);
        }

        ctx.unmap_buffer(output).unwrap();
        ctx.destroy_dynamic_allocator(allocator);
        ctx.destroy();
    }

    #[test]
    #[serial]
    #[ignore = "requires descriptor indexing and compute pipeline support"]
    fn bindless_test() {
        // The GPU context that holds all the data.
        let mut ctx = Context::headless(&Default::default()).unwrap();

        // Make the bind group layout. This describes the bindless bindings.
        let bg_layout = ctx
            .make_bindless_bind_group_layout(&BindGroupLayoutInfo {
                debug_name: "Hello Compute Bindless",
                shaders: &[ShaderInfo {
                    shader_type: ShaderType::Compute,
                    variables: &[
                        BindGroupVariable {
                            var_type: BindGroupVariableType::Storage,
                            binding: 0,
                            count: 256,
                        },
                        BindGroupVariable {
                            var_type: BindGroupVariableType::Storage,
                            binding: 1,
                            count: 256,
                        },
                    ],
                }],
            })
            .unwrap();

        // Make the bind group layout. This describes the dynamic bindings
        let dyn_bg_layout = ctx
            .make_bind_group_layout(&BindGroupLayoutInfo {
                debug_name: "Hello Compute Dynamic Binding",
                shaders: &[ShaderInfo {
                    shader_type: ShaderType::Compute,
                    variables: &[BindGroupVariable {
                        var_type: BindGroupVariableType::DynamicUniform,
                        binding: 0,
                        count: 1,
                    }],
                }],
            })
            .unwrap();

        // Make a pipeline layout. This describes a graphics pipeline's state.
        let pipeline_layout = ctx
            .make_compute_pipeline_layout(&ComputePipelineLayoutInfo {
                bg_layouts: [Some(bg_layout), Some(dyn_bg_layout), None, None],
                shader: &PipelineShaderInfo {
                    stage: ShaderType::Compute,
                    spirv: inline_spirv::inline_spirv!(
                        r#"
#version 450
#extension GL_EXT_nonuniform_qualifier : enable 

layout(local_size_x = 1) in;

layout(set = 0,binding = 0) buffer InputBuffer {
    float data;
} c_input[];

layout(set = 0, binding = 1) buffer OutputBuffer {
    float data;
} c_output[];


layout(set = 1, binding = 0) uniform OutputBuffer {
    float num_to_add;
};

void main() {
    uint index = gl_GlobalInvocationID.x;
    c_output[index].data = c_input[index].data + num_to_add;
}
"#,
                        comp
                    ),
                    specialization: &[],
                },
            })
            .expect("Unable to create Compute Pipeline Layout!");

        // Make a compute pipeline. This describes a compute pass.
        let pipeline = ctx
            .make_compute_pipeline(&ComputePipelineInfo {
                debug_name: "Compute",
                layout: pipeline_layout,
            })
            .unwrap();

        // Make dynamic allocator to use for dynamic buffers.
        let mut allocator = ctx.make_dynamic_allocator(&Default::default()).unwrap();
        const BUFF_SIZE: u32 = std::mem::size_of::<f32>() as u32;
        let initial_data = vec![0; BUFF_SIZE as usize];

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();

        const NUM_BUFFS: u32 = 256;
        for i in 0..NUM_BUFFS {
            inputs.push(IndexedResource {
                resource: ShaderResource::StorageBuffer(
                    ctx.make_buffer(&BufferInfo {
                        debug_name: "input_test",
                        byte_size: BUFF_SIZE,
                        visibility: MemoryVisibility::Gpu,
                        usage: BufferUsage::STORAGE,
                        initial_data: Some(&initial_data),
                    })
                    .unwrap(),
                ),
                slot: i,
            });

            outputs.push(IndexedResource {
                resource: ShaderResource::StorageBuffer(
                    ctx.make_buffer(&BufferInfo {
                        debug_name: "input_test",
                        byte_size: BUFF_SIZE,
                        visibility: MemoryVisibility::CpuAndGpu,
                        usage: BufferUsage::STORAGE,
                        initial_data: Some(&initial_data),
                    })
                    .unwrap(),
                ),
                slot: i,
            });
        }

        // Make bind group what we want to bind to what was described in the Bind Group Layout.
        let bind_group = ctx
            .make_indexed_bind_group(&IndexedBindGroupInfo {
                debug_name: "Hello Compute Bindless",
                layout: bg_layout,
                bindings: &[
                    IndexedBindingInfo {
                        resources: &inputs,
                        binding: 0,
                    },
                    IndexedBindingInfo {
                        resources: &outputs,
                        binding: 1,
                    },
                ],
                ..Default::default()
            })
            .unwrap();

        let dynamic_bind_group = ctx
            .make_bind_group(&BindGroupInfo {
                debug_name: "Hello Compute Dynamic",
                layout: dyn_bg_layout,
                set: 1,
                bindings: &[BindingInfo {
                    resource: ShaderResource::Dynamic(&allocator),
                    binding: 0,
                }],
                ..Default::default()
            })
            .unwrap();

        // Reset the allocator
        allocator.reset();

        // Begin recording commands
        let mut list = ctx.begin_command_list(&Default::default()).unwrap();

        // Bump alloc some data to write the triangle position to.
        let mut buf = allocator.bump().unwrap();
        buf.slice::<f32>()[0] = 5.0;
        list.dispatch_compute(Dispatch {
            compute: pipeline,
            workgroup_size: [NUM_BUFFS as u32, 1, 1],
            bind_groups: [Some(bind_group), Some(dynamic_bind_group), None, None],
            dynamic_buffers: [None, Some(buf), None, None],
            ..Default::default()
        });

        // Submit our recorded commands
        let fence = ctx.submit(&mut list, &Default::default()).unwrap();

        ctx.wait(fence).unwrap();

        for out in outputs {
            match out.resource {
                ShaderResource::StorageBuffer(b) => {
                    let data = ctx.map_buffer::<f32>(b).unwrap();
                    assert!(data[0] == 5.0);
                    ctx.unmap_buffer(b).unwrap();
                }
                _ => {}
            }
        }

        ctx.destroy_dynamic_allocator(allocator);
        ctx.clean_up();
        ctx.destroy();
    }
}
