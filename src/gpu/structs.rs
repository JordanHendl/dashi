use super::{
    BindGroupLayout, Buffer, DynamicAllocator, GraphicsPipelineLayout, Image, ImageView,
    RenderPass, Sampler,
};
use crate::utils::Handle;
use std::hash::{Hash, Hasher};

#[derive(Hash, Clone, Copy, Debug)]
pub enum MemoryVisibility {
    Gpu,
    CpuAndGpu,
}

#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferUsage {
    VERTEX,
    INDEX,
    UNIFORM,
    STORAGE,
}

#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    RGB8,
    BGRA8,
    BGRA8Unorm,
    RGBA8,
    RGBA32F,
    D24S8,
}

#[derive(Hash, Clone, Copy, Debug)]
pub enum WindowBuffering {
    Double,
    Triple,
}

#[derive(Hash, Debug, Copy, Clone)]
pub enum LoadOp {
    Load,
    Clear,
    DontCare,
}

#[derive(Hash, Debug, Copy, Clone)]
pub enum StoreOp {
    Store,
    DontCare,
}

#[derive(Hash, Debug, Copy, Clone)]
pub enum SampleCount {
    S1,
    S2,
}

#[derive(Clone, Copy)]
pub enum BarrierPoint {
    DrawEnd,
    BlitRead,
    BlitWrite,
    Present,
}

#[derive(Debug, Clone, Copy)]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy)]
pub enum SamplerAddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
}

#[derive(Debug, Clone, Copy)]
pub enum SamplerMipmapMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy)]
pub enum BorderColor {
    OpaqueBlack,
    OpaqueWhite,
    TransparentBlack,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplerInfo {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub anisotropy_enable: bool,
    pub max_anisotropy: f32,
    pub border_color: BorderColor,
    pub unnormalized_coordinates: bool,
    pub compare_enable: bool,
    pub mipmap_mode: SamplerMipmapMode,
}

// Implementing Default for SamplerInfo
impl Default for SamplerInfo {
    fn default() -> Self {
        SamplerInfo {
            mag_filter: Filter::Linear,                 // Default to Linear filtering
            min_filter: Filter::Linear,                 // Default to Linear filtering
            address_mode_u: SamplerAddressMode::Repeat, // Default to Repeat wrapping
            address_mode_v: SamplerAddressMode::Repeat, // Default to Repeat wrapping
            address_mode_w: SamplerAddressMode::Repeat, // Default to Repeat wrapping
            anisotropy_enable: false,                   // disable anisotropy by default
            max_anisotropy: 1.0, // Default to a maximum anisotropy level of 1.0
            border_color: BorderColor::OpaqueBlack, // Default to opaque black border color
            unnormalized_coordinates: false, // Use normalized texture coordinates by default
            compare_enable: false, // Disable comparison by default
            mipmap_mode: SamplerMipmapMode::Linear, // Default to Linear mipmap filtering
        }
    }
}

#[derive(Debug, Hash, Default, Clone, Copy)]
pub struct Extent {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Hash, Default, Clone, Copy)]
pub struct Rect2D {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct FRect2D {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Hash, Default)]
pub struct ContextInfo {}

pub struct ImageInfo<'a> {
    pub debug_name: &'a str,
    pub dim: [u32; 3],
    pub format: Format,
    pub mip_levels: u32,
    pub initial_data: Option<&'a [u8]>,
}

pub struct ImageViewInfo<'a> {
    pub debug_name: &'a str,
    pub img: Handle<Image>,
    pub layer: u32,
    pub mip_level: u32,
}

impl<'a> Default for ImageViewInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: "",
            img: Default::default(),
            layer: 0,
            mip_level: 0,
        }
    }
}

#[derive(Hash, Clone, Copy, Debug)]
pub struct BufferInfo<'a> {
    pub debug_name: &'a str,
    pub byte_size: u32,
    pub visibility: MemoryVisibility,
    pub usage: BufferUsage,
    pub initial_data: Option<&'a [u8]>,
}

impl<'a> Default for BufferInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: "",
            byte_size: 1024,
            visibility: MemoryVisibility::CpuAndGpu,
            initial_data: None,
            usage: BufferUsage::UNIFORM,
        }
    }
}
#[derive(Hash, Clone, Copy, Debug)]
pub struct DynamicAllocatorInfo<'a> {
    pub debug_name: &'a str,
    pub byte_size: u32,
}

impl<'a> Default for DynamicAllocatorInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: "",
            byte_size: 1024 * 1024,
        }
    }
}

#[derive(Hash)]
pub struct CommandListInfo<'a> {
    pub debug_name: &'a str,
    pub should_cleanup: bool,
}

impl<'a> Default for CommandListInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: "",
            should_cleanup: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Attachment {
    pub view: Handle<ImageView>,
    pub samples: SampleCount,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub stencil_load_op: LoadOp,
    pub stencil_store_op: StoreOp,
    pub clear_color: [f32; 4],
}

impl Hash for Attachment {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.view.hash(state);
        self.samples.hash(state);
        self.load_op.hash(state);
        self.store_op.hash(state);
        self.stencil_load_op.hash(state);
        self.stencil_store_op.hash(state);

        for p in self.clear_color {
            (p as u32).hash(state);
        }
    }
}
impl Default for Attachment {
    fn default() -> Self {
        Self {
            view: Default::default(),
            samples: SampleCount::S1,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            stencil_load_op: LoadOp::DontCare,
            stencil_store_op: StoreOp::DontCare,
            clear_color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}
#[derive(Hash, Clone, Debug)]
pub enum BindGroupVariableType {
    Uniform,
    DynamicUniform,
    Storage,
    SampledImage,
    StorageImage,
}

#[derive(Hash, Clone, Debug)]
pub enum ShaderType {
    Vertex,
    Fragment,
    Compute,
}

#[derive(Hash, Clone, Debug)]
pub struct BindGroupVariable {
    pub var_type: BindGroupVariableType,
    pub binding: u32,
}

#[derive(Hash, Clone, Debug)]
pub struct ShaderInfo<'a> {
    pub shader_type: ShaderType,
    pub variables: &'a [BindGroupVariable],
}

#[derive(Hash, Clone, Debug)]
pub struct BindGroupLayoutInfo<'a> {
    pub shaders: &'a [ShaderInfo<'a>],
}

pub enum ShaderResource<'a> {
    Buffer(Handle<Buffer>),
    Dynamic(&'a DynamicAllocator),
    SampledImage(Handle<ImageView>, Handle<Sampler>),
}

pub struct BindingInfo<'a> {
    pub resource: ShaderResource<'a>,
    pub binding: u32,
}

pub struct BindGroupInfo<'a> {
    pub layout: Handle<BindGroupLayout>,
    pub bindings: &'a [BindingInfo<'a>],
    pub set: u32,
}

impl<'a> Default for BindGroupInfo<'a> {
    fn default() -> Self {
        Self {
            layout: Default::default(),
            bindings: &[],
            set: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Viewport {
    pub area: FRect2D,
    pub scissor: Rect2D,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            area: Default::default(),
            scissor: Default::default(),
            min_depth: Default::default(),
            max_depth: Default::default(),
        }
    }
}
#[derive(Hash, Debug, Clone, Copy)]
pub enum Topology {
    TriangleList,
}

#[derive(Hash, Debug, Clone, Copy)]
pub enum CullMode {
    Back,
}

#[derive(Hash, Debug, Clone, Copy)]
pub enum VertexOrdering {
    Clockwise,
}

#[derive(Debug, Clone, Copy)]
pub struct GraphicsPipelineDetails {
    pub topology: Topology,
    pub culling: CullMode,
    pub front_face: VertexOrdering,
    pub depth_test: bool,
}

impl Default for GraphicsPipelineDetails {
    fn default() -> Self {
        GraphicsPipelineDetails {
            topology: Topology::TriangleList,
            culling: CullMode::Back,
            front_face: VertexOrdering::Clockwise,
            depth_test: false,
        }
    }
}

#[derive(Hash, Debug)]
pub struct SpecializationInfo<'a> {
    pub slot: usize,
    pub data: &'a [u8], // ConstSlice in Rust can be a reference slice
}

#[derive(Hash, Debug, Clone, Copy)]
pub enum ShaderPrimitiveType {
    Vec2,
    Vec4,
}

#[derive(Hash, Debug)]
pub struct VertexEntryInfo {
    pub format: ShaderPrimitiveType,
    pub location: usize,
    pub offset: usize,
}

#[derive(Hash, Debug, Clone, Copy)]
pub enum VertexRate {
    Vertex,
}

pub struct RenderPassInfo<'a> {
    pub viewport: Viewport,
    pub pipeline_layout: Handle<GraphicsPipelineLayout>,
    pub color_attachments: &'a [Attachment],
    pub depth_stencil_attachment: Option<&'a Attachment>,
}

#[derive(Hash, Debug)]
pub struct VertexDescriptionInfo<'a> {
    pub entries: &'a [VertexEntryInfo], // ConstSlice in Rust can be a reference slice
    pub stride: usize,
    pub rate: VertexRate,
}

#[derive(Hash, Debug)]
pub struct PipelineShaderInfo<'a> {
    pub stage: ShaderType,
    pub spirv: &'a [u32], // ConstSlice in Rust can be a reference slice
    pub specialization: &'a [SpecializationInfo<'a>], // ConstSlice as a reference slice
}

#[derive(Debug)]
pub struct GraphicsPipelineLayoutInfo<'a> {
    pub vertex_info: VertexDescriptionInfo<'a>,
    pub bg_layout: Handle<BindGroupLayout>,
    pub shaders: &'a [PipelineShaderInfo<'a>],
    pub details: GraphicsPipelineDetails,
}

pub struct GraphicsPipelineInfo {
    pub layout: Handle<GraphicsPipelineLayout>,
    pub render_pass: Handle<RenderPass>,
}

#[derive(Debug, Clone)]
pub struct WindowInfo {
    pub title: String,
    pub size: [u32; 2],
    pub resizable: bool,
}

impl Default for WindowInfo {
    fn default() -> Self {
        Self {
            title: "Dashi".to_string(),
            size: [1280, 1024],
            resizable: false,
        }
    }
}

pub struct DisplayInfo {
    pub window: WindowInfo,
    pub vsync: bool,
    pub buffering: WindowBuffering,
}

impl Default for DisplayInfo {
    fn default() -> Self {
        Self {
            window: Default::default(),
            vsync: true,
            buffering: WindowBuffering::Double,
        }
    }
}
