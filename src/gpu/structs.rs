use super::{
    BindGroupLayout, Buffer, ComputePipelineLayout, DynamicAllocator, GraphicsPipelineLayout,
    Image, ImageView, RenderPass, Sampler, SelectedDevice,
};
use crate::{utils::Handle, BindGroup, Semaphore};
use std::hash::{Hash, Hasher};

#[cfg(feature = "dashi-serde")]
use serde::{Deserialize, Serialize};

#[derive(Hash, Clone, Copy, Debug)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum MemoryVisibility {
    Gpu,
    CpuAndGpu,
}

#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum BufferUsage {
    ALL,
    VERTEX,
    INDEX,
    UNIFORM,
    STORAGE,
}

#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum Format {
    R8Sint,
    R8Uint,
    RGB8,
    BGRA8,
    BGRA8Unorm,
    #[default]
    RGBA8,
    RGBA8Unorm,
    RGBA32F,
    D24S8,
}

#[derive(Hash, Clone, Copy, Debug)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum WindowBuffering {
    Double,
    Triple,
}

#[derive(Hash, Debug, Copy, Clone, Default)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum BlendFactor {
    One,
    Zero,
    SrcColor,
    InvSrcColor,
    #[default]
    SrcAlpha,
    InvSrcAlpha,
    DstAlpha,
    InvDstAlpha,
    DstColor,
    InvDstColor,
    BlendFactor,
}

#[derive(Hash, Debug, Copy, Clone, Default)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum BlendOp {
    #[default]
    Add,
    Subtract,
    InvSubtract,
    Min,
    Max,
}

#[derive(Hash, Debug, Copy, Clone, Default)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum LoadOp {
    Load,
    Clear,
    #[default]
    DontCare,
}

#[derive(Hash, Debug, Copy, Clone, Default)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum StoreOp {
    Store,
    #[default]
    DontCare,
}

#[derive(Hash, Debug, Copy, Clone, Default)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum SampleCount {
    #[default]
    S1,
    S2,
}

#[derive(Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum BarrierPoint {
    DrawEnd,
    BlitRead,
    BlitWrite,
    Present,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum Filter {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum SamplerAddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum SamplerMipmapMode {
    Nearest,
    Linear,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum BorderColor {
    OpaqueBlack,
    OpaqueWhite,
    TransparentBlack,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
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
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct Extent {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Hash, Default, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct Rect2D {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, Default, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct FRect2D {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Default)]
pub struct ContextInfo {
    pub device: SelectedDevice,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct IndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: i32,
    pub first_instance: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct IndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

pub struct ImageInfo<'a> {
    pub debug_name: &'a str,
    pub dim: [u32; 3],
    pub layers: u32,
    pub format: Format,
    pub mip_levels: u32,
    pub initial_data: Option<&'a [u8]>,
}

impl<'a> Default for ImageInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: "",
            dim: [1280, 1024, 1],
            layers: 1,
            format: Format::RGBA8,
            mip_levels: 1,
            initial_data: None,
        }
    }
}

#[derive(Hash, Clone, Debug, Default, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum AspectMask {
    #[default]
    Color,
    Depth,
    Stencil,
    DepthStencil,
}

pub struct ImageViewInfo<'a> {
    pub debug_name: &'a str,
    pub img: Handle<Image>,
    pub layer: u32,
    pub mip_level: u32,
    pub aspect: AspectMask,
}

impl<'a> Default for ImageViewInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: "",
            img: Default::default(),
            layer: 0,
            mip_level: 0,
            aspect: Default::default(),
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
    pub usage: BufferUsage,
    pub num_allocations: u32,
    pub byte_size: u32,
    pub allocation_size: u32,
}

impl<'a> Default for DynamicAllocatorInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: "",
            byte_size: 1024 * 1024,
            usage: BufferUsage::UNIFORM,
            num_allocations: 2048,
            allocation_size: 256,
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

pub struct SubmitInfo<'a> {
    pub wait_sems: &'a [Handle<Semaphore>],
    pub signal_sems: &'a [Handle<Semaphore>],
}

impl<'a> Default for SubmitInfo<'a> {
    fn default() -> Self {
        Self {
            wait_sems: &[],
            signal_sems: &[],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AttachmentDescription {
    pub format: Format,
    pub samples: SampleCount,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub stencil_load_op: LoadOp,
    pub stencil_store_op: StoreOp,
}

impl Hash for AttachmentDescription {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.format.hash(state);
        self.samples.hash(state);
        self.load_op.hash(state);
        self.store_op.hash(state);
        self.stencil_load_op.hash(state);
        self.stencil_store_op.hash(state);
    }
}
impl Default for AttachmentDescription {
    fn default() -> Self {
        Self {
            samples: SampleCount::S1,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            stencil_load_op: LoadOp::DontCare,
            stencil_store_op: StoreOp::DontCare,
            format: Format::RGBA8,
        }
    }
}
#[derive(Hash, Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum BindGroupVariableType {
    Uniform,
    DynamicUniform,
    DynamicStorage,
    Storage,
    SampledImage,
    StorageImage,
}

#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum ShaderType {
    Vertex,
    Fragment,
    Compute,
    All,
}

#[derive(Hash, Clone, Debug)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct BindGroupVariable {
    pub var_type: BindGroupVariableType,
    pub binding: u32,
    pub count: u32,
}

impl Default for BindGroupVariable {
    fn default() -> Self {
        Self {
            var_type: BindGroupVariableType::Uniform,
            binding: Default::default(),
            count: 1,
        }
    }
}
#[derive(Hash, Clone, Debug)]
pub struct ShaderInfo<'a> {
    pub shader_type: ShaderType,
    pub variables: &'a [BindGroupVariable],
}

#[derive(Hash, Clone, Debug)]
pub struct BindGroupLayoutInfo<'a> {
    pub debug_name: &'a str,
    pub shaders: &'a [ShaderInfo<'a>],
}

#[derive(Hash, Clone, Debug)]
pub struct BindlessBindGroupLayoutInfo<'a> {
    pub debug_name: &'a str,
    pub sampler_binding: u32,
    pub sampler_length: u32,
    pub const_buffer_binding: u32,
    pub const_buffer_length: u32,
    pub mutable_buffer_binding: u32,
    pub mutable_buffer_length: u32,
}

impl<'a> Default for BindlessBindGroupLayoutInfo<'a> {
    fn default() -> Self {
        const DEFAULT_BINDLESS_LEN: u32 = 2048;
        Self {
            debug_name: "Default",
            sampler_binding: 0,
            sampler_length: DEFAULT_BINDLESS_LEN,
            const_buffer_binding: 1,
            const_buffer_length: DEFAULT_BINDLESS_LEN,
            mutable_buffer_binding: 2,
            mutable_buffer_length: DEFAULT_BINDLESS_LEN,
        }
    }
}

pub enum ShaderResource<'a> {
    Buffer(Handle<Buffer>),
    StorageBuffer(Handle<Buffer>),
    Dynamic(&'a DynamicAllocator),
    DynamicStorage(&'a DynamicAllocator),
    SampledImage(Handle<ImageView>, Handle<Sampler>),
}

pub struct BindingInfo<'a> {
    pub resource: ShaderResource<'a>,
    pub binding: u32,
}

pub struct IndexedResource<'a> {
    pub resource: ShaderResource<'a>,
    pub slot: u32,
}
pub struct IndexedBindingInfo<'a> {
    pub resources: &'a [IndexedResource<'a>],
    pub binding: u32,
}

impl<'a> Default for IndexedBindingInfo<'a> {
    fn default() -> Self {
        Self {
            resources: Default::default(),
            binding: Default::default(),
        }
    }
}

pub struct BindGroupUpdateInfo<'a> {
    pub bg: Handle<BindGroup>,
    pub bindings: &'a [IndexedBindingInfo<'a>],
}

pub struct IndexedBindGroupInfo<'a> {
    pub debug_name: &'a str,
    pub layout: Handle<BindGroupLayout>,
    pub bindings: &'a [IndexedBindingInfo<'a>],
    pub set: u32,
}

impl<'a> Default for IndexedBindGroupInfo<'a> {
    fn default() -> Self {
        Self {
            layout: Default::default(),
            bindings: &[],
            set: 0,
            debug_name: "",
        }
    }
}

pub struct BindGroupInfo<'a> {
    pub debug_name: &'a str,
    pub layout: Handle<BindGroupLayout>,
    pub bindings: &'a [BindingInfo<'a>],
    pub set: u32,
}

pub struct BindlessBindGroupInfo<'a> {
    pub debug_name: &'a str,
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
            debug_name: "",
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct Viewport {
    pub area: FRect2D,
    pub scissor: Rect2D,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Default for Viewport {
    fn default() -> Self {
        Self {
            area: FRect2D {
                x: 0.0,
                y: 0.0,
                w: 1024.0,
                h: 1024.0,
            },
            scissor: Rect2D {
                x: 0,
                y: 0,
                w: 1024,
                h: 1024,
            },
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}
#[derive(Hash, Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum Topology {
    TriangleList,
}

#[derive(Hash, Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum CullMode {
    None,
    Back,
}

#[derive(Hash, Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum VertexOrdering {
    CounterClockwise,
    Clockwise,
}

#[derive(Hash, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum DynamicState {
    Viewport,
    Scissor,
}

#[derive(Debug, Clone, Copy, Default)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct DepthInfo {
    pub should_test: bool,
    pub should_write: bool,
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct WriteMask {
    pub r: bool,
    pub g: bool,
    pub b: bool,
    pub a: bool,
}

impl Default for WriteMask {
    fn default() -> Self {
        Self {
            r: true,
            g: true,
            b: true,
            a: true,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct ColorBlendState {
    pub enable: bool,
    pub src_blend: BlendFactor,
    pub dst_blend: BlendFactor,
    pub blend_op: BlendOp,
    pub src_alpha_blend: BlendFactor,
    pub dst_alpha_blend: BlendFactor,
    pub alpha_blend_op: BlendOp,
    pub write_mask: WriteMask,
}

impl Default for ColorBlendState {
    fn default() -> Self {
        Self {
            enable: true,
            src_blend: BlendFactor::SrcAlpha,
            dst_blend: BlendFactor::InvSrcAlpha,
            blend_op: BlendOp::Add,
            src_alpha_blend: BlendFactor::SrcAlpha,
            dst_alpha_blend: BlendFactor::InvSrcAlpha,
            alpha_blend_op: BlendOp::Add,
            write_mask: Default::default(),
        }
    }
}
#[derive(Debug, Clone)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct GraphicsPipelineDetails {
    pub subpass: u8,
    pub color_blend_states: Vec<ColorBlendState>,
    pub topology: Topology,
    pub culling: CullMode,
    pub front_face: VertexOrdering,
    pub depth_test: Option<DepthInfo>,
    /// Pipeline states that will be configured dynamically at draw time.
    pub dynamic_states: Vec<DynamicState>,
}

impl Default for GraphicsPipelineDetails {
    fn default() -> Self {
        GraphicsPipelineDetails {
            topology: Topology::TriangleList,
            culling: CullMode::Back,
            front_face: VertexOrdering::Clockwise,
            depth_test: None,
            color_blend_states: vec![Default::default()],
            dynamic_states: Vec::new(),
            subpass: 0,
        }
    }
}

#[derive(Hash, Debug, Clone)]
pub struct SpecializationInfo<'a> {
    pub slot: usize,
    pub data: &'a [u8], // ConstSlice in Rust can be a reference slice
}

#[derive(Hash, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum ShaderPrimitiveType {
    Vec2,
    Vec3,
    Vec4,
    IVec4,
    UVec4,
}

#[derive(Hash, Debug, Clone)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct VertexEntryInfo {
    pub format: ShaderPrimitiveType,
    pub location: usize,
    pub offset: usize,
}

#[derive(Hash, Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum VertexRate {
    Vertex,
}

pub struct SubpassDependency {
    pub subpass_id: u32,
    pub attachment_id: u32,
    pub depth_id: u32,
}

pub struct SubpassDescription<'a> {
    pub color_attachments: &'a [AttachmentDescription],
    pub depth_stencil_attachment: Option<&'a AttachmentDescription>,
    pub subpass_dependencies: &'a [SubpassDependency],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClearValue {
    Color([f32; 4]),
    IntColor([i32; 4]),
    UintColor([u32; 4]),
    DepthStencil { depth: f32, stencil: u32 },
}

impl Hash for ClearValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            ClearValue::Color(vals) => {
                0u8.hash(state); // variant tag
                for v in vals {
                    v.to_bits().hash(state); // preserve NaNs and -0.0 vs 0.0
                }
            }
            ClearValue::IntColor(vals) => {
                1u8.hash(state);
                vals.hash(state);
            }
            ClearValue::UintColor(vals) => {
                2u8.hash(state);
                vals.hash(state);
            }
            ClearValue::DepthStencil { depth, stencil } => {
                3u8.hash(state);
                depth.to_bits().hash(state);
                stencil.hash(state);
            }
        }
    }
}
#[derive(Hash, Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum AttachmentType {
    Color,
    Depth,
}

#[derive(Clone, Copy, Debug)]
pub struct Attachment {
    pub img: Handle<ImageView>,
    pub clear: ClearValue,
}

impl Default for Attachment {
    fn default() -> Self {
        Self {
            img: Default::default(),
            clear: ClearValue::Color([0.0, 0.0, 0.0, 1.0]),
        }
    }
}

impl Hash for Attachment {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.img.hash(state);
        self.clear.hash(state);
    }
}

//#[derive(Clone, Default, Hash, PartialEq, Eq)]
//pub struct Subpass<'a> {
//    pub colors: &'a [Attachment],
//    pub depth: Option<Attachment>,
//}

pub struct RenderPassInfo<'a> {
    pub debug_name: &'a str,
    pub viewport: Viewport,
    pub subpasses: &'a [SubpassDescription<'a>],
}

#[derive(Hash, Debug, Clone)]
pub struct VertexDescriptionInfo<'a> {
    pub entries: &'a [VertexEntryInfo], // ConstSlice in Rust can be a reference slice
    pub stride: usize,
    pub rate: VertexRate,
}

#[derive(Hash, Debug, Clone)]
pub struct PipelineShaderInfo<'a> {
    pub stage: ShaderType,
    pub spirv: &'a [u32], // ConstSlice in Rust can be a reference slice
    pub specialization: &'a [SpecializationInfo<'a>], // ConstSlice as a reference slice
}

#[derive(Debug)]
pub struct ComputePipelineLayoutInfo<'a> {
    pub bg_layouts: [Option<Handle<BindGroupLayout>>; 4],
    pub shader: &'a PipelineShaderInfo<'a>,
}

#[derive(Debug)]
pub struct GraphicsPipelineLayoutInfo<'a> {
    pub debug_name: &'a str,
    pub vertex_info: VertexDescriptionInfo<'a>,
    pub bg_layouts: [Option<Handle<BindGroupLayout>>; 4],
    pub shaders: &'a [PipelineShaderInfo<'a>],
    pub details: GraphicsPipelineDetails,
}

pub struct ComputePipelineInfo<'a> {
    pub debug_name: &'a str,
    pub layout: Handle<ComputePipelineLayout>,
}

pub struct GraphicsPipelineInfo<'a> {
    pub debug_name: &'a str,
    pub layout: Handle<GraphicsPipelineLayout>,
    pub render_pass: Handle<RenderPass>,
    pub subpass_id: u8,
}

impl<'a> Default for GraphicsPipelineInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: Default::default(),
            layout: Default::default(),
            render_pass: Default::default(),
            subpass_id: 0,
        }
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
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

#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
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

/// Information used when creating an OpenXR display.
///
/// Currently no configuration options are required but the
/// struct exists for future expansion and a consistent API.
#[cfg(feature = "dashi-openxr")]
#[derive(Default)]
pub struct XrDisplayInfo;
