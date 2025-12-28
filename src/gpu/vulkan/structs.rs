use super::{
    BindTableLayout, Buffer, ComputePipelineLayout, DynamicAllocatorState, GraphicsPipelineLayout,
    Image, RenderPass, Sampler, SelectedDevice, SubpassSampleInfo,
};
use crate::{utils::Handle, BindTable, CommandQueue, Semaphore};
use bitflags::bitflags;
use std::collections::hash_map::{DefaultHasher, Entry};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::{c_void, CStr};
use std::hash::{Hash, Hasher};

use bytemuck::{Pod, Zeroable};
#[cfg(feature = "dashi-serde")]
use serde::{Deserialize, Serialize};

#[inline]
fn hash64<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

fn hash_f32<H: Hasher>(value: f32, state: &mut H) {
    state.write(&value.to_bits().to_le_bytes());
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct DebugMessageSeverity: u32 {
        const VERBOSE = 0x1;
        const INFO = 0x2;
        const WARNING = 0x4;
        const ERROR = 0x8;
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct DebugMessageType: u32 {
        const GENERAL = 0x1;
        const VALIDATION = 0x2;
        const PERFORMANCE = 0x4;
    }
}

pub type DebugMessengerCallback = unsafe extern "system" fn(
    severity: DebugMessageSeverity,
    message_type: DebugMessageType,
    message: &CStr,
    user_data: *mut c_void,
) -> bool;

#[derive(Clone, Copy)]
pub struct DebugMessengerCreateInfo {
    pub message_severity: DebugMessageSeverity,
    pub message_type: DebugMessageType,
    pub user_callback: DebugMessengerCallback,
    pub user_data: *mut c_void,
}

impl Default for DebugMessengerCreateInfo {
    fn default() -> Self {
        Self {
            message_severity: DebugMessageSeverity::WARNING | DebugMessageSeverity::ERROR,
            message_type: DebugMessageType::GENERAL
                | DebugMessageType::VALIDATION
                | DebugMessageType::PERFORMANCE,
            user_callback: dummy_debug_callback,
            user_data: std::ptr::null_mut(),
        }
    }
}

unsafe extern "system" fn dummy_debug_callback(
    _severity: DebugMessageSeverity,
    _message_type: DebugMessageType,
    _message: &CStr,
    _user_data: *mut c_void,
) -> bool {
    false
}

#[repr(C)]
#[derive(Default, Hash, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum MemoryVisibility {
    Gpu,
    #[default]
    CpuAndGpu,
}

#[repr(C)]
#[derive(Default, Hash, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum BufferUsage {
    #[default]
    ALL,
    VERTEX,
    INDEX,
    UNIFORM,
    STORAGE,
}

#[repr(C)]
#[derive(Default, Hash, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum QueueType {
    #[default]
    Graphics,
    Compute,
    Transfer,
}

#[repr(C)]
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

#[repr(C)]
#[derive(Hash, Clone, Copy, Debug)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum WindowBuffering {
    Double,
    Triple,
}

#[repr(C)]
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

#[repr(C)]
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

#[repr(C)]
#[derive(Hash, Debug, Copy, Clone, Default)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum LoadOp {
    Load,
    Clear,
    #[default]
    DontCare,
}

#[repr(C)]
#[derive(Hash, Debug, Copy, Clone, Default)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum StoreOp {
    Store,
    #[default]
    DontCare,
}

#[repr(C)]
#[derive(Hash, Debug, Copy, Clone, Default, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum SampleCount {
    #[default]
    S1,
    S2,
    S4,
    S8,
    S16,
    S32,
    S64,
}

impl SampleCount {
    pub fn from_samples(samples: u32) -> Self {
        // Clamp to range [1, 64]
        let clamped = samples.clamp(1, 64);

        // Round to nearest power of two
        let next_pow2 = clamped.next_power_of_two();
        let prev_pow2 = next_pow2 >> 1;

        let closest_pow2 = if next_pow2 - clamped < clamped - prev_pow2 {
            next_pow2
        } else {
            prev_pow2.max(1)
        };

        match closest_pow2 {
            1 => Self::S1,
            2 => Self::S2,
            4 => Self::S4,
            8 => Self::S8,
            16 => Self::S16,
            32 => Self::S32,
            _ => Self::S64, // covers 64 and above
        }
    }

    pub fn as_u32(self) -> u32 {
        match self {
            Self::S1 => 1,
            Self::S2 => 2,
            Self::S4 => 4,
            Self::S8 => 8,
            Self::S16 => 16,
            Self::S32 => 32,
            Self::S64 => 64,
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum BarrierPoint {
    DrawEnd,
    BlitRead,
    BlitWrite,
    Present,
}

#[repr(C)]
#[derive(Zeroable, Default, Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum Filter {
    #[default]
    Nearest,
    Linear,
}

unsafe impl Pod for Filter {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct XrSwapchainImage {
    raw_handle: u64,
}

impl XrSwapchainImage {
    pub fn raw_handle(&self) -> u64 {
        self.raw_handle
    }

    pub(crate) fn from_raw(raw_handle: u64) -> Self {
        Self { raw_handle }
    }
}
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum SamplerAddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum SamplerMipmapMode {
    Nearest,
    Linear,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum BorderColor {
    OpaqueBlack,
    OpaqueWhite,
    TransparentBlack,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct SubresourceRange {
    pub base_mip: u32,
    pub level_count: u32,
    pub base_layer: u32,
    pub layer_count: u32,
}

impl Default for SubresourceRange {
    fn default() -> Self {
        Self {
            base_mip: Default::default(),
            level_count: 1,
            base_layer: Default::default(),
            layer_count: 1,
        }
    }
}

impl SubresourceRange {
    pub fn new(base_mip: u32, level_count: u32, base_layer: u32, layer_count: u32) -> Self {
        Self {
            base_mip,
            level_count,
            base_layer,
            layer_count,
        }
    }
}

#[repr(C)]
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

#[repr(C)]
#[derive(Debug, Hash, Default, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct Extent {
    pub width: u32,
    pub height: u32,
}
#[repr(C)]
#[derive(Default, Clone, Copy, Debug, Pod, Zeroable, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct Rect2D {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct FRect2D {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Hash for FRect2D {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_f32(self.x, state);
        hash_f32(self.y, state);
        hash_f32(self.w, state);
        hash_f32(self.h, state);
    }
}

#[cfg(feature = "webgpu")]
#[derive(Clone)]
pub enum WebSurfaceInfo {
    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    CanvasId(String),
    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    Canvas(web_sys::HtmlCanvasElement),
    #[cfg(not(target_arch = "wasm32"))]
    Unsupported,
}

#[repr(C)]
pub struct ContextInfo {
    pub device: SelectedDevice,
    #[cfg(feature = "webgpu")]
    pub web_surface: Option<WebSurfaceInfo>,
}

impl Default for ContextInfo {
    fn default() -> Self {
        Self {
            device: SelectedDevice::default(),
            #[cfg(feature = "webgpu")]
            web_surface: None,
        }
    }
}

/// A snapshot of hardware limits exposed through the [`Context`].
///
/// All fields are expressed in API-agnostic terms so they can be used directly
/// when deciding how many resources to bind or how large buffers can be in
/// shader-visible memory.
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct ContextLimits {
    /// Maximum number of sampler bindings that a single shader stage can see
    /// across all currently bound bind tables.
    pub max_sampler_array_len: u32,
    /// Maximum number of sampled textures that a single shader stage can read
    /// from at once.
    pub max_sampled_texture_array_len: u32,
    /// Maximum number of storage textures that a single shader stage can write
    /// to or read from.
    pub max_storage_texture_array_len: u32,
    /// Maximum size, in bytes, of any single uniform/constant buffer binding
    /// that is visible to shaders.
    pub max_uniform_buffer_range: u32,
    /// Maximum size, in bytes, of any single storage buffer binding that is
    /// visible to shaders.
    pub max_storage_buffer_range: u32,
    /// Maximum total size, in bytes, of push constant data provided to shaders
    /// via [`crate::gpu::cmd::CommandStream::push_constants`].
    pub max_push_constant_size: u32,
    /// Maximum number of color attachments that can be written in a single
    /// render pass.
    pub max_color_attachments: u32,
    /// Maximum number of bind tables (descriptor sets) that can be bound
    /// simultaneously across all pipeline stages.
    pub max_bound_bind_tables: u32,
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ImageInfo<'a> {
    pub debug_name: &'a str,
    pub dim: [u32; 3],
    pub layers: u32,
    pub format: Format,
    pub mip_levels: u32,
    pub samples: SampleCount,
    pub initial_data: Option<&'a [u8]>,
}

impl Hash for ImageInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dim.hash(state);
        self.layers.hash(state);
        self.format.hash(state);
        self.mip_levels.hash(state);
        self.samples.hash(state);
    }
}

impl<'a> Default for ImageInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: "",
            dim: [1280, 1024, 1],
            layers: 1,
            format: Format::RGBA8,
            mip_levels: 1,
            samples: SampleCount::S1,
            initial_data: None,
        }
    }
}

#[repr(C)]
#[derive(Zeroable, Hash, Clone, Debug, Default, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum AspectMask {
    #[default]
    Color,
    Depth,
    Stencil,
    DepthStencil,
}

unsafe impl Pod for AspectMask {}

#[repr(C)]
#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct ImageView {
    pub img: Handle<Image>,
    pub range: SubresourceRange,
    pub aspect: AspectMask,
}

impl Default for ImageView {
    fn default() -> Self {
        Self {
            img: Default::default(),
            range: Default::default(),
            aspect: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BufferInfo<'a> {
    pub debug_name: &'a str,
    pub byte_size: u32,
    pub visibility: MemoryVisibility,
    pub usage: BufferUsage,
    pub initial_data: Option<&'a [u8]>,
}

impl Hash for BufferInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.byte_size.hash(state);
        self.visibility.hash(state);
        self.usage.hash(state);
    }
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
#[derive(Clone, Copy, Debug)]
pub struct DynamicAllocatorInfo<'a> {
    pub debug_name: &'a str,
    pub usage: BufferUsage,
    pub num_allocations: u32,
    pub byte_size: u32,
    pub allocation_size: u32,
}

impl Hash for DynamicAllocatorInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.usage.hash(state);
        self.num_allocations.hash(state);
        self.byte_size.hash(state);
        self.allocation_size.hash(state);
    }
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

#[derive(Clone, Copy)]
pub struct CommandQueueInfo<'a> {
    pub debug_name: &'a str,
    pub should_cleanup: bool,
    pub queue_type: QueueType,
}

#[derive(Default)]
pub struct CommandQueueInfo2<'a> {
    pub debug_name: &'a str,
    pub parent: Option<&'a CommandQueue>,
    pub queue_type: QueueType,
}

impl Hash for CommandQueueInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.should_cleanup.hash(state);
        self.queue_type.hash(state);
    }
}

impl Hash for CommandQueueInfo2<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let parent_ptr = self.parent.map(|p| p as *const CommandQueue as usize);
        parent_ptr.hash(state);
        self.queue_type.hash(state);
    }
}

impl<'a> Default for CommandQueueInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: "",
            should_cleanup: true,
            queue_type: QueueType::Graphics,
        }
    }
}

pub struct SubmitInfo<'a> {
    pub wait_sems: &'a [Handle<Semaphore>],
    pub signal_sems: &'a [Handle<Semaphore>],
}

impl Hash for SubmitInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.wait_sems.hash(state);
        self.signal_sems.hash(state);
    }
}

impl<'a> Default for SubmitInfo<'a> {
    fn default() -> Self {
        Self {
            wait_sems: &[],
            signal_sems: &[],
        }
    }
}

#[repr(C)]
#[derive(Pod, Zeroable, Default, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubmitInfo2 {
    pub wait_sems: [Handle<Semaphore>; 4],
    pub signal_sems: [Handle<Semaphore>; 4],
}

#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
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

#[repr(u8)]
#[derive(Hash, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub enum BindTableVariableType {
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

#[derive(Hash, Clone, Debug, Eq, PartialEq)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct BindTableVariable {
    pub var_type: BindTableVariableType,
    pub binding: u32,
    pub count: u32,
}

impl Default for BindTableVariable {
    fn default() -> Self {
        Self {
            var_type: BindTableVariableType::Uniform,
            binding: Default::default(),
            count: 1,
        }
    }
}

#[derive(Hash, Clone, Debug)]
pub struct ShaderInfo<'a> {
    pub shader_type: ShaderType,
    pub variables: &'a [BindTableVariable],
}

#[derive(Clone, Debug)]
pub struct BindTableLayoutInfo<'a> {
    pub debug_name: &'a str,
    pub shaders: &'a [ShaderInfo<'a>],
}

impl Hash for BindTableLayoutInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_from_shaders(self.shaders).hash(state);
    }
}

impl<'a> Default for BindTableLayoutInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: "Default",
            shaders: &[],
        }
    }
}

#[inline]
fn stage_bit(s: ShaderType) -> u8 {
    match s {
        ShaderType::Vertex => 1 << 0,
        ShaderType::Fragment => 1 << 1,
        ShaderType::Compute => 1 << 2,
        ShaderType::All => 0xFF,
    }
}

/// Internal normalized key so layout hashes are independent of declaration order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct NormalizedBinding {
    binding: u32,
    var_ty: u8,
    count: u32,
    stages: u8, // aggregated bitmask across all shaders that reference this binding
}

fn hash_from_shaders(shaders: &[ShaderInfo<'_>]) -> u64 {
    // Aggregate by (binding, var_ty, count)
    let mut agg: BTreeMap<(u32, u8, u32), u8> = BTreeMap::new();

    for sh in shaders {
        let bit = stage_bit(sh.shader_type);
        for v in sh.variables {
            let k = (v.binding, v.var_type.clone() as u8, v.count);
            agg.entry(k)
                .and_modify(|stages| *stages |= bit)
                .or_insert(bit);
        }
    }

    // Normalize into a sorted vector (BTreeMap already sorted by key)
    let norm: Vec<NormalizedBinding> = agg
        .into_iter()
        .map(|((binding, var_ty, count), stages)| NormalizedBinding {
            binding,
            var_ty,
            count,
            stages,
        })
        .collect();

    // Hash the normalized list
    hash64(&norm)
}

/// Stable, order-independent hash for a BindTableLayoutInfo (ignores `debug_name`).
pub fn hash_bind_table_layout_info(info: &BindTableLayoutInfo<'_>) -> u64 {
    hash_from_shaders(info.shaders)
}

pub(crate) fn layout_binding_requirements(
    variables: &[BindTableVariable],
) -> Option<HashMap<u32, (BindTableVariableType, u32)>> {
    let mut requirements = HashMap::new();

    for var in variables {
        match requirements.entry(var.binding) {
            Entry::Vacant(entry) => {
                entry.insert((var.var_type, var.count));
            }
            Entry::Occupied(entry) => {
                let (expected_type, expected_count) = *entry.get();
                if expected_type != var.var_type || expected_count != var.count {
                    return None;
                }
            }
        }
    }

    Some(requirements)
}

pub(crate) fn resource_var_type(resource: &ShaderResource) -> BindTableVariableType {
    match resource {
        ShaderResource::Buffer(_) | ShaderResource::ConstBuffer(_) => {
            BindTableVariableType::Uniform
        }
        ShaderResource::Dynamic(_) => BindTableVariableType::DynamicUniform,
        ShaderResource::DynamicStorage(_) => BindTableVariableType::DynamicStorage,
        ShaderResource::StorageBuffer(_) => BindTableVariableType::Storage,
        ShaderResource::SampledImage(_, _) => BindTableVariableType::SampledImage,
    }
}

pub(crate) fn indexed_bindings_compatible_with_layout(
    variables: &[BindTableVariable],
    bindings: &[IndexedBindingInfo],
) -> bool {
    let Some(requirements) = layout_binding_requirements(variables) else {
        return false;
    };

    let mut seen_slots: HashMap<u32, HashSet<u32>> = HashMap::new();

    for binding in bindings {
        let Some((expected_type, expected_count)) = requirements.get(&binding.binding) else {
            return false;
        };

        let slots = seen_slots.entry(binding.binding).or_default();

        for res in binding.resources {
            if resource_var_type(&res.resource) != *expected_type {
                return false;
            }

            if res.slot >= *expected_count {
                return false;
            }

            if !slots.insert(res.slot) {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod layout_validation_tests {
    use super::*;

    #[test]
    fn accepts_matching_bind_table_bindings() {
        let layout_vars = vec![
            BindTableVariable {
                var_type: BindTableVariableType::Uniform,
                binding: 0,
                count: 1,
            },
            BindTableVariable {
                var_type: BindTableVariableType::Storage,
                binding: 1,
                count: 3,
            },
        ];

        let bindings = [
            IndexedBindingInfo {
                binding: 0,
                resources: &[IndexedResource {
                    slot: 0,
                    resource: ShaderResource::Buffer(BufferView::new(Handle::new(0, 0))),
                }],
            },
            IndexedBindingInfo {
                binding: 1,
                resources: &[IndexedResource {
                    slot: 0,
                    resource: ShaderResource::StorageBuffer(BufferView::new(Handle::new(1, 0))),
                }],
            },
        ];

        assert!(indexed_bindings_compatible_with_layout(
            &layout_vars,
            &bindings
        ));
    }

    #[test]
    fn rejects_mismatched_binding_type() {
        let layout_vars = vec![BindTableVariable {
            var_type: BindTableVariableType::SampledImage,
            binding: 0,
            count: 1,
        }];

        let bindings = [IndexedBindingInfo {
            binding: 0,
            resources: &[IndexedResource {
                slot: 0,
                resource: ShaderResource::Buffer(BufferView::new(Handle::new(0, 0))),
            }],
        }];

        assert!(!indexed_bindings_compatible_with_layout(
            &layout_vars,
            &bindings
        ));
    }

    #[test]
    fn validates_indexed_slots_and_types() {
        let layout_vars = vec![BindTableVariable {
            var_type: BindTableVariableType::SampledImage,
            binding: 2,
            count: 2,
        }];

        let valid_bindings = [IndexedBindingInfo {
            resources: &[IndexedResource {
                resource: ShaderResource::SampledImage(
                    ImageView {
                        img: Handle::new(0, 0),
                        range: Default::default(),
                        aspect: AspectMask::Color,
                    },
                    Handle::new(1, 0),
                ),
                slot: 1,
            }],
            binding: 2,
        }];

        assert!(indexed_bindings_compatible_with_layout(
            &layout_vars,
            &valid_bindings,
        ));

        let invalid_bindings = [IndexedBindingInfo {
            resources: &[IndexedResource {
                resource: ShaderResource::SampledImage(
                    ImageView {
                        img: Handle::new(0, 0),
                        range: Default::default(),
                        aspect: AspectMask::Color,
                    },
                    Handle::new(1, 0),
                ),
                slot: 3,
            }],
            binding: 2,
        }];

        assert!(!indexed_bindings_compatible_with_layout(
            &layout_vars,
            &invalid_bindings,
        ));
    }
}

#[derive(Debug, Clone, Copy, Hash)]
pub struct BufferView {
    pub handle: Handle<Buffer>,
    pub size: u64,
    pub offset: u64,
}

impl BufferView {
    pub fn new(handle: Handle<Buffer>) -> Self {
        Self {
            handle,
            size: 0,
            offset: 0,
        }
    }
}

impl From<Handle<Buffer>> for BufferView {
    fn from(value: Handle<Buffer>) -> Self {
        BufferView::new(value)
    }
}

#[derive(Debug, Clone, Hash)]
pub enum ShaderResource {
    Buffer(BufferView),
    ConstBuffer(BufferView),
    StorageBuffer(BufferView),
    Dynamic(DynamicAllocatorState),
    DynamicStorage(DynamicAllocatorState),
    SampledImage(ImageView, Handle<Sampler>),
}

#[derive(Debug, Clone, Hash)]
pub struct IndexedResource {
    pub resource: ShaderResource,
    pub slot: u32,
}

#[derive(Hash, Clone)]
pub struct IndexedBindingInfo<'a> {
    pub resources: &'a [IndexedResource],
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

#[derive(Hash)]
pub struct BindTableUpdateInfo<'a> {
    pub table: Handle<BindTable>,
    pub bindings: &'a [IndexedBindingInfo<'a>],
}

#[derive(Clone)]
pub struct BindTableInfo<'a> {
    pub debug_name: &'a str,
    pub layout: Handle<BindTableLayout>,
    pub bindings: &'a [IndexedBindingInfo<'a>],
    pub set: u32,
}

impl Hash for BindTableInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.layout.hash(state);
        self.bindings.hash(state);
        self.set.hash(state);
    }
}

impl<'a> Default for BindTableInfo<'a> {
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

impl Hash for Viewport {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.area.hash(state);
        self.scissor.hash(state);
        hash_f32(self.min_depth, state);
        hash_f32(self.max_depth, state);
    }
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

#[derive(Debug, Clone, Copy, Default, Hash)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
pub struct DepthInfo {
    pub should_test: bool,
    pub should_write: bool,
}

#[derive(Debug, Clone, Copy, Hash)]
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

#[derive(Debug, Clone, Copy, Hash)]
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
    pub sample_count: SampleCount,
    pub min_sample_shading: f32,
    /// Pipeline states that will be configured dynamically at draw time.
    pub dynamic_states: Vec<DynamicState>,
}

impl Hash for GraphicsPipelineDetails {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.subpass.hash(state);
        self.color_blend_states.hash(state);
        self.topology.hash(state);
        self.culling.hash(state);
        self.front_face.hash(state);
        self.depth_test.hash(state);
        self.sample_count.hash(state);
        hash_f32(self.min_sample_shading, state);
        self.dynamic_states.hash(state);
    }
}

impl Default for GraphicsPipelineDetails {
    fn default() -> Self {
        GraphicsPipelineDetails {
            topology: Topology::TriangleList,
            culling: CullMode::None,
            front_face: VertexOrdering::Clockwise,
            depth_test: None,
            color_blend_states: vec![Default::default()],
            sample_count: SampleCount::S1,
            min_sample_shading: 0.0,
            dynamic_states: vec![DynamicState::Viewport, DynamicState::Scissor],
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

#[derive(Hash, Clone, Copy, Debug)]
#[cfg_attr(feature = "dashi-serde", derive(Serialize, Deserialize))]
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

impl Hash for SubpassDescription<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.color_attachments.hash(state);
        self.depth_stencil_attachment.hash(state);
        self.subpass_dependencies.hash(state);
    }
}

#[repr(C)]
#[derive(Zeroable, Debug, Clone, Copy, PartialEq)]
pub enum ClearValue {
    Color([f32; 4]),
    IntColor([i32; 4]),
    UintColor([u32; 4]),
    DepthStencil { depth: f32, stencil: u32 },
}

unsafe impl Pod for ClearValue {}

#[cfg(feature = "dashi-serde")]
impl Serialize for ClearValue {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        match self {
            ClearValue::Color(v) => {
                let mut st = s.serialize_struct("ClearValue", 2)?;
                st.serialize_field("kind", "color")?;
                st.serialize_field("value", v)?;
                st.end()
            }
            ClearValue::IntColor(v) => {
                let mut st = s.serialize_struct("ClearValue", 2)?;
                st.serialize_field("kind", "int_color")?;
                st.serialize_field("value", v)?;
                st.end()
            }
            ClearValue::UintColor(v) => {
                let mut st = s.serialize_struct("ClearValue", 2)?;
                st.serialize_field("kind", "uint_color")?;
                st.serialize_field("value", v)?;
                st.end()
            }
            ClearValue::DepthStencil { depth, stencil } => {
                let mut st = s.serialize_struct("ClearValue", 3)?;
                st.serialize_field("kind", "depth_stencil")?;
                st.serialize_field("depth", depth)?;
                st.serialize_field("stencil", stencil)?;
                st.end()
            }
        }
    }
}

#[cfg(feature = "dashi-serde")]
impl<'de> Deserialize<'de> for ClearValue {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::{
            de::{Error, MapAccess, Visitor},
            Deserialize,
        };
        use std::fmt;
        #[derive(Deserialize)]
        struct Tag {
            kind: String,
        }
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = ClearValue;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "ClearValue")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut kind: Option<String> = None;
                let mut value_f: Option<[f32; 4]> = None;
                let mut value_i: Option<[i32; 4]> = None;
                let mut value_u: Option<[u32; 4]> = None;
                let mut depth: Option<f32> = None;
                let mut stencil: Option<u32> = None;
                while let Some(k) = map.next_key::<String>()? {
                    match k.as_str() {
                        "kind" => kind = Some(map.next_value()?),
                        "value_f" | "value" => {
                            // allow "value" for color variants
                            // we'll disambiguate by kind below
                            // try f32, then i32, then u32
                            // serde will backtrack appropriately
                            value_f = map.next_value::<Option<[f32; 4]>>()?;
                        }
                        "depth" => depth = Some(map.next_value()?),
                        "stencil" => stencil = Some(map.next_value()?),
                        _ => {
                            let _: serde_json::Value = map.next_value()?;
                        }
                    }
                }
                let k = kind.ok_or_else(|| Error::missing_field("kind"))?;
                match k.as_str() {
                    "color" => Ok(ClearValue::Color(
                        value_f.ok_or_else(|| Error::missing_field("value"))?,
                    )),
                    "int_color" => {
                        // YAML/JSON may coerce; re-deserialize as i32 array if needed
                        Err(Error::custom(
                            "int_color: please provide via Graphics config value_int",
                        ))
                    }
                    "uint_color" => Err(Error::custom(
                        "uint_color: please provide via Graphics config value_uint",
                    )),
                    "depth_stencil" => Ok(ClearValue::DepthStencil {
                        depth: depth.ok_or_else(|| Error::missing_field("depth"))?,
                        stencil: stencil.ok_or_else(|| Error::missing_field("stencil"))?,
                    }),
                    _ => Err(Error::custom("unknown ClearValue.kind")),
                }
            }
        }
        d.deserialize_map(V)
    }
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
    pub img: ImageView,
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

#[derive(Clone, Debug, Default)]
pub struct RenderPassAttachmentInfo {
    pub name: Option<String>,
    pub view: ImageView,
    pub clear_value: Option<ClearValue>,
    pub description: AttachmentDescription,
}

impl Hash for RenderPassAttachmentInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.view.hash(state);
        self.clear_value.hash(state);
        self.description.hash(state);
    }
}

#[derive(Clone, Debug, Default)]
pub struct RenderPassSubpassTargets {
    pub color_attachments: [Option<RenderPassAttachmentInfo>; 4],
    pub depth_attachment: Option<RenderPassAttachmentInfo>,
}

impl Hash for RenderPassSubpassTargets {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.color_attachments.hash(state);
        self.depth_attachment.hash(state);
    }
}

impl RenderPassSubpassTargets {
    pub fn color_views(&self) -> [Option<ImageView>; 8] {
        let mut views = [None; 8];
        for (dst, src) in views.iter_mut().zip(self.color_attachments.iter()) {
            *dst = src.as_ref().map(|att| att.view);
        }
        views
    }

    pub fn color_clear_values(&self) -> [Option<ClearValue>; 8] {
        let mut clears = [None; 8];
        for (dst, src) in clears.iter_mut().zip(self.color_attachments.iter()) {
            *dst = src.as_ref().and_then(|att| att.clear_value);
        }
        clears
    }

    pub fn depth_view(&self) -> Option<ImageView> {
        self.depth_attachment.as_ref().map(|att| att.view)
    }

    pub fn depth_clear_value(&self) -> Option<ClearValue> {
        self.depth_attachment
            .as_ref()
            .and_then(|att| att.clear_value)
    }

    pub fn find_color_attachment_mut(
        &mut self,
        name: &str,
    ) -> Option<&mut RenderPassAttachmentInfo> {
        self.color_attachments
            .iter_mut()
            .filter_map(|att| att.as_mut())
            .find(|att| att.name.as_deref() == Some(name))
    }
}

#[derive(Clone, Debug, Default)]
pub struct RenderPassWithImages {
    pub render_pass: Handle<RenderPass>,
    pub subpasses: Vec<RenderPassSubpassTargets>,
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

impl Hash for RenderPassInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.viewport.hash(state);
        self.subpasses.hash(state);
    }
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
    pub bt_layouts: [Option<Handle<BindTableLayout>>; 4],
    pub shader: &'a PipelineShaderInfo<'a>,
}

impl Hash for ComputePipelineLayoutInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bt_layouts.hash(state);
        self.shader.hash(state);
    }
}

#[derive(Debug)]
pub struct GraphicsPipelineLayoutInfo<'a> {
    pub debug_name: &'a str,
    pub vertex_info: VertexDescriptionInfo<'a>,
    pub bt_layouts: [Option<Handle<BindTableLayout>>; 4],
    pub shaders: &'a [PipelineShaderInfo<'a>],
    pub details: GraphicsPipelineDetails,
}

impl Hash for GraphicsPipelineLayoutInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex_info.hash(state);
        self.bt_layouts.hash(state);
        self.shaders.hash(state);
        self.details.hash(state);
    }
}

pub struct ComputePipelineInfo<'a> {
    pub debug_name: &'a str,
    pub layout: Handle<ComputePipelineLayout>,
}

#[derive(Clone, Debug, Default)]
pub struct RenderPassSubpassInfo {
    pub color_formats: Vec<Format>,
    pub depth_format: Option<Format>,
    pub samples: SubpassSampleInfo,
}

pub struct GraphicsPipelineInfo<'a> {
    pub debug_name: &'a str,
    pub layout: Handle<GraphicsPipelineLayout>,
    pub attachment_formats: Vec<Format>,
    pub depth_format: Option<Format>,
    pub subpass_samples: SubpassSampleInfo,
    pub subpass_id: u8,
}

impl Hash for ComputePipelineInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.layout.hash(state);
    }
}

impl Hash for GraphicsPipelineInfo<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.layout.hash(state);
        self.attachment_formats.hash(state);
        self.depth_format.hash(state);
        self.subpass_samples.hash(state);
        self.subpass_id.hash(state);
    }
}

impl<'a> Default for GraphicsPipelineInfo<'a> {
    fn default() -> Self {
        Self {
            debug_name: Default::default(),
            layout: Default::default(),
            attachment_formats: Default::default(),
            depth_format: None,
            subpass_samples: Default::default(),
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

impl Hash for WindowInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.size.hash(state);
        self.resizable.hash(state);
    }
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

impl Hash for DisplayInfo {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.window.hash(state);
        self.vsync.hash(state);
        self.buffering.hash(state);
    }
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

// -----------------------------------------------------------------------------
// JSON/YAML Config types and borrowed views (feature-gated)
// -----------------------------------------------------------------------------
#[cfg(feature = "dashi-serde")]
pub mod cfg {
    use super::*;
    use serde::{Deserialize, Serialize};

    // ---------------- RenderPass (authoring) ----------------
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AttachmentCfg {
        #[serde(flatten)]
        pub description: AttachmentDescription,
        #[serde(default)]
        pub debug_name: Option<String>,
        #[serde(default)]
        pub clear_value: Option<ClearValue>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SubpassDescriptionCfg {
        #[serde(default)]
        pub color_attachments: Vec<AttachmentCfg>,
        #[serde(default)]
        pub depth_stencil_attachment: Option<AttachmentCfg>,
        #[serde(default)]
        pub subpass_dependencies: Vec<SubpassDependency>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct RenderPassCfg {
        pub debug_name: String,
        pub viewport: Viewport,
        pub subpasses: Vec<SubpassDescriptionCfg>,
    }

    impl RenderPassCfg {
        pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
            serde_json::from_str(s)
        }
        pub fn from_yaml(s: &str) -> Result<Self, serde_yaml::Error> {
            serde_yaml::from_str(s)
        }
    }

    // ---------------- BindTableLayout authoring ----------------
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ShaderInfoCfg {
        pub stage: ShaderType,
        #[serde(default)]
        pub variables: Vec<BindTableVariable>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BindTableLayoutCfg {
        pub debug_name: String,
        #[serde(default)]
        pub shaders: Vec<ShaderInfoCfg>,
    }

    impl BindTableLayoutCfg {
        pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
            serde_json::from_str(s)
        }
        pub fn from_yaml(s: &str) -> Result<Self, serde_yaml::Error> {
            serde_yaml::from_str(s)
        }
        pub fn vec_from_yaml(s: &str) -> Result<Vec<Self>, serde_yaml::Error> {
            serde_yaml::from_str(s)
        }
    }

    pub struct BindTableLayoutBorrowed<'a> {
        cfg: &'a BindTableLayoutCfg,
        shaders: Vec<ShaderInfo<'a>>,
    }

    impl BindTableLayoutBorrowed<'_> {
        pub fn info(&self) -> BindTableLayoutInfo<'_> {
            BindTableLayoutInfo {
                debug_name: &self.cfg.debug_name,
                shaders: &self.shaders,
            }
        }
    }

    impl BindTableLayoutCfg {
        pub fn borrow(&self) -> BindTableLayoutBorrowed<'_> {
            let shaders = self
                .shaders
                .iter()
                .map(|shader| ShaderInfo {
                    shader_type: shader.stage,
                    variables: &shader.variables,
                })
                .collect();

            BindTableLayoutBorrowed { cfg: self, shaders }
        }
    }

    // ---------------- Pipeline Layout/Shader authoring ----------------
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SpecializationCfg {
        pub slot: usize,
        /// Usually authored as a byte array in YAML or base64 string in JSON.
        pub data: Vec<u8>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct PipelineShaderCfg {
        pub stage: ShaderType,
        /// Optional path to a `.spv` file. If absent, `spirv` words are used.
        #[serde(default)]
        pub spirv_path: Option<String>,
        #[serde(default)]
        pub spirv: Vec<u32>,
        #[serde(default)]
        pub specialization: Vec<SpecializationCfg>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct VertexDescriptionCfg {
        pub entries: Vec<VertexEntryInfo>,
        pub stride: usize,
        pub rate: VertexRate,
    }

    #[derive(Debug, Clone, Serialize, Deserialize, Default)]
    pub struct PipelineLayoutsRef {
        /// Names for BG/BT layouts you resolve to Handles at build time.
        pub bt_layouts: [Option<String>; 4],
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GraphicsPipelineLayoutCfg {
        pub debug_name: String,
        pub vertex_info: VertexDescriptionCfg,
        #[serde(default)]
        pub layouts: PipelineLayoutsRef,
        pub shaders: Vec<PipelineShaderCfg>,
        #[serde(default)]
        pub details: GraphicsPipelineDetails,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ComputePipelineLayoutCfg {
        #[serde(default)]
        pub debug_name: Option<String>,
        #[serde(default)]
        pub layouts: PipelineLayoutsRef,
        pub shader: PipelineShaderCfg,
    }

    // ---------------- Pipeline authoring ----------------
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct GraphicsPipelineCfg {
        pub debug_name: String,
        /// Name key for a previously created layout.
        pub layout: String,
        /// Name key for a previously created render pass.
        pub render_pass: String,
        pub subpass_id: u8,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ComputePipelineCfg {
        pub debug_name: String,
        /// Name key for a previously created layout.
        pub layout: String,
    }

    impl GraphicsPipelineLayoutCfg {
        pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
            serde_json::from_str(s)
        }
        pub fn from_yaml(s: &str) -> Result<Self, serde_yaml::Error> {
            serde_yaml::from_str(s)
        }
    }
    impl ComputePipelineLayoutCfg {
        pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
            serde_json::from_str(s)
        }
        pub fn from_yaml(s: &str) -> Result<Self, serde_yaml::Error> {
            serde_yaml::from_str(s)
        }
    }
    impl GraphicsPipelineCfg {
        pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
            serde_json::from_str(s)
        }
        pub fn from_yaml(s: &str) -> Result<Self, serde_yaml::Error> {
            serde_yaml::from_str(s)
        }
    }
    impl ComputePipelineCfg {
        pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
            serde_json::from_str(s)
        }
        pub fn from_yaml(s: &str) -> Result<Self, serde_yaml::Error> {
            serde_yaml::from_str(s)
        }
    }

    // ---------------- Cfg → Runtime borrowed conversions ----------------
    impl GraphicsPipelineCfg {
        /// Resolve string keys to handles and return a borrowed `GraphicsPipelineInfo`.
        pub fn to_info<'a>(
            &'a self,
            layouts: &std::collections::HashMap<String, Handle<GraphicsPipelineLayout>>,
            subpass_info: RenderPassSubpassInfo,
        ) -> Result<GraphicsPipelineInfo<'a>, String> {
            let layout = *layouts
                .get(&self.layout)
                .ok_or_else(|| format!("Unknown GraphicsPipelineLayout key: {}", self.layout))?;
            Ok(GraphicsPipelineInfo {
                debug_name: &self.debug_name,
                layout,
                attachment_formats: subpass_info.color_formats,
                depth_format: subpass_info.depth_format,
                subpass_samples: subpass_info.samples,
                subpass_id: self.subpass_id,
            })
        }
    }

    pub(crate) fn load_text(path: &str) -> anyhow::Result<String> {
        Ok(std::fs::read_to_string(path)?)
    }
}
