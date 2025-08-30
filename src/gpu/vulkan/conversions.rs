use ash::vk;
use crate::{
    AspectMask, BarrierPoint, BlendFactor, BlendOp, BorderColor, ColorBlendState, DynamicState,
    Filter, Format, GPUError, LoadOp, Rect2D, SampleCount, SamplerAddressMode, SamplerInfo,
    SamplerMipmapMode, StoreOp, WriteMask,
};

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

impl From<SamplerMipmapMode> for vk::SamplerMipmapMode {
    fn from(mipmap_mode: SamplerMipmapMode) -> Self {
        match mipmap_mode {
            SamplerMipmapMode::Nearest => vk::SamplerMipmapMode::NEAREST,
            SamplerMipmapMode::Linear => vk::SamplerMipmapMode::LINEAR,
        }
    }
}

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

impl From<SamplerInfo> for vk::SamplerCreateInfo {
    fn from(info: SamplerInfo) -> Self {
        vk::SamplerCreateInfo {
            mag_filter: info.mag_filter.into(),
            min_filter: info.min_filter.into(),
            address_mode_u: info.address_mode_u.into(),
            address_mode_v: info.address_mode_v.into(),
            address_mode_w: info.address_mode_w.into(),
            anisotropy_enable: if info.anisotropy_enable { vk::TRUE } else { vk::FALSE },
            max_anisotropy: info.max_anisotropy,
            border_color: info.border_color.into(),
            unnormalized_coordinates: if info.unnormalized_coordinates { vk::TRUE } else { vk::FALSE },
            compare_enable: if info.compare_enable { vk::TRUE } else { vk::FALSE },
            mipmap_mode: info.mipmap_mode.into(),
            ..Default::default()
        }
    }
}

pub(super) fn convert_rect2d_to_vulkan(rect: Rect2D) -> vk::Rect2D {
    vk::Rect2D {
        offset: vk::Offset2D { x: rect.x as i32, y: rect.y as i32 },
        extent: vk::Extent2D { width: rect.w, height: rect.h },
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

pub(super) fn lib_to_vk_image_format(fmt: &Format) -> vk::Format {
    match fmt {
        Format::RGB8 => vk::Format::R8G8B8_SRGB,
        Format::RGBA32F => vk::Format::R32G32B32A32_SFLOAT,
        Format::RGBA8 => vk::Format::R8G8B8A8_SRGB,
        Format::BGRA8 => vk::Format::B8G8R8A8_SRGB,
        Format::BGRA8Unorm => vk::Format::B8G8R8A8_SNORM,
        Format::D24S8 => vk::Format::D24_UNORM_S8_UINT,
        Format::R8Uint => vk::Format::R8_UINT,
        Format::R8Sint => vk::Format::R8_SINT,
        Format::RGBA8Unorm => vk::Format::R8G8B8A8_UNORM,
    }
}

pub(super) fn vk_to_lib_image_format(fmt: vk::Format) -> Result<Format, GPUError> {
    match fmt {
        vk::Format::R8G8B8_SRGB => Ok(Format::RGB8),
        vk::Format::R32G32B32A32_SFLOAT => Ok(Format::RGBA32F),
        vk::Format::R8G8B8A8_SRGB => Ok(Format::RGBA8),
        vk::Format::B8G8R8A8_SRGB => Ok(Format::BGRA8),
        vk::Format::B8G8R8A8_SNORM => Ok(Format::BGRA8Unorm),
        vk::Format::R8_SINT => Ok(Format::R8Sint),
        vk::Format::R8_UINT => Ok(Format::R8Uint),
        other => Err(GPUError::UnsupportedFormat(other)),
    }
}

/// Returns the number of color channels for a [`Format`].
pub fn channel_count(fmt: &Format) -> u32 {
    match fmt {
        Format::RGB8 => 3,
        Format::BGRA8 | Format::BGRA8Unorm | Format::RGBA8 | Format::RGBA8Unorm | Format::RGBA32F => 4,
        Format::D24S8 => 4,
        Format::R8Sint | Format::R8Uint => 1,
    }
}

/// Returns the number of bytes each channel in the [`Format`] occupies.
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

/// Calculates the dimensions of a mip level for a 3‑D texture.
pub fn mip_dimensions(dim: [u32; 3], level: u32) -> [u32; 3] {
    [
        std::cmp::max(1, dim[0] >> level),
        std::cmp::max(1, dim[1] >> level),
        std::cmp::max(1, dim[2] >> level),
    ]
}

pub(super) fn convert_load_op(load_op: LoadOp) -> vk::AttachmentLoadOp {
    match load_op {
        LoadOp::Load => vk::AttachmentLoadOp::LOAD,
        LoadOp::Clear => vk::AttachmentLoadOp::CLEAR,
        LoadOp::DontCare => vk::AttachmentLoadOp::DONT_CARE,
    }
}

pub(super) fn convert_store_op(store_op: StoreOp) -> vk::AttachmentStoreOp {
    match store_op {
        StoreOp::Store => vk::AttachmentStoreOp::STORE,
        StoreOp::DontCare => vk::AttachmentStoreOp::DONT_CARE,
    }
}

pub(super) fn convert_sample_count(sample_count: SampleCount) -> vk::SampleCountFlags {
    match sample_count {
        SampleCount::S1 => vk::SampleCountFlags::TYPE_1,
        SampleCount::S2 => vk::SampleCountFlags::TYPE_2,
    }
}
