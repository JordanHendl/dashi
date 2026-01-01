use crate::utils::Handle;
use ash::vk;
use vk_mem;

use super::ImageInfo;

#[derive(Debug)]
pub struct Image {
    pub(crate) img: vk::Image,
    pub(crate) alloc: vk_mem::Allocation,
    pub(crate) layouts: Vec<vk::ImageLayout>,
    pub(crate) info_handle: Handle<ImageInfoRecord>,
}

#[derive(Debug)]
pub(crate) struct ImageInfoRecord {
    pub(crate) info: ImageInfo<'static>,
    debug_name: String,
}

impl ImageInfoRecord {
    pub(crate) fn new(info: &ImageInfo) -> Self {
        let debug_name = info.debug_name.to_string();
        // SAFETY: `debug_name` owns the backing string data for the lifetime of this record.
        let debug_name_ref: &'static str =
            unsafe { std::mem::transmute::<&str, &'static str>(debug_name.as_str()) };
        let info = ImageInfo {
            debug_name: debug_name_ref,
            dim: info.dim,
            layers: info.layers,
            format: info.format,
            mip_levels: info.mip_levels,
            samples: info.samples,
            cube_compatible: info.cube_compatible,
            initial_data: None,
        };

        Self { info, debug_name }
    }
}

#[derive(Debug)]
pub struct Sampler {
    pub(crate) sampler: vk::Sampler,
}

#[derive(Debug)]
pub struct VkImageView {
    pub(crate) img: Handle<Image>,
    pub(crate) range: vk::ImageSubresourceRange,
    pub(crate) view: vk::ImageView,
}
