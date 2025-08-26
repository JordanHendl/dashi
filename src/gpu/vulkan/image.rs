use crate::utils::Handle;
use ash::vk;
use vk_mem;

use super::Format;

#[derive(Debug)]
pub struct Image {
    pub(crate) img: vk::Image,
    pub(crate) alloc: vk_mem::Allocation,
    pub(crate) dim: [u32; 3],
    pub(crate) format: Format,
    pub(crate) layouts: Vec<vk::ImageLayout>,
    pub(crate) sub_layers: vk::ImageSubresourceLayers,
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

