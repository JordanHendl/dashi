use crate::gpu::vulkan::{ContextInfo, ContextLimits, GPUError, Result, VulkanContext};
use crate::utils::Pool;

pub mod builders;
pub mod commands;
pub mod display;
pub mod structs;

pub use structs::WebGpuLimits;

pub use crate::gpu::vulkan::{
    BindTable, BindTableLayout, Buffer, BufferView, CommandQueue, ComputePipeline,
    ComputePipelineLayout, DynamicAllocator, Fence, GraphicsPipeline, GraphicsPipelineLayout,
    Image, ImageView, RenderPass, Sampler, Semaphore, VkImageView,
};

#[derive(Default)]
pub struct ResourcePools {
    pub buffers: Pool<Buffer>,
    pub render_passes: Pool<RenderPass>,
    pub semaphores: Pool<Semaphore>,
    pub fences: Pool<Fence>,
    pub images: Pool<Image>,
    pub image_views: Pool<VkImageView>,
    pub samplers: Pool<Sampler>,
    pub bind_table_layouts: Pool<BindTableLayout>,
    pub bind_tables: Pool<BindTable>,
    pub graphics_pipeline_layouts: Pool<GraphicsPipelineLayout>,
    pub graphics_pipelines: Pool<GraphicsPipeline>,
    pub compute_pipeline_layouts: Pool<ComputePipelineLayout>,
    pub compute_pipelines: Pool<ComputePipeline>,
}

pub struct Context {
    inner: VulkanContext,
    limits: ContextLimits,
    #[allow(dead_code)]
    pools: ResourcePools,
    #[allow(dead_code)]
    headless: bool,
}

impl Context {
    pub fn new(info: &ContextInfo) -> Result<Self> {
        let inner = VulkanContext::new(info)?;
        let limits = ContextLimits::from(WebGpuLimits::default());
        Ok(Self {
            inner,
            limits,
            pools: ResourcePools::default(),
            headless: false,
        })
    }

    pub fn headless(info: &ContextInfo) -> Result<Self> {
        let inner = VulkanContext::headless(info)?;
        let limits = ContextLimits::from(WebGpuLimits::default());
        Ok(Self {
            inner,
            limits,
            pools: ResourcePools::default(),
            headless: true,
        })
    }

    pub fn limits(&self) -> ContextLimits {
        self.limits
    }

    pub fn destroy(self) {
        self.inner.destroy();
    }

    pub(crate) fn backend_mut_ptr(&mut self) -> *mut VulkanContext {
        &mut self.inner as *mut VulkanContext
    }

    pub(crate) fn report_unimplemented(&self, feature: &'static str) -> Result<(), GPUError> {
        Err(GPUError::Unimplemented(feature))
    }
}

impl std::ops::Deref for Context {
    type Target = VulkanContext;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for Context {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
