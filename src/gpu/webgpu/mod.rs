use crate::gpu::vulkan::{ContextInfo, ContextLimits, GPUError, Result};

#[cfg(not(target_arch = "wasm32"))]
use crate::gpu::vulkan::VulkanContext;
#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
pub struct Context {
    inner: VulkanContext,
    limits: ContextLimits,
    #[allow(dead_code)]
    pools: ResourcePools,
    #[allow(dead_code)]
    headless: bool,
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
impl std::ops::Deref for Context {
    type Target = VulkanContext;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl std::ops::DerefMut for Context {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(target_arch = "wasm32")]
pub struct Context {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    limits: ContextLimits,
    headless: bool,
}

#[cfg(target_arch = "wasm32")]
impl Context {
    pub fn new(info: &ContextInfo) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let canvas = crate::gpu::webgpu::resolve_canvas(info)?;
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|_| GPUError::SwapchainConfigError("Failed to create WebGPU surface"))?;
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            },
        ))
        .ok_or(GPUError::SwapchainConfigError(
            "No compatible WebGPU adapter available",
        ))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("dashi-webgpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .map_err(|_| GPUError::SwapchainConfigError("Unable to create WebGPU device"))?;
        let limits = ContextLimits::from(WebGpuLimits::default());
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            limits,
            headless: false,
        })
    }

    pub fn headless(_info: &ContextInfo) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                compatible_surface: None,
                force_fallback_adapter: true,
            },
        ))
        .ok_or(GPUError::SwapchainConfigError(
            "No compatible WebGPU adapter available",
        ))?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("dashi-webgpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .map_err(|_| GPUError::SwapchainConfigError("Unable to create WebGPU device"))?;
        let limits = ContextLimits::from(WebGpuLimits::default());
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            limits,
            headless: true,
        })
    }

    pub fn limits(&self) -> ContextLimits {
        self.limits
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn make_display(&self, info: &ContextInfo) -> Result<display::Display<'_>> {
        if self.headless {
            return Err(GPUError::HeadlessDisplayNotSupported);
        }
        let canvas = crate::gpu::webgpu::resolve_canvas(info)?;
        let surface = self
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|_| GPUError::SwapchainConfigError("Failed to create WebGPU surface"))?;
        let capabilities = surface.get_capabilities(&self.adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .unwrap_or(capabilities.formats[0]);
        let alpha_mode = capabilities.alpha_modes[0];
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: canvas.width().max(1),
            height: canvas.height().max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: Vec::new(),
        };
        surface.configure(&self.device, &config);
        Ok(display::Display::new(surface, config))
    }

    pub fn destroy(self) {}
}

#[cfg(target_arch = "wasm32")]
fn resolve_canvas(info: &ContextInfo) -> Result<web_sys::HtmlCanvasElement> {
    use wasm_bindgen::JsCast;

    let web_surface = info.web_surface.as_ref().ok_or(
        GPUError::SwapchainConfigError("WebGPU surface requires a canvas descriptor"),
    )?;
    match web_surface {
        crate::gpu::vulkan::WebSurfaceInfo::Canvas(canvas) => Ok(canvas.clone()),
        crate::gpu::vulkan::WebSurfaceInfo::CanvasId(canvas_id) => {
            let document = web_sys::window()
                .and_then(|window| window.document())
                .ok_or(GPUError::SwapchainConfigError(
                    "Unable to access browser document",
                ))?;
            let element = document
                .get_element_by_id(canvas_id)
                .ok_or(GPUError::SwapchainConfigError(
                    "Canvas element not found",
                ))?;
            element
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .map_err(|_| {
                    GPUError::SwapchainConfigError("Element is not an HtmlCanvasElement")
                })
        }
        #[cfg(not(target_arch = "wasm32"))]
        crate::gpu::vulkan::WebSurfaceInfo::Unsupported => Err(
            GPUError::SwapchainConfigError("WebGPU surface is unsupported on this platform"),
        ),
    }
}
