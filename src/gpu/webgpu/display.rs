//! WebGPU swapchain/display helpers.

#[cfg(target_arch = "wasm32")]
use crate::gpu::vulkan::{GPUError, Result};

#[cfg(target_arch = "wasm32")]
pub struct Display<'a> {
    surface: wgpu::Surface<'a>,
    config: wgpu::SurfaceConfiguration,
}

#[cfg(target_arch = "wasm32")]
pub struct Frame {
    pub view: wgpu::TextureView,
    surface_texture: wgpu::SurfaceTexture,
}

#[cfg(target_arch = "wasm32")]
impl<'a> Display<'a> {
    pub(crate) fn new(surface: wgpu::Surface<'a>, config: wgpu::SurfaceConfiguration) -> Self {
        Self { surface, config }
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    pub fn configure(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.config.width = width.max(1);
        self.config.height = height.max(1);
        self.surface.configure(device, &self.config);
    }

    pub fn acquire_frame(&mut self, device: &wgpu::Device) -> Result<Frame> {
        match self.surface.get_current_texture() {
            Ok(texture) => Ok(Frame {
                view: texture
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                surface_texture: texture,
            }),
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(device, &self.config);
                self.surface
                    .get_current_texture()
                    .map(|texture| Frame {
                        view: texture
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default()),
                        surface_texture: texture,
                    })
                    .map_err(|_| {
                        GPUError::SwapchainConfigError("Failed to acquire WebGPU frame")
                    })
            }
            Err(wgpu::SurfaceError::OutOfMemory) => Err(GPUError::SwapchainConfigError(
                "WebGPU surface is out of memory",
            )),
            Err(wgpu::SurfaceError::Timeout) => Err(GPUError::SwapchainConfigError(
                "Timed out acquiring WebGPU frame",
            )),
        }
    }
}

#[cfg(target_arch = "wasm32")]
impl Frame {
    pub fn present(self) {
        self.surface_texture.present();
    }
}
