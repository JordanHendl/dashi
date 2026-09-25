//! Vulkan external resources. OS handles name memory or semaphore payloads,
//! never `VkImage`/`VkBuffer` objects. No presentation API is involved here.
use crate::{
    Buffer, BufferInfo, CommandQueue, Context, Handle, Image, ImageInfo, Result, Semaphore,
};
use std::marker::PhantomData;

#[cfg(windows)]
type PlatformHandle = std::os::windows::io::OwnedHandle;
#[cfg(unix)]
type PlatformHandle = std::os::fd::OwnedFd;

/// An owned external resource handle. Dropping it closes the OS handle.
/// Import metadata must travel with the handle; a raw value alone is insufficient.
#[derive(Debug)]
pub struct OSHandle<T> {
    pub(crate) inner: PlatformHandle,
    marker: PhantomData<fn() -> T>,
}

impl<T> OSHandle<T> {
    pub(crate) fn new(inner: PlatformHandle) -> Self {
        Self {
            inner,
            marker: PhantomData,
        }
    }
    pub fn try_clone(&self) -> std::io::Result<Self> {
        self.inner.try_clone().map(Self::new)
    }
    #[cfg(windows)]
    pub fn as_win32_handle(&self) -> std::os::windows::io::BorrowedHandle<'_> {
        use std::os::windows::io::AsHandle;
        self.inner.as_handle()
    }
    #[cfg(windows)]
    pub fn into_win32_handle(self) -> std::os::windows::io::OwnedHandle {
        self.inner
    }
    #[cfg(windows)]
    pub fn from_win32_handle(handle: std::os::windows::io::OwnedHandle) -> Self {
        Self::new(handle)
    }
    #[cfg(unix)]
    pub fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        use std::os::fd::AsFd;
        self.inner.as_fd()
    }
    #[cfg(unix)]
    pub fn into_fd(self) -> std::os::fd::OwnedFd {
        self.inner
    }
    #[cfg(unix)]
    pub fn from_fd(handle: std::os::fd::OwnedFd) -> Self {
        Self::new(handle)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceIdentity {
    pub device_uuid: [u8; 16],
    pub driver_uuid: [u8; 16],
    pub device_luid: [u8; 8],
    pub luid_valid: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExternalResourceCapabilities {
    pub memory: bool,
    pub binary_semaphore: bool,
}

/// Dedicated, offset-zero allocation. `handle_type` is the Vulkan opaque OS
/// memory handle flag appropriate for this platform.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExternalMemoryInfo {
    pub device: DeviceIdentity,
    pub allocation_size: u64,
    pub memory_type_index: u32,
    pub handle_type: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExternalImageDescriptor {
    pub memory: ExternalMemoryInfo,
    pub extent: [u32; 3],
    pub format: i32,
    pub usage: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExternalBufferDescriptor {
    pub memory: ExternalMemoryInfo,
    pub byte_size: u64,
    pub usage: u32,
}

pub struct ExportedImage {
    pub handle: OSHandle<Image>,
    pub descriptor: ExternalImageDescriptor,
}
pub struct ExportedBuffer {
    pub handle: OSHandle<Buffer>,
    pub descriptor: ExternalBufferDescriptor,
}
pub struct ExportedSemaphore {
    pub handle: OSHandle<Semaphore>,
    pub device: DeviceIdentity,
}

impl Context {
    pub fn device_identity(&self) -> Result<DeviceIdentity> {
        Ok(self.external_vulkan()?.device_identity())
    }
    pub fn external_resource_capabilities(&self) -> Result<ExternalResourceCapabilities> {
        Ok(self.external_vulkan()?.external_resource_capabilities())
    }
    /// Creates a dedicated exportable 2D, single-layer, single-mip color image.
    pub fn make_external_image(&mut self, info: &ImageInfo) -> Result<Handle<Image>> {
        self.external_vulkan_mut()?.make_external_image(info)
    }
    pub fn export_image(&self, image: Handle<Image>) -> Result<ExportedImage> {
        self.external_vulkan()?.export_image(image)
    }
    /// Imports dedicated opaque memory. The descriptor must describe this handle.
    /// # Safety
    /// The exporter and caller must coordinate access and allocation lifetime.
    pub unsafe fn import_image(
        &mut self,
        info: &ImageInfo,
        exported: ExportedImage,
    ) -> Result<Handle<Image>> {
        unsafe { self.external_vulkan_mut()?.import_image(info, exported) }
    }
    /// External buffers are device-local; stage CPU data through ordinary buffers.
    pub fn make_external_buffer(&mut self, info: &BufferInfo) -> Result<Handle<Buffer>> {
        self.external_vulkan_mut()?.make_external_buffer(info)
    }
    pub fn export_buffer(&self, buffer: Handle<Buffer>) -> Result<ExportedBuffer> {
        self.external_vulkan()?.export_buffer(buffer)
    }
    /// # Safety
    /// The descriptor must describe the memory and accesses must be synchronized.
    pub unsafe fn import_buffer(
        &mut self,
        info: &BufferInfo,
        exported: ExportedBuffer,
    ) -> Result<Handle<Buffer>> {
        unsafe { self.external_vulkan_mut()?.import_buffer(info, exported) }
    }
    pub fn make_external_semaphore(&mut self) -> Result<Handle<Semaphore>> {
        self.external_vulkan_mut()?.make_external_semaphore()
    }
    pub fn export_semaphore(&self, semaphore: Handle<Semaphore>) -> Result<ExportedSemaphore> {
        self.external_vulkan()?.export_semaphore(semaphore)
    }
    /// # Safety
    /// The handle must name a binary semaphore on the reported device.
    pub unsafe fn import_semaphore(
        &mut self,
        exported: ExportedSemaphore,
    ) -> Result<Handle<Semaphore>> {
        unsafe { self.external_vulkan_mut()?.import_semaphore(exported) }
    }
    /// Release a shared image in TRANSFER_SRC_OPTIMAL to an external device.
    /// Signal the producer semaphore in this queue's submission.
    pub fn release_external_image(
        &mut self,
        queue: &mut CommandQueue,
        image: Handle<Image>,
    ) -> Result<()> {
        self.external_vulkan_mut()?
            .transfer_external_image(queue, image, false)
    }
    /// Acquire a shared image returned in TRANSFER_SRC_OPTIMAL. The submission
    /// must wait for the external consumer's completion semaphore.
    pub fn acquire_external_image(
        &mut self,
        queue: &mut CommandQueue,
        image: Handle<Image>,
    ) -> Result<()> {
        self.external_vulkan_mut()?
            .transfer_external_image(queue, image, true)
    }
    pub fn release_external_buffer(
        &mut self,
        queue: &mut CommandQueue,
        buffer: Handle<Buffer>,
    ) -> Result<()> {
        self.external_vulkan_mut()?
            .transfer_external_buffer(queue, buffer, false)
    }
    pub fn acquire_external_buffer(
        &mut self,
        queue: &mut CommandQueue,
        buffer: Handle<Buffer>,
    ) -> Result<()> {
        self.external_vulkan_mut()?
            .transfer_external_buffer(queue, buffer, true)
    }
}
