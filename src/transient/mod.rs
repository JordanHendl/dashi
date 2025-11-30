use std::ptr::NonNull;

use crate::utils::Pool;
use crate::{utils::Handle, Image, ImageInfo};
use crate::{
    Buffer, BufferInfo, BufferUsage, Context, Format, GPUError, MemoryVisibility, SampleCount,
};

#[derive(Default, Clone)]
pub struct ImageData {
    pub image: Handle<Image>,
    pub info: ImageInfo<'static>,
    pub handle: Handle<ImageData>,
}

#[derive(Default, Clone)]
pub struct BufferData {
    pub buffer: Handle<Buffer>,
    pub info: BufferInfo<'static>,
    pub handle: Handle<BufferData>,
}

#[derive(Default, Clone)]
pub struct ImageRequest {
    pub dim: [u32; 3],
    pub layers: u32,
    pub format: Format,
    pub mip_levels: u32,
    pub samples: SampleCount,
}

#[derive(Default)]
pub struct BufferRequest {
    pub byte_size: u32,
    pub visibility: MemoryVisibility,
    pub usage: BufferUsage,
}

pub struct ImagePool {
    ctx: NonNull<Context>,
    pool: Pool<ImageData>,
}

pub struct BufferPool {
    ctx: NonNull<Context>,
    pool: Pool<BufferData>,
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

impl ImagePool {
    pub fn new(ctx: &mut Context) -> Result<Self, GPUError> {
        const INITIAL_SIZE: usize = 256;
        Ok(Self {
            ctx: NonNull::new(ctx).expect("Context reference is NULL"),
            pool: Pool::new(INITIAL_SIZE),
        })
    }

    fn adequate(request: &ImageRequest, info: &ImageData) -> bool {
        info.image.valid()
            && request.layers == info.info.layers
            && request.mip_levels == info.info.mip_levels
            && request.dim == info.info.dim
            && request.format == info.info.format
            && request.samples == info.info.samples
    }
    
    pub fn make_image(&mut self, request: &ImageRequest) -> Result<ImageData, GPUError> {
        let ctx: &mut Context = unsafe { self.ctx.as_mut() };
        let mut best_fit: Option<ImageData> = None;
        let mut best_fit_slot = 0;
        self.pool.for_each_unoccupied(|a, idx| {
            if Self::adequate(request, a) {
                best_fit = Some(a.clone());
                best_fit_slot = idx;
            }
        });

        // Case: No best fit... so make one!
        if best_fit.is_none() {
            let info = ImageInfo {
                debug_name: "",
                initial_data: None,
                dim: request.dim,
                layers: request.layers,
                format: request.format,
                mip_levels: request.mip_levels,
                samples: request.samples,
            };

            let mut info = ImageData {
                image: ctx.make_image(&info)?,
                info,
                handle: Default::default(),
            };

            info.handle = self.pool.insert(info.clone()).expect("pool fail");
            self.pool.get_mut_ref(info.handle).unwrap().handle = info.handle;
            best_fit = Some(info);
        } else if let Some(info) = best_fit.as_mut() {
            info.handle = self
                .pool
                .insert_at(info.clone(), best_fit_slot)
                .expect("WEE");
        }

        Ok(best_fit.unwrap())
    }

    pub fn release(&mut self, image: &ImageData) {
        self.pool.release(image.handle);
    }
}

impl BufferPool {
    pub fn new(ctx: &mut Context) -> Result<Self, GPUError> {
        const INITIAL_SIZE: usize = 256;
        Ok(Self {
            ctx: NonNull::new(ctx).expect("Context reference is NULL"),
            pool: Pool::new(INITIAL_SIZE),
        })
    }
    fn adequate(request: &BufferRequest, info: &BufferData) -> bool {
        info.buffer.valid()
            && request.visibility == info.info.visibility
            && request.usage == info.info.usage
            && request.byte_size <= info.info.byte_size
    }

    pub fn make_buffer(&mut self, request: &BufferRequest) -> Result<BufferData, GPUError> {
        let ctx: &mut Context = unsafe { self.ctx.as_mut() };
        let mut best_fit: Option<BufferData> = None;
        let mut best_fit_slot = 0;
        self.pool.for_each_unoccupied(|a, idx| {
            if Self::adequate(request, a) {
                if let Some(fit) = best_fit.as_ref() {
                    if a.info.byte_size < fit.info.byte_size {
                        best_fit = Some(a.clone());
                        best_fit_slot = idx;
                    }
                } else {
                    best_fit = Some(a.clone());
                }
            }
        });
        // Case: No best fit... so make one!
        if best_fit.is_none() {
            let buffer_info = BufferInfo {
                debug_name: "",
                byte_size: request.byte_size,
                visibility: request.visibility,
                usage: request.usage,
                initial_data: None,
            };

            let mut info = BufferData {
                buffer: ctx.make_buffer(&buffer_info)?,
                info: buffer_info,
                handle: Default::default(),
            };

            info.handle = self.pool.insert(info.clone()).expect("pool fail");
            self.pool.get_mut_ref(info.handle).unwrap().handle = info.handle;
            best_fit = Some(info);
        } else if let Some(info) = best_fit.as_mut() {
            info.handle = self
                .pool
                .insert_at(info.clone(), best_fit_slot)
                .expect("WEE");
        }
        Ok(best_fit.unwrap())
    }

    pub fn release(&mut self, buffer: &BufferData) {
        self.pool.release(buffer.handle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    fn default_image_request() -> ImageRequest {
        ImageRequest {
            dim: [64, 64, 1],
            layers: 1,
            format: Format::RGBA8,
            mip_levels: 1,
            samples: SampleCount::S1,
        }
    }

    fn default_buffer_request(byte_size: u32) -> BufferRequest {
        BufferRequest {
            byte_size,
            visibility: MemoryVisibility::CpuAndGpu,
            usage: BufferUsage::UNIFORM,
        }
    }

    #[test]
    #[serial]
    fn image_pool_reuses_matching_image_from_pool() {
        let mut ctx = Context::headless(&Default::default()).expect("create headless context");
        let mut pool = ImagePool::new(&mut ctx).expect("make image pool");

        let request = default_image_request();
        let first = pool.make_image(&request).expect("allocate first image");

        // Return the image to the pool so that it becomes a reuse candidate.
        pool.release(&first);

        let reused = pool.make_image(&request).expect("reuse image");

        assert_eq!(first.image, reused.image);
        assert_eq!(first.info.dim, reused.info.dim);
        assert_eq!(first.info.layers, reused.info.layers);
        assert_eq!(first.info.format, reused.info.format);
        assert_eq!(first.info.mip_levels, reused.info.mip_levels);
        assert_eq!(first.info.samples, reused.info.samples);
    }

    #[test]
    #[serial]
    fn image_pool_creates_new_image_when_adequate_not_found() {
        let mut ctx = Context::headless(&Default::default()).expect("create headless context");
        let mut pool = ImagePool::new(&mut ctx).expect("make image pool");

        let base_request = default_image_request();
        let first = pool
            .make_image(&base_request)
            .expect("allocate initial image");

        pool.release(&first);

        // Request a different format to force a fresh allocation.
        let second_request = ImageRequest {
            format: Format::BGRA8Unorm,
            ..base_request
        };

        let second = pool
            .make_image(&second_request)
            .expect("allocate mismatched image");

        assert_ne!(first.image, second.image);
        assert_eq!(second.info.format, Format::BGRA8Unorm);
    }

    #[test]
    #[serial]
    fn buffer_pool_reuses_smallest_adequate_buffer() {
        let mut ctx = Context::headless(&Default::default()).expect("create headless context");
        let mut pool = BufferPool::new(&mut ctx).expect("make buffer pool");

        let small_request = default_buffer_request(1024);
        let small = pool
            .make_buffer(&small_request)
            .expect("allocate small buffer");
        pool.release(&small);

        let medium_request = default_buffer_request(2048);
        let medium = pool
            .make_buffer(&medium_request)
            .expect("allocate medium buffer");
        pool.release(&medium);

        let reuse_request = default_buffer_request(1500);
        let reused = pool
            .make_buffer(&reuse_request)
            .expect("reuse adequate buffer");

        assert_eq!(reused.buffer, medium.buffer);
        assert!(reused.info.byte_size >= reuse_request.byte_size);
    }

    #[test]
    #[serial]
    fn buffer_pool_allocates_new_when_no_fit_exists() {
        let mut ctx = Context::headless(&Default::default()).expect("create headless context");
        let mut pool = BufferPool::new(&mut ctx).expect("make buffer pool");

        let small_request = default_buffer_request(512);
        let small = pool
            .make_buffer(&small_request)
            .expect("allocate initial buffer");

        pool.release(&small);

        // Request a buffer larger than any available candidate.
        let large_request = default_buffer_request(small_request.byte_size * 4);
        let large = pool
            .make_buffer(&large_request)
            .expect("allocate new large buffer");

        assert_ne!(small.buffer, large.buffer);
        assert_eq!(large.info.byte_size, large_request.byte_size);
    }
}
