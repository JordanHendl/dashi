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
}
