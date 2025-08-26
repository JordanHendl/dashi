use crate::utils::Handle;
use ash::vk;
use offset_allocator;
use vk_mem;

#[derive(Debug)]
pub struct Buffer {
    pub(crate) buf: vk::Buffer,
    pub(crate) alloc: vk_mem::Allocation,
    pub(crate) offset: u32,
    pub(crate) size: u32,
    pub(crate) suballocated: bool,
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
            alloc: unsafe { std::mem::transmute_copy(&self.alloc) },
            offset: self.offset.clone(),
            size: self.size.clone(),
            suballocated: self.suballocated.clone(),
        }
    }
}

impl Handle<Buffer> {
    /// Creates a [`DynamicBuffer`] view into this buffer at `byte_offset`.
    ///
    /// The caller must ensure the offset is within the buffer's bounds
    /// and correctly aligned for the data that will be written.
    pub fn to_unmapped_dynamic(&self, byte_offset: u32) -> DynamicBuffer {
        DynamicBuffer {
            handle: self.clone(),
            alloc: offset_allocator::Allocation {
                offset: byte_offset,
                metadata: 0,
            },
            ptr: std::ptr::null_mut(),
            size: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DynamicBuffer {
    pub(crate) handle: Handle<Buffer>,
    pub(crate) alloc: offset_allocator::Allocation,
    pub(crate) ptr: *mut u8,
    pub(crate) size: u16,
}

impl Default for DynamicBuffer {
    fn default() -> Self {
        Self {
            handle: Default::default(),
            alloc: offset_allocator::Allocation {
                offset: 0,
                metadata: 0,
            },
            ptr: std::ptr::null_mut(),
            size: Default::default(),
        }
    }
}

impl DynamicBuffer {
    /// Returns the handle to the underlying [`Buffer`].
    pub fn handle(&self) -> Handle<Buffer> {
        self.handle
    }

    /// Returns the byte offset into the underlying buffer.
    pub fn offset(&self) -> u32 {
        self.alloc.offset
    }

    /// Provides a mutable slice over the mapped memory.
    ///
    /// The slice is valid only while the buffer remains mapped and the
    /// allocator that created it has not been reset or bumped.
    pub fn slice<T>(&mut self) -> &mut [T] {
        let typed_map: *mut T = unsafe { std::mem::transmute(self.ptr) };
        unsafe {
            std::slice::from_raw_parts_mut(
                typed_map,
                self.size as usize / std::mem::size_of::<T>(),
            )
        }
    }
}

#[derive(Clone)]
pub struct DynamicAllocator {
    pub(crate) allocator: offset_allocator::Allocator,
    pub(crate) ptr: *mut u8,
    pub(crate) min_alloc_size: u32,
    pub(crate) pool: Handle<Buffer>,
}

impl Default for DynamicAllocator {
    fn default() -> Self {
        Self {
            allocator: Default::default(),
            ptr: std::ptr::null_mut(),
            min_alloc_size: Default::default(),
            pool: Default::default(),
        }
    }
}

impl DynamicAllocator {
    /// Resets the allocator, invalidating all previously bumped buffers.
    ///
    /// # Safety
    /// Only call this when no GPU operations are reading from the
    /// allocations produced by this allocator.
    pub fn reset(&mut self) {
        self.allocator.reset();
    }

    /// Allocates a new [`DynamicBuffer`] of `min_alloc_size` bytes.
    ///
    /// # Safety
    /// The caller must ensure that no pending GPU reads exist for
    /// buffers previously returned by this allocator before calling this
    /// method.
    pub fn bump(&mut self) -> Option<DynamicBuffer> {
        let alloc = self.allocator.allocate(self.min_alloc_size)?;
        Some(DynamicBuffer {
            handle: self.pool,
            alloc,
            ptr: unsafe { self.ptr.offset(alloc.offset as isize) },
            size: self.min_alloc_size as u16,
        })
    }
}

