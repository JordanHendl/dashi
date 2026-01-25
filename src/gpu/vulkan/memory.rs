use crate::utils::Handle;
use ash::vk;
use offset_allocator;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use vk_mem;

use super::BufferInfo;

#[derive(Debug)]
pub struct Buffer {
    pub(crate) buf: vk::Buffer,
    pub(crate) alloc: vk_mem::Allocation,
    pub(crate) offset: u32,
    pub(crate) size: u32,
    pub(crate) suballocated: bool,
    pub(crate) info_handle: Handle<BufferInfoRecord>,
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
            alloc: unsafe { std::mem::transmute_copy(&self.alloc) },
            offset: self.offset.clone(),
            size: self.size.clone(),
            suballocated: self.suballocated.clone(),
            info_handle: self.info_handle,
        }
    }
}

#[derive(Debug)]
pub(crate) struct BufferInfoRecord {
    pub(crate) info: BufferInfo<'static>,
    debug_name: String,
}

impl BufferInfoRecord {
    pub(crate) fn new(info: &BufferInfo) -> Self {
        let debug_name = info.debug_name.to_string();
        // SAFETY: `debug_name` owns the backing string data for the lifetime of this record.
        let debug_name_ref: &'static str =
            unsafe { std::mem::transmute::<&str, &'static str>(debug_name.as_str()) };
        let info = BufferInfo {
            debug_name: debug_name_ref,
            byte_size: info.byte_size,
            visibility: info.visibility,
            usage: info.usage,
            initial_data: None,
        };

        Self { info, debug_name }
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
            std::slice::from_raw_parts_mut(typed_map, self.size as usize / std::mem::size_of::<T>())
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynamicAllocatorState {
    pub pool: Handle<Buffer>,
    pub report: offset_allocator::StorageReport,
    pub min_alloc_size: u32,
    pub rollover: bool,
}

impl Hash for DynamicAllocatorState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pool.hash(state);
        self.report.total_free_space.hash(state);
        self.report.largest_free_region.hash(state);
        self.min_alloc_size.hash(state);
        self.rollover.hash(state);
    }
}

#[derive(Clone)]
pub struct DynamicAllocator {
    pub(crate) allocator: Arc<Mutex<offset_allocator::Allocator>>,
    pub(crate) ptr: *mut u8,
    pub(crate) min_alloc_size: u32,
    pub(crate) pool: Handle<Buffer>,
    pub(crate) rollover: bool,
}

impl Default for DynamicAllocator {
    fn default() -> Self {
        Self {
            allocator: Arc::new(Mutex::new(Default::default())),
            ptr: std::ptr::null_mut(),
            min_alloc_size: Default::default(),
            pool: Default::default(),
            rollover: true,
        }
    }
}

impl DynamicAllocator {
    /// Resets the allocator, invalidating all previously bumped buffers.
    ///
    /// # Safety
    /// Only call this when no GPU operations are reading from the
    /// allocations produced by this allocator.
    pub fn reset(&self) {
        let mut allocator = self
            .allocator
            .lock()
            .expect("dynamic allocator lock poisoned");
        allocator.reset();
    }

    pub fn state(&self) -> DynamicAllocatorState {
        let allocator = self
            .allocator
            .lock()
            .expect("dynamic allocator lock poisoned");
        DynamicAllocatorState {
            pool: self.pool,
            report: allocator.storage_report(),
            min_alloc_size: self.min_alloc_size,
            rollover: self.rollover,
        }
    }

    /// Allocates a new [`DynamicBuffer`] of `min_alloc_size` bytes.
    ///
    /// # Safety
    /// The caller must ensure that no pending GPU reads exist for
    /// buffers previously returned by this allocator before calling this
    /// method. If rollover is enabled, the allocator may be reset when it
    /// reaches the end of its space, invalidating previously returned
    /// buffers.
    pub fn bump(&self) -> Option<DynamicBuffer> {
        let mut allocator = self
            .allocator
            .lock()
            .expect("dynamic allocator lock poisoned");
        let alloc = match allocator.allocate(self.min_alloc_size) {
            Some(alloc) => alloc,
            None if self.rollover => {
                allocator.reset();
                allocator.allocate(self.min_alloc_size)?
            }
            None => return None,
        };
        Some(DynamicBuffer {
            handle: self.pool,
            alloc,
            ptr: unsafe { self.ptr.offset(alloc.offset as isize) },
            size: self.min_alloc_size as u16,
        })
    }
}
