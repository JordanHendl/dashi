#[cfg(test)]
use crate::QueueType;
use crate::{
    cmd::Recording, gpu::driver::command::CopyBuffer, Buffer, BufferInfo, BufferView,
    CommandStream, Context, MemoryVisibility,
};

use super::{DynamicPool, Handle, Pool};
use crate::Result;

pub struct DynamicGPUPool {
    buffer: Handle<Buffer>,
    staging: Handle<Buffer>,
    pool: DynamicPool,
    dirty: Vec<bool>,
    item_size: usize,
    _ctx: *mut Context,
}

impl Default for DynamicGPUPool {
    fn default() -> Self {
        Self {
            buffer: Default::default(),
            staging: Default::default(),
            pool: Default::default(),
            dirty: Vec::new(),
            item_size: 0,
            _ctx: std::ptr::null_mut(),
        }
    }
}

pub struct GPUPool<T> {
    buffer: Handle<Buffer>,
    staging: Handle<Buffer>,
    pool: Pool<T>,
    dirty: Vec<bool>,
    _ctx: *mut Context,
}

impl<T> Default for GPUPool<T> {
    fn default() -> Self {
        Self {
            buffer: Default::default(),
            staging: Default::default(),
            pool: Default::default(),
            dirty: Vec::new(),
            _ctx: std::ptr::null_mut(),
        }
    }
}
/////////////////////////////////////////////////////////////////////////////////
///
///
///

impl<T> GPUPool<T> {
    pub fn new(ctx: &mut Context, info: &BufferInfo) -> Result<Self> {
        let len = info.byte_size as usize / std::mem::size_of::<T>();

        let mut b = info.clone();
        let staging_name = format!("{} Staging Buffer", info.debug_name);
        b.visibility = MemoryVisibility::Gpu;
        let buffer = ctx.make_buffer(&b)?;

        b.debug_name = &staging_name;
        b.visibility = MemoryVisibility::CpuAndGpu;
        let staging = ctx.make_buffer(&b)?;

        let mapped = ctx.map_buffer_mut::<u8>(BufferView::new(staging))?;
        let pool = Pool::new_preallocated(mapped.as_mut_ptr(), len);

        Ok(Self {
            buffer,
            staging,
            pool,
            dirty: vec![false; len],
            _ctx: ctx,
        })
    }

    pub fn destroy(&mut self) -> Result<()> {
        unsafe { &*self._ctx }.unmap_buffer(self.staging)?;
        Ok(())
    }

    pub fn get_gpu_handle(&self) -> Handle<Buffer> {
        return self.buffer;
    }

    pub fn sync_down(&mut self) -> Result<CommandStream<Recording>> {
        let list = CommandStream::new().begin().copy_buffers(&CopyBuffer {
            src: self.buffer,
            dst: self.staging,
            ..Default::default()
        });

        Ok(list)
    }

    pub fn sync_up(&mut self) -> Result<CommandStream<Recording>> {
        let list = CommandStream::new().begin().copy_buffers(&CopyBuffer {
            src: self.staging,
            dst: self.buffer,
            ..Default::default()
        });
        self.clear_dirty();

        Ok(list)
    }

    pub fn sync_up_dirty(&mut self) -> Result<CommandStream<Recording>> {
        let mut list = CommandStream::new().begin();
        let item_size = std::mem::size_of::<T>();
        let mut idx = 0usize;
        while idx < self.dirty.len() {
            if !self.dirty[idx] {
                idx += 1;
                continue;
            }
            let start = idx;
            while idx < self.dirty.len() && self.dirty[idx] {
                idx += 1;
            }
            let count = idx - start;
            let byte_offset = (start * item_size) as u32;
            let byte_amount = (count * item_size) as u32;
            list = list.copy_buffers(&CopyBuffer {
                src: self.staging,
                dst: self.buffer,
                src_offset: byte_offset,
                dst_offset: byte_offset,
                amount: byte_amount,
            });
        }
        self.clear_dirty();

        Ok(list)
    }

    pub fn get_empty(&self) -> &[u32] {
        &self.pool.get_empty()
    }

    pub fn insert(&mut self, item: T) -> Option<Handle<T>> {
        let handle = self.pool.insert(item);
        if let Some(handle) = handle {
            self.mark_dirty(handle.slot);
        }
        handle
    }

    pub fn for_each_occupied<F>(&self, func: F)
    where
        F: FnMut(&T),
    {
        self.pool.for_each_occupied(func);
    }

    pub fn len(&self) -> usize {
        return self.pool.len();
    }

    pub fn for_each_occupied_mut<F>(&mut self, func: F)
    where
        F: FnMut(&mut T),
    {
        if !self.dirty.is_empty() {
            self.dirty.fill(true);
        }
        self.pool.for_each_occupied_mut(func);
    }

    pub fn release(&mut self, item: Handle<T>) {
        self.pool.release(item);
        self.clear_dirty_slot(item.slot);
    }

    pub fn get_ref(&self, item: Handle<T>) -> Option<&T> {
        self.pool.get_ref(item)
    }

    pub fn get_mut_ref(&mut self, item: Handle<T>) -> Option<&mut T> {
        self.mark_dirty(item.slot);
        self.pool.get_mut_ref(item)
    }

    pub fn clear(&mut self) {
        self.pool.clear();
        self.clear_dirty();
    }

    fn mark_dirty(&mut self, slot: u16) {
        let idx = slot as usize;
        if let Some(entry) = self.dirty.get_mut(idx) {
            *entry = true;
        }
    }

    fn clear_dirty_slot(&mut self, slot: u16) {
        let idx = slot as usize;
        if let Some(entry) = self.dirty.get_mut(idx) {
            *entry = false;
        }
    }

    fn clear_dirty(&mut self) {
        if !self.dirty.is_empty() {
            self.dirty.fill(false);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////
///
///

impl DynamicGPUPool {
    pub fn new(
        ctx: &mut Context,
        info: &BufferInfo,
        item_size: usize,
        item_align: usize,
    ) -> Result<Self> {
        let len = info.byte_size as usize / item_size;

        let mut b = info.clone();
        let staging_name = format!("{} Staging Buffer", info.debug_name);
        b.visibility = MemoryVisibility::Gpu;
        let buffer = ctx.make_buffer(&b)?;

        b.debug_name = &staging_name;
        b.visibility = MemoryVisibility::CpuAndGpu;
        let staging = ctx.make_buffer(&b)?;

        let mapped = ctx.map_buffer_mut::<u8>(BufferView::new(staging))?;
        let pool = DynamicPool::new_preallocated(
            mapped.as_mut_ptr(),
            len,
            item_size as u32,
            item_align as u32,
        );

        Ok(Self {
            buffer,
            staging,
            pool,
            dirty: vec![false; len],
            item_size,
            _ctx: ctx,
        })
    }

    pub fn destroy(&mut self) -> Result<()> {
        unsafe { &*self._ctx }.unmap_buffer(self.staging)?;
        Ok(())
    }

    pub fn get_gpu_handle(&self) -> Handle<Buffer> {
        return self.buffer;
    }

    pub fn sync_down(&mut self) -> Result<CommandStream<Recording>> {
        let list = CommandStream::new().begin().copy_buffers(&CopyBuffer {
            src: self.buffer,
            dst: self.staging,
            ..Default::default()
        });

        Ok(list)
    }

    pub fn sync_up(&mut self) -> Result<CommandStream<Recording>> {
        let list = CommandStream::new().begin().copy_buffers(&CopyBuffer {
            src: self.staging,
            dst: self.buffer,
            ..Default::default()
        });
        self.clear_dirty();

        Ok(list)
    }

    pub fn sync_up_dirty(&mut self) -> Result<CommandStream<Recording>> {
        let mut list = CommandStream::new().begin();
        let mut idx = 0usize;
        while idx < self.dirty.len() {
            if !self.dirty[idx] {
                idx += 1;
                continue;
            }
            let start = idx;
            while idx < self.dirty.len() && self.dirty[idx] {
                idx += 1;
            }
            let count = idx - start;
            let byte_offset = (start * self.item_size) as u32;
            let byte_amount = (count * self.item_size) as u32;
            list = list.copy_buffers(&CopyBuffer {
                src: self.staging,
                dst: self.buffer,
                src_offset: byte_offset,
                dst_offset: byte_offset,
                amount: byte_amount,
            });
        }
        self.clear_dirty();

        Ok(list)
    }

    pub fn get_empty(&self) -> &[u32] {
        &self.pool.get_empty()
    }

    pub fn insert<T>(&mut self, item: T) -> Option<Handle<T>> {
        let handle = self.pool.insert(item);
        if let Some(handle) = handle {
            self.mark_dirty(handle.slot);
        }
        handle
    }

    //    pub fn for_each_occupied<F>(&self, func: F)
    //    where
    //        F: FnMut(&T),
    //    {
    //        self.pool.for_each_occupied(func);
    //    }
    //
    //    pub fn len(&self) -> usize {
    //        return self.pool.len();
    //    }
    //
    //    pub fn for_each_occupied_mut<F>(&mut self, func: F)
    //    where
    //        F: FnMut(&mut T),
    //    {
    //        self.pool.for_each_occupied_mut(func);
    //    }

    pub fn release<T>(&mut self, item: Handle<T>) {
        self.pool.release(item);
        self.clear_dirty_slot(item.slot);
    }

    pub fn get_ref<T>(&self, item: Handle<T>) -> Option<&T> {
        self.pool.get_ref(item)
    }

    pub fn get_mut_ref<T>(&mut self, item: Handle<T>) -> Option<&mut T> {
        self.mark_dirty(item.slot);
        self.pool.get_mut_ref(item)
    }

    pub fn clear(&mut self) {
        self.pool.clear();
        self.clear_dirty();
    }

    fn mark_dirty(&mut self, slot: u16) {
        let idx = slot as usize;
        if let Some(entry) = self.dirty.get_mut(idx) {
            *entry = true;
        }
    }

    fn clear_dirty_slot(&mut self, slot: u16) {
        let idx = slot as usize;
        if let Some(entry) = self.dirty.get_mut(idx) {
            *entry = false;
        }
    }

    fn clear_dirty(&mut self) {
        if !self.dirty.is_empty() {
            self.dirty.fill(false);
        }
    }
}

///
#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    #[test]
    #[serial]
    fn test_gpu_pool() {
        const TEST_AMT: usize = 2048;
        #[derive(Default)]
        struct S {
            _big_data: [u32; 16],
        }

        let mut ctx = Context::headless(&Default::default()).unwrap();

        let mut pool = GPUPool::new(
            &mut ctx,
            &BufferInfo {
                debug_name: "Test Buffer",
                byte_size: (std::mem::size_of::<S>() * TEST_AMT) as u32,
                visibility: MemoryVisibility::CpuAndGpu,
                initial_data: None,
                ..Default::default()
            },
        )
        .unwrap();

        assert!(pool.len() == TEST_AMT);

        let mut p = Vec::new();

        for _it in 0..TEST_AMT {
            p.push(pool.insert(S::default()).expect("ASSERT: Should insert."));
        }

        for _it in 0..TEST_AMT {
            assert!(pool.insert(S::default()) == None);
        }
        assert!(pool.len() == TEST_AMT);

        let mut list = ctx
            .begin_command_queue(QueueType::Graphics, "", false)
            .unwrap();

        let stream = pool.sync_up().unwrap();
        ctx.submit(&mut list, &Default::default())
            .expect("ASSERT: Should be able to sync data up");

        ctx.sync_current_device();
        pool.destroy().unwrap();
        ctx.destroy();
    }
}
