use crate::{Buffer, BufferCopy, BufferInfo, CommandList, Context, MemoryVisibility};

use super::{Handle, Pool};

pub struct GPUPool<T> {
    buffer: Handle<Buffer>,
    staging: Handle<Buffer>,
    pool: Pool<T>,
    _ctx: *mut Context,
}

impl<T> GPUPool<T> {
    pub fn new(ctx: &mut Context, info: &BufferInfo) -> Self {
        let len = info.byte_size as usize / std::mem::size_of::<T>();

        let mut b = info.clone();
        let staging_name = format!("{} Staging Buffer", info.debug_name);
        b.visibility = MemoryVisibility::Gpu;
        let buffer = ctx.make_buffer(&b).unwrap();

        b.debug_name = &staging_name;
        b.visibility = MemoryVisibility::CpuAndGpu;
        let staging = ctx.make_buffer(&b).unwrap();

        let mapped = ctx.map_buffer_mut::<u8>(staging).unwrap();
        let pool = Pool::new_preallocated(mapped.as_mut_ptr(), len);

        Self {
            buffer,
            staging,
            pool,
            _ctx: ctx,
        }
    }

    pub fn sync_down(&mut self, list: &mut CommandList) {
        list.copy_buffers(&BufferCopy {
            src: self.buffer,
            dst: self.staging,
            ..Default::default()
        });
    }

    pub fn sync_up(&mut self, list: &mut CommandList) {
        list.copy_buffers(&BufferCopy {
            src: self.staging,
            dst: self.buffer,
            ..Default::default()
        });
    }

    pub fn get_empty(&self) -> &[u32] {
        &self.pool.get_empty()
    }

    pub fn insert(&mut self, item: T) -> Option<Handle<T>> {
        return self.pool.insert(item);
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
        self.pool.for_each_occupied_mut(func);
    }

    pub fn release(&mut self, item: Handle<T>) {
        self.pool.release(item);
    }

    pub fn get_ref(&self, item: Handle<T>) -> Option<&T> {
        self.pool.get_ref(item)
    }

    pub fn get_mut_ref(&mut self, item: Handle<T>) -> Option<&mut T> {
        self.pool.get_mut_ref(item)
    }

    pub fn clear(&mut self) {
        self.pool.clear();
    }
}

#[test]
fn test_gpu_pool() {
    const TEST_AMT: usize = 2048;
    #[derive(Default)]
    struct S {
        _big_data: [u32; 16],
    }

    let mut ctx = Context::new(&Default::default()).unwrap();

    let mut pool = GPUPool::new(&mut ctx, &BufferInfo {
        debug_name: "Test Buffer",
        byte_size: (std::mem::size_of::<S>() * TEST_AMT) as u32,
        visibility: MemoryVisibility::CpuAndGpu,
        initial_data: None,
        ..Default::default()
    });

    assert!(pool.len() == TEST_AMT);

    let mut p = Vec::new();

    for _it in 0..TEST_AMT {
        p.push(pool.insert(S::default()).expect("ASSERT: Should insert."));
    }

    for _it in 0..TEST_AMT {
        assert!(pool.insert(S::default()) == None);
    }
    assert!(pool.len() == TEST_AMT);


    let mut list = ctx.begin_command_list(&Default::default()).unwrap();
    pool.sync_up(&mut list);
    ctx.submit(&mut list, &Default::default()).expect("ASSERT: Should be able to sync data up");
}
