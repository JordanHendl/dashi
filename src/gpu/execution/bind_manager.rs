// src/gpu/vulkan/binding_manager.rs

use crate::gpu::vulkan::{
    BindGroup, BindGroupLayout, BindGroupLayoutInfo, BindGroupVariableType, BindTable,
    BindTableLayout, BindTableLayoutInfo, Context, ShaderInfo, ShaderType,
};
use crate::{hash_bind_group_layout_info, hash_bind_table_layout_info};
use crate::utils::Handle;

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex, RwLock},
};


/// ----- Keys used by the caches ------------------------------------------------

#[derive(Clone, Debug, Eq)]
pub struct LayoutKey(pub u64);
impl PartialEq for LayoutKey {
    fn eq(&self, o: &Self) -> bool { self.0 == o.0 }
}
impl Hash for LayoutKey {
    fn hash<H: Hasher>(&self, state: &mut H) { self.0.hash(state) }
}

#[derive(Clone, Debug, Eq)]
pub struct TableLayoutKey(pub u64);
impl PartialEq for TableLayoutKey {
    fn eq(&self, o: &Self) -> bool { self.0 == o.0 }
}
impl Hash for TableLayoutKey {
    fn hash<H: Hasher>(&self, state: &mut H) { self.0.hash(state) }
}

impl From<&BindGroupLayoutInfo<'_>> for LayoutKey {
    fn from(info: &BindGroupLayoutInfo<'_>) -> Self {
        LayoutKey(hash_bind_group_layout_info(info))
    }
}
impl From<&BindTableLayoutInfo<'_>> for TableLayoutKey {
    fn from(info: &BindTableLayoutInfo<'_>) -> Self {
        TableLayoutKey(hash_bind_table_layout_info(info))
    }
}

/// ----- Per-thread freelists for a given frame ---------------------------------

#[derive(Default)]
struct ThreadPools {
    free_bind_groups: VecDeque<Handle<BindGroup>>,
    free_bind_tables: VecDeque<Handle<BindTable>>,
}

struct FramePools {
    per_thread: Vec<Mutex<ThreadPools>>,
}
impl FramePools {
    fn new(thread_count: usize) -> Self {
        let mut per_thread = Vec::with_capacity(thread_count);
        for _ in 0..thread_count {
            per_thread.push(Mutex::new(ThreadPools::default()));
        }
        Self { per_thread }
    }
}

/// ----- BindingManager ----------------------------------------------------------

pub struct BindingManager {
    ctx: *mut Context,
    frames_in_flight: usize,
    thread_count: usize,

    // Global caches (shared across frames/threads)
    bgl_cache: RwLock<HashMap<LayoutKey, Handle<BindGroupLayout>>>,
    btl_cache: RwLock<HashMap<TableLayoutKey, Handle<BindTableLayout>>>,

    // Per-frame reuse pools
    frames: Vec<FramePools>,
}

unsafe impl Send for BindingManager {}
unsafe impl Sync for BindingManager {}

impl BindingManager {
    pub fn new(ctx: *mut Context, frames_in_flight: usize, thread_count: usize) -> Arc<Self> {
        let mut frames = Vec::with_capacity(frames_in_flight);
        for _ in 0..frames_in_flight {
            frames.push(FramePools::new(thread_count));
        }
        Arc::new(Self {
            ctx,
            frames_in_flight,
            thread_count,
            bgl_cache: RwLock::new(HashMap::new()),
            btl_cache: RwLock::new(HashMap::new()),
            frames,
        })
    }

    #[inline] pub fn frames_in_flight(&self) -> usize { self.frames_in_flight }
    #[inline] pub fn thread_count(&self) -> usize { self.thread_count }

    // ----- Layout cache: key-taking versions -----------------------------------

    pub fn get_or_create_bind_group_layout<F>(
        &self,
        key: LayoutKey,
        create: F,
    ) -> Handle<BindGroupLayout>
    where
        F: FnOnce(&mut Context) -> Handle<BindGroupLayout>,
    {
        if let Some(h) = self.bgl_cache.read().unwrap().get(&key).cloned() {
            return h;
        }
        let mut w = self.bgl_cache.write().unwrap();
        if let Some(h) = w.get(&key).cloned() {
            return h;
        }
        let h = create(unsafe { &mut *self.ctx });
        w.insert(key, h);
        h
    }

    pub fn get_or_create_bind_table_layout<F>(
        &self,
        key: TableLayoutKey,
        create: F,
    ) -> Handle<BindTableLayout>
    where
        F: FnOnce(&mut Context) -> Handle<BindTableLayout>,
    {
        if let Some(h) = self.btl_cache.read().unwrap().get(&key).cloned() {
            return h;
        }
        let mut w = self.btl_cache.write().unwrap();
        if let Some(h) = w.get(&key).cloned() {
            return h;
        }
        let h = create(unsafe { &mut *self.ctx });
        w.insert(key, h);
        h
    }

    // ----- Layout cache: direct-Info convenience helpers -----------------------

    /// Convenience: pass a BindGroupLayoutInfo directly.
    pub fn get_or_create_bgl_from_info<F>(
        &self,
        info: &BindGroupLayoutInfo<'_>,
        create: F,
    ) -> Handle<BindGroupLayout>
    where
        F: FnOnce(&mut Context, &BindGroupLayoutInfo<'_>) -> Handle<BindGroupLayout>,
    {
        let key = LayoutKey::from(info);
        self.get_or_create_bind_group_layout(key, |ctx| create(ctx, info))
    }

    /// Convenience: pass a BindTableLayoutInfo directly.
    pub fn get_or_create_btl_from_info<F>(
        &self,
        info: &BindTableLayoutInfo<'_>,
        create: F,
    ) -> Handle<BindTableLayout>
    where
        F: FnOnce(&mut Context, &BindTableLayoutInfo<'_>) -> Handle<BindTableLayout>,
    {
        let key = TableLayoutKey::from(info);
        self.get_or_create_bind_table_layout(key, |ctx| create(ctx, info))
    }

    // ----- Instance reuse (per-frame, per-thread) -------------------------------

    pub fn alloc_bind_group<C, U>(
        &self,
        frame_idx: usize,
        thread_idx: usize,
        create: C,
        update: U,
    ) -> Handle<BindGroup>
    where
        C: FnOnce(&mut Context) -> Handle<BindGroup>,
        U: FnOnce(&mut Context, Handle<BindGroup>),
    {
        let pools = &self.frames[frame_idx % self.frames_in_flight].per_thread;
        let mut tp = pools[thread_idx % self.thread_count].lock().unwrap();

        if let Some(h) = tp.free_bind_groups.pop_front() {
            update(unsafe { &mut *self.ctx }, h);
            h
        } else {
            create(unsafe { &mut *self.ctx })
        }
    }

    pub fn recycle_bind_group(&self, frame_idx: usize, thread_idx: usize, h: Handle<BindGroup>) {
        let pools = &self.frames[frame_idx % self.frames_in_flight].per_thread;
        let mut tp = pools[thread_idx % self.thread_count].lock().unwrap();
        tp.free_bind_groups.push_back(h);
    }

    pub fn alloc_bind_table<C, U>(
        &self,
        frame_idx: usize,
        thread_idx: usize,
        create: C,
        update: U,
    ) -> Handle<BindTable>
    where
        C: FnOnce(&mut Context) -> Handle<BindTable>,
        U: FnOnce(&mut Context, Handle<BindTable>),
    {
        let pools = &self.frames[frame_idx % self.frames_in_flight].per_thread;
        let mut tp = pools[thread_idx % self.thread_count].lock().unwrap();

        if let Some(h) = tp.free_bind_tables.pop_front() {
            update(unsafe { &mut *self.ctx }, h);
            h
        } else {
            create(unsafe { &mut *self.ctx })
        }
    }

    pub fn recycle_bind_table(&self, frame_idx: usize, thread_idx: usize, h: Handle<BindTable>) {
        let pools = &self.frames[frame_idx % self.frames_in_flight].per_thread;
        let mut tp = pools[thread_idx % self.thread_count].lock().unwrap();
        tp.free_bind_tables.push_back(h);
    }

    pub fn begin_frame(&self, _frame_idx: usize) {
        // Optional place to cap freelists or roll windows if desired.
    }

    pub fn drain_frame_bind_groups<D>(&self, frame_idx: usize, destroy: D)
    where
        D: Fn(&mut Context, Handle<BindGroup>),
    {
        let pools = &self.frames[frame_idx % self.frames_in_flight].per_thread;
        for tp in pools {
            let mut tp = tp.lock().unwrap();
            while let Some(h) = tp.free_bind_groups.pop_front() {
                destroy(unsafe { &mut *self.ctx }, h);
            }
        }
    }

    pub fn drain_frame_bind_tables<D>(&self, frame_idx: usize, destroy: D)
    where
        D: Fn(&mut Context, Handle<BindTable>),
    {
        let pools = &self.frames[frame_idx % self.frames_in_flight].per_thread;
        for tp in pools {
            let mut tp = tp.lock().unwrap();
            while let Some(h) = tp.free_bind_tables.pop_front() {
                destroy(unsafe { &mut *self.ctx }, h);
            }
        }
    }
}
