use std::alloc::{alloc_zeroed, Layout};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
#[repr(C)]
#[derive(Debug)]
pub struct Handle<T> {
    pub slot: u16,
    pub generation: u16,
    phantom: PhantomData<T>,
}

impl<T> Handle<T> {
    pub fn valid(&self) -> bool {
        return self.slot != std::u16::MAX && self.generation != std::u16::MAX;
    }
}

impl<T> Eq for Handle<T> {
}


impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.slot == other.slot && self.generation == other.generation
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            slot: self.slot.clone(),
            generation: self.generation.clone(),
            phantom: self.phantom.clone(),
        }
    }
}

impl<T> Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.slot.hash(state);
        self.generation.hash(state);
        self.phantom.hash(state);
    }
}

impl<T> Copy for Handle<T> {}
impl<T> Default for Handle<T> {
    fn default() -> Self {
        Self {
            slot: std::u16::MAX,
            generation: std::u16::MAX,
            phantom: Default::default(),
        }
    }
}

struct ItemList<T> {
    items: *mut T,
    end: *mut T,
    imported: bool,
    phantom: PhantomData<T>,
}

impl<T> ItemList<T> {
    fn new(len: u32) -> Self {
        unsafe {
            let byte_size = len as usize * std::mem::size_of::<T>();
            let layout = Layout::from_size_align(byte_size, 1).unwrap();
            let ptr = alloc_zeroed(layout);
            Self {
                items: ptr as *mut T,
                end: (ptr as *mut T).offset(len as isize),
                phantom: PhantomData::default(),
                imported: false,
            }
        }
    }

    fn new_from_prealloc(ptr: *mut u8, len: u32) -> Self {
        Self {
            items: ptr as *mut T,
            end: unsafe { (ptr as *mut T).offset(len as isize) },
            phantom: PhantomData::default(),
            imported: true,
        }
    }

    fn byte_size(&self) -> usize {
        return self.len() * std::mem::size_of::<T>();
    }

//    fn as_slice(&self) -> &[T] {
//        return unsafe { std::slice::from_raw_parts(self.items, self.len()) };
//    }
//
//    fn as_slice_mut(&self) -> &mut [T] {
//        return unsafe { std::slice::from_raw_parts_mut(self.items, self.len()) };
//    }

    fn expand(&mut self, amt: usize) {
        if !self.imported {
            let len = self.len() + amt;
            unsafe {
                let byte_size = len as usize * std::mem::size_of::<T>();
                let layout = Layout::from_size_align(byte_size, 1).unwrap();
                let ptr = alloc_zeroed(layout);

                let src = std::slice::from_raw_parts(self.items as *const u8, self.byte_size());
                let dst = std::slice::from_raw_parts_mut(ptr, byte_size);

                dst[0..src.len()].copy_from_slice(src);

                self.items = ptr as *mut T;
                self.end = self.items.offset(len as isize);
            }
        }
    }

    fn len(&self) -> usize {
        return unsafe { self.end.offset_from(self.items) as usize };
    }

//    fn free(&self) {
//        if !self.imported {
//            let byte_size = self.len() as usize * std::mem::size_of::<T>();
//            let layout = Layout::from_size_align(byte_size, std::mem::size_of::<T>()).unwrap();
//            unsafe { dealloc(self.items as *mut u8, layout) };
//        }
//    }

    fn iter(&self) -> ItemListRef<'_, T> {
        ItemListRef {
            holder: self,
            curr: 0,
        }
    }
    fn iter_mut(&mut self) -> ItemListRefMut<'_, T> {
        ItemListRefMut {
            holder: self,
            curr: 0,
        }
    }
}

impl<T> IndexMut<usize> for ItemList<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        let v = unsafe { self.items.offset(index as isize) };
        return unsafe { v.as_mut().unwrap() };
    }
}

impl<T> Index<usize> for ItemList<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        let v = unsafe { self.items.offset(index as isize) };
        return unsafe { v.as_mut().unwrap() };
    }
}

struct ItemListRef<'a, T> {
    holder: &'a ItemList<T>,
    curr: usize,
}

struct ItemListRefMut<'a, T> {
    holder: &'a mut ItemList<T>,
    curr: usize,
}

impl<'a, T> Iterator for ItemListRefMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr != self.holder.len() {
            let ptr = self.holder.items;
            let c = self.curr;
            self.curr += 1;
            return Some(unsafe { ptr.offset(c as isize).as_mut().unwrap() });
        } else {
            return None;
        }
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.holder.len() - self.curr
    }
}

impl<'a, T> Iterator for ItemListRef<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr != self.holder.len() {
            let c = self.curr;
            self.curr += 1;
            return Some(&self.holder[c]);
        } else {
            return None;
        }
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.holder.len() - self.curr
    }
}

impl<'a, T> IntoIterator for &'a ItemList<T> {
    type Item = &'a T;

    type IntoIter = ItemListRef<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        ItemListRef {
            holder: self,
            curr: 0,
        }
    }
}

pub struct Pool<T> {
    items: ItemList<T>,
    empty: Vec<u32>,
    generation: Vec<u16>,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        const INITIAL_SIZE: usize = 1024;
        let mut p = Pool {
            items: ItemList::new(INITIAL_SIZE as u32),
            empty: Vec::with_capacity(INITIAL_SIZE),
            generation: vec![0; INITIAL_SIZE],
        };

        p.empty = (0..(INITIAL_SIZE) as u32).collect();
        assert!(!p.generation.is_empty());
        return p;
    }
}
impl<T> Pool<T> {
    pub fn new(initial_size: usize) -> Self {
        let mut p = Pool {
            items: ItemList::new(initial_size as u32),
            empty: Vec::with_capacity(initial_size),
            generation: vec![0; initial_size],
        };

        assert!(!p.generation.is_empty());
        p.empty = (0..(initial_size) as u32).collect();
        return p;
    }

    pub fn new_preallocated<G>(ptr: *mut G, len: usize) -> Self {
        let mut p = Pool {
            items: ItemList::new_from_prealloc(ptr as *mut u8, len as u32),
            empty: Vec::with_capacity(len),
            generation: vec![0; len],
        };

        p.empty = (0..(len) as u32).collect();
        return p;
    }

    pub fn get_empty(&self) -> &[u32] {
        &self.empty
    }

    pub fn insert(&mut self, item: T) -> Option<Handle<T>> {
        const DEFAULT_EXPAND_AMT: usize = 1024;
        if let Some(empty_slot) = self.empty.pop() {
            self.items[empty_slot as usize] = item;

            assert!(!self.generation.is_empty());
            return Some(Handle {
                slot: empty_slot as u16,
                generation: self.generation[empty_slot as usize],
                phantom: PhantomData,
            });
        } else {
            self.expand(DEFAULT_EXPAND_AMT);
            if let Some(empty_slot) = self.empty.pop() {
                self.items[empty_slot as usize] = item;

                assert!(!self.generation.is_empty());
                return Some(Handle {
                    slot: empty_slot as u16,
                    generation: self.generation[empty_slot as usize],
                    phantom: PhantomData,
                });
            }
        }
        return None;
    }

    pub fn expand(&mut self, amount: usize) {
        let old_len = self.items.len();
        self.items.expand(amount);

        println!("Expanded: {} -> {}", old_len, self.items.len());
        if self.items.len() > old_len {
            self.generation.resize_with(self.items.len(), || 0);
            for i in old_len..(self.items.len()) {
                self.empty.push(i as u32);
            }
        }
    }

    pub fn len(&self) -> usize {
        return self.items.len();
    }

    pub fn for_each_occupied<F>(&self, mut func: F)
    where
        F: FnMut(&T),
    {
        for (i, item) in self.items.iter().enumerate() {
            let c = i as u32;
            if !self.empty.contains(&c) {
                func(item);
            }
        }
    }

    pub fn for_each_occupied_mut<F>(&mut self, mut func: F)
    where
        F: FnMut(&mut T),
    {
        for (i, item) in self.items.iter_mut().enumerate() {
            let c = i as u32;
            if !self.empty.contains(&c) {
                func(item);
            }
        }
    }

    pub fn release(&mut self, item: Handle<T>) {
        self.empty.push(item.slot as u32);
    }

    pub fn get_ref(&self, item: Handle<T>) -> Option<&T> {
        assert!(item.valid());
        assert!(self.items.len() != 0);
        assert!(!self.generation.is_empty());
        let slot = item.slot as u32;
        if self.generation[slot as usize] == item.generation {
            return Some(&self.items[slot as usize]);
        } else {
            None
        }
    }

    pub fn get_mut_ref(&mut self, item: Handle<T>) -> Option<&mut T> {
        assert!(item.valid());
        assert!(!self.generation.is_empty());
        let slot = item.slot as usize;
        if self.generation[slot] == item.generation {
            return Some(&mut self.items[slot as usize]);
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        self.empty = (0..(self.items.len()) as u32).collect();
        self.generation.fill(0);
        assert!(self.generation.is_empty());
    }
}

#[test]
fn test_pool() {
    const TEST_AMT: usize = 2048;
    #[derive(Default, Debug)]
    struct S {
        _big_data: [u32; 16],
    }
    let mut pool: Pool<S> = Pool::new(TEST_AMT);
    assert!(pool.items.len() == TEST_AMT);

    let mut p = Vec::new();

    for _it in 0..TEST_AMT + 1 {
        p.push(pool.insert(S::default()).expect("ASSERT: Should insert."));
    }

    assert!(pool.items.len() > TEST_AMT);

    pool.for_each_occupied_mut(|f| {
        f._big_data[0] = 5;
        println!("{}", f._big_data[0]);
    });
}

#[test]
fn test_pool_imported() {
    const TEST_AMT: usize = 2048;
    #[derive(Default)]
    struct S {
        _big_data: [u32; 16],
    }
    let byte_size = TEST_AMT as usize * std::mem::size_of::<S>();
    let layout = Layout::from_size_align(byte_size, 1).unwrap();
    let ptr = unsafe { alloc_zeroed(layout) };

    let mut pool: Pool<S> = Pool::new_preallocated(ptr, TEST_AMT);
    assert!(pool.items.len() == TEST_AMT);

    let mut p = Vec::new();

    for _it in 0..TEST_AMT {
        p.push(pool.insert(S::default()).expect("ASSERT: Should insert."));
    }

    for _it in 0..TEST_AMT {
        assert!(pool.insert(S::default()) == None);
    }
    assert!(pool.items.len() == TEST_AMT);
}
