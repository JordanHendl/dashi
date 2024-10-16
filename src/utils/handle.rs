use std::hash::Hash;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Handle<T> {
    pub slot: u16,
    pub generation: u16,
    phantom: PhantomData<T>,
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
            slot: Default::default(),
            generation: Default::default(),
            phantom: Default::default(),
        }
    }
}

pub struct Pool<T> {
    items: Vec<Option<T>>,
    empty: Vec<usize>,
    generation: Vec<u16>,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        const INITIAL_SIZE: usize = 1024;
        let mut p = Pool {
            items: Vec::with_capacity(INITIAL_SIZE),
            empty: Vec::with_capacity(INITIAL_SIZE),
            generation: vec![0; INITIAL_SIZE],
        };

        p.empty = (0..INITIAL_SIZE).collect();
        p.items.resize_with(INITIAL_SIZE, || None);
        return p;
    }
}
impl<T> Pool<T> {
    pub fn new(initial_size: usize) -> Self {
        let mut p = Pool {
            items: Vec::with_capacity(initial_size),
            empty: Vec::with_capacity(initial_size),
            generation: vec![0; initial_size],
        };

        p.empty = (0..initial_size).collect();
        p.items.resize_with(initial_size, || None);

        return p;
    }

    pub fn insert(&mut self, item: T) -> Option<Handle<T>> {
        let empty_slot = self.empty.pop()?;

        self.items[empty_slot] = Some(item);

        return Some(Handle {
            slot: empty_slot as u16,
            generation: self.generation[empty_slot],
            phantom: PhantomData,
        });
    }

    pub fn release(&mut self, item: Handle<T>) {
        self.empty.push(item.slot as usize);
    }

    pub fn get_ref(&self, item: Handle<T>) -> Option<&T> {
        let slot = item.slot as usize;
        if self.generation[slot] == item.generation {
            return Some(&self.items[slot].as_ref().unwrap());
        } else {
            None
        }
    }

    pub fn get_mut_ref(&mut self, item: Handle<T>) -> Option<&mut T> {
        let slot = item.slot as usize;
        if self.generation[slot] == item.generation {
            return Some(self.items[slot].as_mut().unwrap());
        } else {
            None
        }
    }
}
