use utils::Pool;

use crate::*;

pub struct ResourceList<T> {
    pub pool: Pool<T>,
    pub entries: Vec<Handle<T>>,
}

impl<T> Default for ResourceList<T> {
    fn default() -> Self {
        Self {
            pool: Default::default(),
            entries: Default::default(),
        }
    }
}

#[allow(dead_code)]
impl<T> ResourceList<T> {
    pub fn new(size: usize) -> Self {
        Self {
            pool: Pool::new(size),
            entries: Vec::with_capacity(size),
        }
    }

    pub fn push(&mut self, v: T) -> Handle<T> {
        let h = self.pool.insert(v).unwrap();
        self.entries.push(h);
        h
    }

    pub fn release(&mut self, h: Handle<T>) {
        if let Some(idx) = self.entries.iter().position(|a| a.slot == h.slot) {
            self.entries.remove(idx);
            self.pool.release(h);
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn get_ref(&self, h: Handle<T>) -> &T {
        self.pool.get_ref(h).unwrap()
    }

    pub fn get_ref_mut(&mut self, h: Handle<T>) -> &mut T {
        self.pool.get_mut_ref(h).unwrap()
    }

    #[allow(dead_code)]
    pub fn for_each_occupied<F>(&self, func: F)
    where
        F: Fn(&T),
    {
        for item in &self.entries {
            let r = self.pool.get_ref(item.clone()).unwrap();
            func(r);
        }
    }

    pub fn for_each_handle<F>(&self, mut func: F)
    where
        F: FnMut(Handle<T>),
    {
        for h in &self.entries {
            func(*h);
        }
    }

    #[allow(dead_code)]
    pub fn for_each_occupied_mut<F>(&mut self, mut func: F)
    where
        F: FnMut(&T),
    {
        for item in &self.entries {
            let r = self.pool.get_mut_ref(item.clone()).unwrap();
            func(r);
        }
    }
}

impl ResourceList<ShaderResource> {
    pub fn to_indexed_resources(&self) -> Vec<IndexedResource> {
        let mut v = Vec::new();
        let mut slot = 0;
        self.pool.for_each_occupied(|f| {
            slot += 1;
            v.push(IndexedResource {
                resource: f.clone(),
                slot: slot - 1,
            });
        });

        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageView, Sampler};

    #[test]
    fn push_and_get_round_trip() {
        let mut list = ResourceList::default();
        let handle = list.push(42u32);
        assert_eq!(*list.get_ref(handle), 42);
        assert_eq!(*list.get_ref_mut(handle), 42);
    }

    #[test]
    fn release_removes_entries() {
        let mut list = ResourceList::new(4);
        let handle_a = list.push("a");
        let handle_b = list.push("b");

        list.release(handle_a);
        assert_eq!(list.entries.len(), 1);
        assert_eq!(*list.get_ref(handle_b), "b");
    }

    #[test]
    fn to_indexed_resources_assigns_slots() {
        let mut list = ResourceList::default();
        let res_a =
            ShaderResource::SampledImage(ImageView::default(), Handle::<Sampler>::new(5, 0));
        let res_b =
            ShaderResource::SampledImage(ImageView::default(), Handle::<Sampler>::new(6, 0));
        list.push(res_a.clone());
        list.push(res_b.clone());

        let indexed = list.to_indexed_resources();
        assert_eq!(indexed.len(), 2);
        assert_eq!(indexed[0].slot, 0);
        assert_eq!(indexed[1].slot, 1);
        assert!(matches!(
            indexed[0].resource,
            ShaderResource::SampledImage(_, _)
        ));
        assert!(matches!(
            indexed[1].resource,
            ShaderResource::SampledImage(_, _)
        ));
    }
}
