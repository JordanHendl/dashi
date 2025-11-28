use resource_pool::resource_list::ResourceList;

use crate::{IndexedResource, ShaderResource};

pub type Handle<T> = resource_pool::Handle<T>;
pub type Pool<T> = resource_pool::Pool<T>;
pub mod gpupool;

pub fn resource_list_to_indexed_resources(list: &ResourceList<ShaderResource>) -> Vec<IndexedResource> {
    let mut v = Vec::new();
    let mut slot = 0;
    list.pool.for_each_occupied(|f| {
        slot += 1;
        v.push(IndexedResource {
            resource: f.clone(),
            slot: slot - 1,
        });
    });

    v
}
