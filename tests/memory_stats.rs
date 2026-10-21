use ash::vk;
use dashi::{BufferInfo, BufferUsage, Context, ContextInfo, DeviceFilter, DeviceSelector};
use std::alloc::{Layout, alloc, dealloc};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

static CREATED: AtomicU64 = AtomicU64::new(0);
static FREED: AtomicU64 = AtomicU64::new(0);
const HEADER: usize = 3 * size_of::<usize>();

unsafe extern "system" fn allocate(
    _user: *mut c_void,
    size: usize,
    alignment: usize,
    _scope: vk::SystemAllocationScope,
) -> *mut c_void {
    let alignment = alignment.max(align_of::<usize>());
    let Some(total) = size
        .checked_add(alignment)
        .and_then(|n| n.checked_add(HEADER))
    else {
        return std::ptr::null_mut();
    };
    let Ok(layout) = Layout::from_size_align(total, align_of::<usize>()) else {
        return std::ptr::null_mut();
    };
    let base = unsafe { alloc(layout) };
    if base.is_null() {
        return std::ptr::null_mut();
    }
    let address = (base as usize + HEADER + alignment - 1) & !(alignment - 1);
    let header = (address - HEADER) as *mut usize;
    unsafe {
        header.write(base as usize);
        header.add(1).write(total);
        header.add(2).write(size);
    }
    CREATED.fetch_add(1, Ordering::Relaxed);
    address as *mut c_void
}

unsafe extern "system" fn free(_user: *mut c_void, pointer: *mut c_void) {
    if pointer.is_null() {
        return;
    }
    let header = (pointer as usize - HEADER) as *const usize;
    let (base, total) = unsafe { (header.read(), header.add(1).read()) };
    let layout = Layout::from_size_align(total, align_of::<usize>()).unwrap();
    unsafe { dealloc(base as *mut u8, layout) };
    FREED.fetch_add(1, Ordering::Relaxed);
}

unsafe extern "system" fn reallocate(
    user: *mut c_void,
    pointer: *mut c_void,
    size: usize,
    alignment: usize,
    scope: vk::SystemAllocationScope,
) -> *mut c_void {
    if pointer.is_null() {
        return unsafe { allocate(user, size, alignment, scope) };
    }
    if size == 0 {
        unsafe { free(user, pointer) };
        return std::ptr::null_mut();
    }
    let header = (pointer as usize - HEADER) as *const usize;
    let old_size = unsafe { header.add(2).read() };
    let replacement = unsafe { allocate(user, size, alignment, scope) };
    if !replacement.is_null() {
        unsafe {
            std::ptr::copy_nonoverlapping(
                pointer.cast::<u8>(),
                replacement.cast::<u8>(),
                old_size.min(size),
            );
            free(user, pointer);
        }
    }
    replacement
}

#[test]
fn vulkan_host_callbacks_and_vma_stats_follow_resource_lifetime() {
    let Ok(selector) = DeviceSelector::new() else {
        return;
    };
    let Some(device) = selector.select(DeviceFilter::default()) else {
        return;
    };
    let callbacks = vk::AllocationCallbacks {
        p_user_data: std::ptr::null_mut(),
        pfn_allocation: Some(allocate),
        pfn_reallocation: Some(reallocate),
        pfn_free: Some(free),
        pfn_internal_allocation: None,
        pfn_internal_free: None,
    };
    let mut ctx = Context::headless(&ContextInfo {
        device,
        vulkan_allocation_callbacks: Some(callbacks),
        ..Default::default()
    })
    .expect("headless Vulkan context");
    let before = ctx.gpu_memory_stats().unwrap().unwrap();
    let buffer = ctx
        .make_buffer(&BufferInfo {
            debug_name: "memory stats test",
            byte_size: 4096,
            usage: BufferUsage::STORAGE,
            ..Default::default()
        })
        .unwrap();
    let during = ctx.gpu_memory_stats().unwrap().unwrap();
    assert!(during.allocations_created > before.allocations_created);
    assert!(
        during.heaps.iter().map(|h| h.allocation_count).sum::<u64>()
            > before.heaps.iter().map(|h| h.allocation_count).sum::<u64>()
    );
    ctx.destroy_buffer(buffer);
    let after = ctx.gpu_memory_stats().unwrap().unwrap();
    assert_eq!(after.allocations_freed, during.allocations_freed + 1);
    let vertex = inline_spirv::inline_spirv!(
        "#version 450\nvoid main(){gl_Position=vec4(0,0,0,1);}",
        vert
    );
    let fragment = inline_spirv::inline_spirv!(
        "#version 450\nlayout(location=0) out vec4 color;void main(){color=vec4(1);}",
        frag
    );
    let layout = ctx
        .make_graphics_pipeline_layout(&dashi::GraphicsPipelineLayoutInfo {
            debug_name: "allocator retirement",
            bt_layouts: [None; 4],
            shaders: &[
                dashi::PipelineShaderInfo {
                    stage: dashi::ShaderType::Vertex,
                    spirv: vertex,
                    entry_point: "main",
                    specialization: &[],
                },
                dashi::PipelineShaderInfo {
                    stage: dashi::ShaderType::Fragment,
                    spirv: fragment,
                    entry_point: "main",
                    specialization: &[],
                },
            ],
            vertex_info: dashi::VertexDescriptionInfo {
                entries: &[],
                stride: 0,
                rate: dashi::VertexRate::Vertex,
            },
            details: dashi::GraphicsPipelineDetails {
                color_blend_states: vec![dashi::ColorBlendState {
                    enable: false,
                    ..Default::default()
                }],
                ..Default::default()
            },
        })
        .unwrap();
    let formats = vec![dashi::Format::RGBA8];
    let pipeline = ctx
        .make_graphics_pipeline(&dashi::GraphicsPipelineInfo {
            layout,
            attachment_formats: formats,
            subpass_samples: dashi::SubpassSampleInfo {
                color_samples: vec![dashi::SampleCount::S1],
                depth_sample: None,
            },
            ..Default::default()
        })
        .unwrap();
    let freed = FREED.load(Ordering::Relaxed);
    ctx.destroy_graphics_pipeline(pipeline);
    ctx.destroy_graphics_pipeline_layout(layout);
    assert!(
        FREED.load(Ordering::Relaxed) > freed,
        "pipeline retirement uses the supplied allocator"
    );
    ctx.destroy();
    assert!(CREATED.load(Ordering::Relaxed) > 0);
    assert_eq!(
        CREATED.load(Ordering::Relaxed),
        FREED.load(Ordering::Relaxed)
    );
}
