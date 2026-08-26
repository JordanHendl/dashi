mod common;

use common::ValidationContext;
use dashi::gpu::vulkan::*;
use serial_test::serial;

#[test]
#[serial]
fn bind_table_layout_exposes_descriptor_flags() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();

    let layout = ctx
        .make_bind_table_layout(&BindTableLayoutInfo {
            debug_name: "bind_table_flags",
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Compute,
                variables: &[BindTableVariable {
                    var_type: BindTableVariableType::Uniform,
                    binding: 0,
                    count: 4,
                }],
            }],
        })
        .unwrap();

    let flags = ctx.bind_table_layout_flags(layout);
    assert!(
        flags.update_after_bind,
        "bind tables must support update-after-bind"
    );
    assert!(
        flags.partially_bound,
        "bind tables must allow partially bound descriptors"
    );
}

#[test]
#[serial]
fn bind_table_updates_validate_variable_types() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();

    let layout = ctx
        .make_bind_table_layout(&BindTableLayoutInfo {
            debug_name: "bind_table_updates",
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Compute,
                variables: &[BindTableVariable {
                    var_type: BindTableVariableType::DynamicUniform,
                    binding: 0,
                    count: 1,
                }],
            }],
        })
        .unwrap();
    let mut allocator = ctx.make_dynamic_allocator(&Default::default()).unwrap();
    let mut allocator_b = ctx.make_dynamic_allocator(&Default::default()).unwrap();

    let table = ctx
        .make_bind_table(&BindTableInfo {
            debug_name: "bind_table_updates",
            layout,
            bindings: &[IndexedBindingInfo {
                resources: &[IndexedResource {
                    resource: ShaderResource::Dynamic(allocator.state().clone()),
                    slot: 0,
                }],
                binding: 0,
            }],
            ..Default::default()
        })
        .unwrap();

    ctx.update_bind_table(&BindTableUpdateInfo {
        table,
        bindings: &[IndexedBindingInfo {
            resources: &[IndexedResource {
                resource: ShaderResource::Dynamic(allocator_b.state().clone()),
                slot: 0,
            }],
            binding: 0,
        }],
    })
    .expect("update with dynamic uniform should succeed");

    let bad_buffer = ctx
        .make_buffer(&BufferInfo {
            debug_name: "wrong_type",
            byte_size: 64,
            visibility: MemoryVisibility::Gpu,
            usage: BufferUsage::STORAGE,
            ..Default::default()
        })
        .unwrap();

    let err = ctx
        .update_bind_table(&BindTableUpdateInfo {
            table,
            bindings: &[IndexedBindingInfo {
                resources: &[IndexedResource {
                    resource: ShaderResource::StorageBuffer(BufferView::new(bad_buffer)),
                    slot: 0,
                }],
                binding: 0,
            }],
        })
        .unwrap_err();

    match err {
        GPUError::InvalidBindTableBinding { .. } => {}
        other => panic!("unexpected error: {other:?}"),
    }

    ctx.destroy_buffer(bad_buffer);
    ctx.destroy_dynamic_allocator(allocator);
    ctx.destroy_dynamic_allocator(allocator_b);
}

#[test]
#[serial]
fn partial_bind_table_updates_preserve_untouched_buffer_requirements() {
    let mut ctx = ValidationContext::headless(&Default::default()).unwrap();

    let layout = ctx
        .make_bind_table_layout(&BindTableLayoutInfo {
            debug_name: "partial_bind_table_updates",
            shaders: &[ShaderInfo {
                shader_type: ShaderType::Compute,
                variables: &[BindTableVariable {
                    var_type: BindTableVariableType::Storage,
                    binding: 0,
                    count: 2,
                }],
            }],
        })
        .unwrap();
    let make_buffer = |ctx: &mut ValidationContext, name| {
        ctx.make_buffer(&BufferInfo {
            debug_name: name,
            byte_size: 64,
            visibility: MemoryVisibility::Gpu,
            usage: BufferUsage::STORAGE,
            ..Default::default()
        })
        .unwrap()
    };
    let first = make_buffer(&mut ctx, "partial_bind_table_updates_first");
    let replaced = make_buffer(&mut ctx, "partial_bind_table_updates_replaced");
    let replacement = make_buffer(&mut ctx, "partial_bind_table_updates_replacement");

    let table = ctx
        .make_bind_table(&BindTableInfo {
            debug_name: "partial_bind_table_updates",
            layout,
            bindings: &[IndexedBindingInfo {
                resources: &[
                    IndexedResource {
                        resource: ShaderResource::StorageBuffer(BufferView::new(first)),
                        slot: 0,
                    },
                    IndexedResource {
                        resource: ShaderResource::StorageBuffer(BufferView::new(replaced)),
                        slot: 1,
                    },
                ],
                binding: 0,
            }],
            ..Default::default()
        })
        .unwrap();

    ctx.update_bind_table(&BindTableUpdateInfo {
        table,
        bindings: &[IndexedBindingInfo {
            resources: &[IndexedResource {
                resource: ShaderResource::StorageBuffer(BufferView::new(replacement)),
                slot: 1,
            }],
            binding: 0,
        }],
    })
    .unwrap();

    let requirements = ctx.bind_table_buffer_requirements(table, ShaderStageMask::COMPUTE);
    assert!(requirements.iter().any(|item| item.buffer == first));
    assert!(requirements.iter().any(|item| item.buffer == replacement));
    assert!(!requirements.iter().any(|item| item.buffer == replaced));

    ctx.destroy_buffer(first);
    ctx.destroy_buffer(replaced);
    ctx.destroy_buffer(replacement);
}
