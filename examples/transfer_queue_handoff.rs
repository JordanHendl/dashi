use bytemuck::cast_slice;
use dashi::driver::command::{CopyBuffer, Dispatch};
use dashi::gpu::{self, QueueType};
use dashi::*;

/// Demonstrates handing a buffer from the transfer queue to compute and graphics queues.
/// The transfer list uploads data, after which a compute dispatch doubles the values and
/// copies them back for verification. The queue ownership transfer is handled automatically
/// by the resource state tracker.
fn main() -> Result<(), GPUError> {
    let mut ctx = gpu::Context::headless(&ContextInfo::default())?;

    let values: [u32; 4] = [1, 2, 3, 4];
    let byte_size = (values.len() * std::mem::size_of::<u32>()) as u32;

    let staging = ctx.make_buffer(&BufferInfo {
        debug_name: "transfer_staging",
        byte_size,
        visibility: MemoryVisibility::CpuAndGpu,
        usage: BufferUsage::STORAGE,
        initial_data: Some(cast_slice(&values)),
    })?;

    let device_buffer = ctx.make_buffer(&BufferInfo {
        debug_name: "device_storage",
        byte_size,
        visibility: MemoryVisibility::Gpu,
        usage: BufferUsage::STORAGE,
        initial_data: None,
    })?;

    let readback = ctx.make_buffer(&BufferInfo {
        debug_name: "compute_readback",
        byte_size,
        visibility: MemoryVisibility::CpuAndGpu,
        usage: BufferUsage::STORAGE,
        initial_data: None,
    })?;

    let ctx_ptr = &mut ctx as *mut _;

    // Upload data on the transfer queue.
    let mut transfer_list =
        ctx.pool_mut(QueueType::Transfer)
            .begin(ctx_ptr, "transfer_upload", false)?;
    let upload_stream = CommandStream::new_with_queue(QueueType::Transfer)
        .begin()
        .copy_buffers(&CopyBuffer {
            src: staging,
            dst: device_buffer,
            src_offset: 0,
            dst_offset: 0,
            amount: byte_size,
        });
    upload_stream.end().append(&mut transfer_list);
    let upload_fence = ctx.submit(&mut transfer_list, &Default::default())?;
    ctx.wait(upload_fence)?;

    // Build a simple compute pipeline that doubles each element in the buffer.
    let shader_info = ShaderInfo {
        shader_type: ShaderType::Compute,
        variables: &[BindTableVariable {
            var_type: BindTableVariableType::Storage,
            binding: 0,
            count: 1,
        }],
    };

    let table_layout = ctx.make_bind_table_layout(&BindTableLayoutInfo {
        debug_name: "transfer_compute_table_layout",
        shaders: &[shader_info],
    })?;

    let pipeline_layout = ctx.make_compute_pipeline_layout(&ComputePipelineLayoutInfo {
        bt_layouts: [Some(table_layout), None, None, None],
        shader: &PipelineShaderInfo {
            stage: ShaderType::Compute,
            spirv: inline_spirv::inline_spirv!(
                r#"
                #version 450
                layout(local_size_x = 1) in;
                layout(set = 0, binding = 0) buffer Data { uint values[]; } data;
                void main() {
                    uint idx = gl_GlobalInvocationID.x;
                    data.values[idx] = data.values[idx] * 2u;
                }
                "#,
                comp
            ),
            specialization: &[],
        },
    })?;

    let pipeline = ctx.make_compute_pipeline(&ComputePipelineInfo {
        debug_name: "transfer_compute_pipeline",
        layout: pipeline_layout,
    })?;

    let bind_table = ctx.make_bind_table(&BindTableInfo {
        debug_name: "transfer_compute_table",
        layout: table_layout,
        bindings: &[IndexedBindingInfo {
            binding: 0,
            resources: &[IndexedResource {
                slot: 0,
                resource: ShaderResource::StorageBuffer(BufferView::new(device_buffer)),
            }],
        }],
        set: 0,
    })?;

    // Dispatch on the compute queue and copy the results back for verification.
    let mut compute_list =
        ctx.pool_mut(QueueType::Compute)
            .begin(ctx_ptr, "compute_dispatch", false)?;
    let compute_stream = CommandStream::new_with_queue(QueueType::Compute).begin();
    let compute_stream = compute_stream
        .dispatch(&Dispatch {
            x: values.len() as u32,
            y: 1,
            z: 1,
            pipeline,
            bind_tables: [Some(bind_table), None, None, None],
            dynamic_buffers: Default::default(),
        })
        .copy_buffers(&CopyBuffer {
            src: device_buffer,
            dst: readback,
            src_offset: 0,
            dst_offset: 0,
            amount: byte_size,
        });
    compute_stream.end().append(&mut compute_list);

    let compute_fence = ctx.submit(&mut compute_list, &Default::default())?;
    ctx.wait(compute_fence)?;

    let results: &[u32] = ctx.map_buffer(BufferView::new(readback))?;
    println!("Uploaded values: {:?}", values);
    println!("Doubled values:  {:?}", results);
    ctx.unmap_buffer(readback)?;

    ctx.destroy();
    Ok(())
}
