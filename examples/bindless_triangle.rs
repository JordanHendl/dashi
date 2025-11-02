use dashi::builders::{BindTableBuilder, BindTableLayoutBuilder};
use dashi::*;
use glam::Mat4;

const NUM_TRANSFORMS: usize = 1024;
fn main() -> Result<(), GPUError> {
    // Create a headless context.
    let mut ctx = gpu::Context::new(&ContextInfo::default())?;

    // Describe a single dynamic uniform buffer at binding 0.
    let shader_info = ShaderInfo {
        shader_type: ShaderType::All,
        variables: &[BindGroupVariable {
            var_type: BindGroupVariableType::StorageImage,
            binding: 0,
            count: NUM_TRANSFORMS as u32,
        }],
    };

    // Build the layout and table.
    let layout = BindTableLayoutBuilder::new("bindless_layout")
        .shader(shader_info)
        .build(&mut ctx)?;

    // Allocate a dynamic buffer and bind it.
    let allocator = ctx.make_dynamic_allocator(&Default::default())?;
    let mut transforms = Vec::new();

    for _i in 0..NUM_TRANSFORMS {
        transforms.push(Mat4::default());
        // Should be random transforms
        todo!();
    }

    let mut idx = 0;
    let transforms_gpu: Vec<IndexedResource> = transforms
        .iter()
        .map(|a| {
            let b = ctx
                .make_buffer(&BufferInfo {
                    debug_name: "t",
                    byte_size: std::mem::size_of::<Mat4>() as u32,
                    initial_data: Some(&unsafe { [a].align_to::<u8>().1 }),
                    ..Default::default()
                })
                .expect("Unable to make transform buffer");
            idx += 1;
            IndexedResource {
                resource: ShaderResource::Buffer(b),
                slot: idx - 1,
            }
        })
        .collect();

    let table = BindTableBuilder::new("bindless_table")
        .layout(layout)
        .binding(0, &transforms_gpu)
        .build(&mut ctx)?;

    println!("Created bind table: {:?}", table);
        
    // Should render a triangle NUM_TRANSFORMS times and in the shader, ref the specified index.
    // Use dynamic allocator for setting the id the draw instance is supposed to use.
    todo!();
    Ok(())
}
