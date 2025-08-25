use dashi::builders::{BindTableBuilder, BindTableLayoutBuilder};
use dashi::*;

/// Minimal example showing how to create a bindless `BindTable`.
/// Use `BindTable` when you need to index many resources from shaders or
/// frequently change resource bindings. For small, fixed sets of resources,
/// a classic `BindGroup` is usually simpler.
fn main() -> Result<(), GPUError> {
    // Create a headless context.
    let mut ctx = gpu::Context::headless(&ContextInfo::default())?;

    // Describe a single dynamic uniform buffer at binding 0.
    let shader_info = ShaderInfo {
        shader_type: ShaderType::All,
        variables: &[BindGroupVariable {
            var_type: BindGroupVariableType::DynamicUniform,
            binding: 0,
            count: 1,
        }],
    };

    // Build the layout and table.
    let layout = BindTableLayoutBuilder::new("bindless_layout")
        .shader(shader_info)
        .build(&mut ctx)?;

    let allocator = ctx.make_dynamic_allocator(&Default::default())?;
    let table = BindTableBuilder::new("bindless_table")
        .layout(layout)
        .binding(
            0,
            &[IndexedResource {
                slot: 0,
                resource: ShaderResource::Dynamic(&allocator),
            }],
        )
        .build(&mut ctx)?;

    println!("Created BindTable: {:?}", table);
    Ok(())
}
