use std::{
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

fn main() {
    let shader_dir = Path::new("shaders");
    let spv_dir = Path::new("./spv");

    // Create the .spv directory if it doesn't exist
    if !spv_dir.exists() {
        fs::create_dir(spv_dir).expect("Failed to create '.spv' directory");
    }

    // Recursively traverse the shader directory
    traverse_and_compile(shader_dir, spv_dir);

    // Tell Cargo to rerun the build script if shaders change
    println!("cargo:rerun-if-changed=shaders");
}

/// Recursively traverses a directory and compiles GLSL files to SPIR-V
fn traverse_and_compile(shader_dir: &Path, spv_dir: &Path) {
    for entry in fs::read_dir(shader_dir).expect("Failed to read shader directory") {
        let entry = entry.expect("Failed to access directory entry");
        let path = entry.path();

        if path.is_dir() {
            // If it's a directory, recursively traverse it
            traverse_and_compile(&path, spv_dir);
        } else if let Some(extension) = path.extension() {
            if extension == "glsl" {
                // Compile the GLSL file to SPIR-V
                compile_glsl_to_spirv(&path, spv_dir);
            }
        }
    }
}

/// Compiles a GLSL file to SPIR-V using the `shaderc` crate
fn compile_glsl_to_spirv(glsl_path: &Path, spv_dir: &Path) {
    let filename = glsl_path.file_stem().expect("Invalid filename").to_str().unwrap();
    let output_path = spv_dir.join(format!("{}.spv", filename));

    println!("Compiling {:?} to {:?}", glsl_path, output_path);

    // Read the GLSL source code from the file
    let source = fs::read_to_string(glsl_path).expect("Failed to read GLSL file");

    // Create a shaderc compiler
    let mut compiler = shaderc::Compiler::new().expect("Failed to initialize shaderc compiler");
    let mut options = shaderc::CompileOptions::new().expect("Failed to initialize shaderc options");

    // Set the compilation options, like specifying it as a vertex shader, fragment shader, etc.
    let shader_kind = guess_shader_kind(glsl_path);

    let binary_result = compiler
        .compile_into_spirv(
            &source,
            shader_kind,
            glsl_path.to_str().unwrap(),
            "main",
            Some(&options),
        )
        .expect("Failed to compile GLSL to SPIR-V");

    // Write the compiled SPIR-V binary to the output file
    fs::write(&output_path, binary_result.as_binary_u8())
        .expect("Failed to write SPIR-V file");

    println!("Successfully compiled to {:?}", output_path);

    // Touch the output file to ensure it triggers rebuilds if it changes
    File::create(output_path)
        .expect("Failed to create SPIR-V file")
        .flush()
        .expect("Failed to flush file");
}

    /// Guesses the shader kind based on the file name content
fn guess_shader_kind(path: &Path) -> shaderc::ShaderKind {
    let filename = path.to_str().unwrap().to_lowercase();

    if filename.contains(".vert") {
        shaderc::ShaderKind::Vertex
    } else if filename.contains(".frag") {
        shaderc::ShaderKind::Fragment
    } else if filename.contains(".comp") {
        shaderc::ShaderKind::Compute
    } else if filename.contains(".geom") {
        shaderc::ShaderKind::Geometry
    } else if filename.contains(".tesc") {
        shaderc::ShaderKind::TessControl
    } else if filename.contains(".tese") {
        shaderc::ShaderKind::TessEvaluation
    } else {
        shaderc::ShaderKind::InferFromSource
    }
}

