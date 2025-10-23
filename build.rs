use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn compile_shader(source_path: &Path, kind: shaderc::ShaderKind) -> PathBuf {
    println!("cargo:rerun-if-changed={}", source_path.display());

    let source = fs::read_to_string(source_path).expect("failed to read shader source");
    let compiler = shaderc::Compiler::new().expect("failed to create shader compiler");
    let mut options = shaderc::CompileOptions::new().expect("failed to create shader options");
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);
    options.set_generate_debug_info();

    let artifact = compiler
        .compile_into_spirv(
            &source,
            kind,
            source_path
                .file_name()
                .and_then(|name| name.to_str())
                .expect("shader source must have valid file name"),
            "main",
            Some(&options),
        )
        .expect("failed to compile shader");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let mut output_path = out_dir.join(
        source_path
            .file_name()
            .expect("shader source must have file name"),
    );
    output_path.set_extension("spv");

    fs::write(&output_path, artifact.as_binary_u8()).expect("failed to write shader artifact");

    output_path
}

fn main() {
    let vert_path = Path::new("examples/shaders/hello_triangle.vert");
    let frag_path = Path::new("examples/shaders/hello_triangle.frag");

    let vert_spv = compile_shader(vert_path, shaderc::ShaderKind::Vertex);
    let frag_spv = compile_shader(frag_path, shaderc::ShaderKind::Fragment);

    let vert_spv_str = vert_spv.to_string_lossy().replace('\\', "/");
    let frag_spv_str = frag_spv.to_string_lossy().replace('\\', "/");

    println!(
        "cargo:rustc-env=HELLO_TRIANGLE_VERT_SPV={}",
        vert_spv_str
    );
    println!(
        "cargo:rustc-env=HELLO_TRIANGLE_FRAG_SPV={}",
        frag_spv_str
    );
}
