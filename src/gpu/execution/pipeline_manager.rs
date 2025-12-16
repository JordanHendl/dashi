use std::collections::HashMap;
use std::sync::RwLock;

use anyhow::{Context as _, Result};

use crate::gpu::vulkan::{
    ComputePipeline, ComputePipelineInfo, ComputePipelineLayout, ComputePipelineLayoutInfo,
    Context, GraphicsPipeline, GraphicsPipelineInfo, GraphicsPipelineLayout,
    GraphicsPipelineLayoutInfo, RenderPass,
};
use crate::utils::Handle;

use super::BindingLayoutManager;

pub struct PipelineManager {
    ctx: *mut Context,
    binding_layouts: *const BindingLayoutManager,
    graphics_layouts: RwLock<HashMap<String, Handle<GraphicsPipelineLayout>>>,
    compute_layouts: RwLock<HashMap<String, Handle<ComputePipelineLayout>>>,
    graphics_pipelines: RwLock<HashMap<String, Handle<GraphicsPipeline>>>,
    compute_pipelines: RwLock<HashMap<String, Handle<ComputePipeline>>>,
}

impl PipelineManager {
    pub fn new(ctx: *mut Context, binding_layouts: &BindingLayoutManager) -> Self {
        Self {
            ctx,
            binding_layouts: binding_layouts as *const _,
            graphics_layouts: Default::default(),
            compute_layouts: Default::default(),
            graphics_pipelines: Default::default(),
            compute_pipelines: Default::default(),
        }
    }

    pub fn binding_layouts(&self) -> &BindingLayoutManager {
        unsafe { &*self.binding_layouts }
    }

    pub fn graphics_pipeline_layout(&self, name: &str) -> Option<Handle<GraphicsPipelineLayout>> {
        self.graphics_layouts.read().unwrap().get(name).copied()
    }

    pub fn compute_pipeline_layout(&self, name: &str) -> Option<Handle<ComputePipelineLayout>> {
        self.compute_layouts.read().unwrap().get(name).copied()
    }

    pub fn graphics_pipeline(&self, name: &str) -> Option<Handle<GraphicsPipeline>> {
        self.graphics_pipelines.read().unwrap().get(name).copied()
    }

    pub fn compute_pipeline(&self, name: &str) -> Option<Handle<ComputePipeline>> {
        self.compute_pipelines.read().unwrap().get(name).copied()
    }

    pub fn register_graphics_pipeline_layout(
        &self,
        name: impl Into<String>,
        info: &GraphicsPipelineLayoutInfo<'_>,
    ) -> Result<Handle<GraphicsPipelineLayout>> {
        let name = name.into();
        if let Some(handle) = self.graphics_layouts.read().unwrap().get(&name).copied() {
            return Ok(handle);
        }

        let handle = unsafe { &mut *self.ctx }
            .make_graphics_pipeline_layout(info)
            .with_context(|| format!("creating graphics pipeline layout '{}'", name))?;

        self.graphics_layouts
            .write()
            .unwrap()
            .insert(name.clone(), handle);
        Ok(handle)
    }

    pub fn register_compute_pipeline_layout(
        &self,
        name: impl Into<String>,
        info: &ComputePipelineLayoutInfo<'_>,
    ) -> Result<Handle<ComputePipelineLayout>> {
        let name = name.into();
        if let Some(handle) = self.compute_layouts.read().unwrap().get(&name).copied() {
            return Ok(handle);
        }

        let handle = unsafe { &mut *self.ctx }
            .make_compute_pipeline_layout(info)
            .with_context(|| format!("creating compute pipeline layout '{}'", name))?;

        self.compute_layouts
            .write()
            .unwrap()
            .insert(name.clone(), handle);
        Ok(handle)
    }

    pub fn register_graphics_pipeline(
        &self,
        name: impl Into<String>,
        info: &GraphicsPipelineInfo<'_>,
    ) -> Result<Handle<GraphicsPipeline>> {
        let name = name.into();
        if let Some(handle) = self.graphics_pipelines.read().unwrap().get(&name).copied() {
            return Ok(handle);
        }

        let handle = unsafe { &mut *self.ctx }
            .make_graphics_pipeline(info)
            .with_context(|| format!("creating graphics pipeline '{}'", name))?;

        self.graphics_pipelines
            .write()
            .unwrap()
            .insert(name.clone(), handle);
        Ok(handle)
    }

    pub fn register_compute_pipeline(
        &self,
        name: impl Into<String>,
        info: &ComputePipelineInfo<'_>,
    ) -> Result<Handle<ComputePipeline>> {
        let name = name.into();
        if let Some(handle) = self.compute_pipelines.read().unwrap().get(&name).copied() {
            return Ok(handle);
        }

        let handle = unsafe { &mut *self.ctx }
            .make_compute_pipeline(info)
            .with_context(|| format!("creating compute pipeline '{}'", name))?;

        self.compute_pipelines
            .write()
            .unwrap()
            .insert(name.clone(), handle);
        Ok(handle)
    }
}

#[cfg(feature = "dashi-serde")]
mod serde_support {
    use std::fs;

    use super::*;
    use anyhow::{anyhow, bail, Context as _};
    use std::collections::HashMap;

    use super::{PipelineManager, Result};
    use crate::gpu::vulkan::structs::cfg;
    use crate::gpu::vulkan::{
        GraphicsPipelineInfo, GraphicsPipelineLayout, GraphicsPipelineLayoutInfo,
        PipelineShaderInfo, RenderPass, SpecializationInfo, VertexDescriptionInfo,
    };
    use crate::utils::Handle;

    #[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
    struct PipelinesCfg {
        #[serde(default)]
        graphics_pipeline_layouts: Vec<NamedGraphicsPipelineLayoutCfg>,
        #[serde(default)]
        compute_pipeline_layouts: Vec<NamedComputePipelineLayoutCfg>,
        #[serde(default)]
        graphics_pipelines: Vec<NamedGraphicsPipelineCfg>,
        #[serde(default)]
        compute_pipelines: Vec<NamedComputePipelineCfg>,
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct NamedGraphicsPipelineLayoutCfg {
        name: String,
        layout: cfg::GraphicsPipelineLayoutCfg,
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct NamedComputePipelineLayoutCfg {
        name: String,
        layout: cfg::ComputePipelineLayoutCfg,
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct NamedGraphicsPipelineCfg {
        name: String,
        pipeline: cfg::GraphicsPipelineCfg,
    }

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct NamedComputePipelineCfg {
        name: String,
        pipeline: cfg::ComputePipelineCfg,
    }

    impl PipelinesCfg {
        fn from_yaml(yaml: &str) -> Result<Self, serde_yaml::Error> {
            serde_yaml::from_str(yaml)
        }
    }

    impl PipelineManager {
        pub fn load_from_yaml(
            &self,
            yaml: &str,
            render_passes: &HashMap<String, Handle<RenderPass>>,
        ) -> Result<()> {
            let cfg = PipelinesCfg::from_yaml(yaml).context("parsing pipelines YAML")?;
            self.load_from_cfg(&cfg, render_passes)
        }

        pub fn load_from_yaml_file(
            &self,
            path: &str,
            render_passes: &HashMap<String, Handle<RenderPass>>,
        ) -> Result<()> {
            let contents = fs::read_to_string(path)
                .with_context(|| format!("loading pipelines YAML file '{}'", path))?;
            self.load_from_yaml(&contents, render_passes)
        }

        pub fn load_from_cfg(
            &self,
            cfg: &PipelinesCfg,
            render_passes: &HashMap<String, Handle<RenderPass>>,
        ) -> Result<()> {
            for entry in &cfg.graphics_pipeline_layouts {
                self.register_graphics_pipeline_layout_cfg(&entry.name, &entry.layout)
                    .with_context(|| {
                        format!("registering graphics pipeline layout '{}'", entry.name)
                    })?;
            }

            for entry in &cfg.compute_pipeline_layouts {
                self.register_compute_pipeline_layout_cfg(&entry.name, &entry.layout)
                    .with_context(|| {
                        format!("registering compute pipeline layout '{}'", entry.name)
                    })?;
            }

            for entry in &cfg.graphics_pipelines {
                self.register_graphics_pipeline_cfg(&entry.name, &entry.pipeline, render_passes)
                    .with_context(|| format!("registering graphics pipeline '{}'", entry.name))?;
            }

            for entry in &cfg.compute_pipelines {
                self.register_compute_pipeline_cfg(&entry.name, &entry.pipeline)
                    .with_context(|| format!("registering compute pipeline '{}'", entry.name))?;
            }

            Ok(())
        }

        fn register_graphics_pipeline_layout_cfg(
            &self,
            name: impl Into<String>,
            cfg: &cfg::GraphicsPipelineLayoutCfg,
        ) -> Result<Handle<GraphicsPipelineLayout>> {
            let name = name.into();

            let mut bt_layout_refs = cfg.layouts.bt_layouts.clone();
            for (dst, legacy) in bt_layout_refs.iter_mut().zip(cfg.layouts.bg_layouts.iter()) {
                if dst.is_none() {
                    *dst = legacy.clone();
                }
            }

            let bt_layouts = self
                .binding_layouts()
                .resolve_bind_table_layouts(&bt_layout_refs)
                .with_context(|| {
                    format!(
                        "resolving bind table layouts for graphics pipeline layout '{}'",
                        name
                    )
                })?;

            let mut shader_words: Vec<Vec<u32>> = Vec::with_capacity(cfg.shaders.len());
            let mut specialization_data: Vec<Vec<Vec<u8>>> = Vec::with_capacity(cfg.shaders.len());

            // Pass 1: build all owned data first (no references yet)
            for shader in &cfg.shaders {
                shader_words.push(load_shader_words(shader)?);
                specialization_data.push(
                    shader
                        .specialization
                        .iter()
                        .map(|spec| spec.data.clone())
                        .collect(),
                );
            }

            // Pass 2: now create SpecializationInfo slices borrowing from specialization_data
            let mut specialization_infos: Vec<Vec<SpecializationInfo<'_>>> =
                Vec::with_capacity(cfg.shaders.len());
            for (shader, data_vec) in cfg.shaders.iter().zip(specialization_data.iter()) {
                let infos: Vec<SpecializationInfo<'_>> = data_vec
                    .iter()
                    .zip(shader.specialization.iter())
                    .map(|(data, spec)| SpecializationInfo {
                        slot: spec.slot,
                        data,
                    })
                    .collect();
                specialization_infos.push(infos);
            }

            let shader_infos: Vec<PipelineShaderInfo<'_>> = cfg
                .shaders
                .iter()
                .enumerate()
                .map(|(index, shader)| PipelineShaderInfo {
                    stage: shader.stage,
                    spirv: &shader_words[index],
                    specialization: &specialization_infos[index],
                })
                .collect();

            let vertex_info = VertexDescriptionInfo {
                entries: &cfg.vertex_info.entries,
                stride: cfg.vertex_info.stride,
                rate: cfg.vertex_info.rate,
            };

            let layout_info = GraphicsPipelineLayoutInfo {
                debug_name: &cfg.debug_name,
                vertex_info,
                bg_layouts: [None; 4],
                bt_layouts,
                shaders: &shader_infos,
                details: cfg.details.clone(),
            };

            self.register_graphics_pipeline_layout(name, &layout_info)
        }

        fn register_compute_pipeline_layout_cfg(
            &self,
            name: impl Into<String>,
            cfg: &cfg::ComputePipelineLayoutCfg,
        ) -> Result<Handle<ComputePipelineLayout>> {
            let name = name.into();

            let mut bt_layout_refs = cfg.layouts.bt_layouts.clone();
            for (dst, legacy) in bt_layout_refs.iter_mut().zip(cfg.layouts.bg_layouts.iter()) {
                if dst.is_none() {
                    *dst = legacy.clone();
                }
            }

            let bt_layouts = self
                .binding_layouts()
                .resolve_bind_table_layouts(&bt_layout_refs)
                .with_context(|| {
                    format!(
                        "resolving bind table layouts for compute pipeline layout '{}'",
                        name
                    )
                })?;

            let spirv = load_shader_words(&cfg.shader)?;
            let specialization_data: Vec<Vec<u8>> = cfg
                .shader
                .specialization
                .iter()
                .map(|spec| spec.data.clone())
                .collect();
            let specialization_infos: Vec<SpecializationInfo<'_>> = specialization_data
                .iter()
                .zip(cfg.shader.specialization.iter())
                .map(|(data, spec)| SpecializationInfo {
                    slot: spec.slot,
                    data,
                })
                .collect();

            let shader_info = PipelineShaderInfo {
                stage: cfg.shader.stage,
                spirv: &spirv,
                specialization: &specialization_infos,
            };

            let info = ComputePipelineLayoutInfo {
                bg_layouts: [None; 4],
                bt_layouts,
                shader: &shader_info,
            };

            self.register_compute_pipeline_layout(name, &info)
        }

        fn register_graphics_pipeline_cfg(
            &self,
            name: impl Into<String>,
            cfg: &cfg::GraphicsPipelineCfg,
            render_passes: &HashMap<String, Handle<RenderPass>>,
        ) -> Result<Handle<GraphicsPipeline>> {
            let name = name.into();
            let layouts_guard = self.graphics_layouts.read().unwrap();
            let layout = *layouts_guard
                .get(&cfg.layout)
                .ok_or_else(|| anyhow!("Unknown GraphicsPipelineLayout key: {}", cfg.layout))?;
            let render_pass = *render_passes
                .get(&cfg.render_pass)
                .ok_or_else(|| anyhow!("Unknown RenderPass key: {}", cfg.render_pass))?;
            let subpass_info = unsafe { &*self.ctx }
                .render_pass_subpass_info(render_pass, cfg.subpass_id)
                .ok_or_else(|| {
                    anyhow!(
                        "Unknown subpass {} for render pass {}",
                        cfg.subpass_id,
                        cfg.render_pass
                    )
                })?;
            let info = cfg
                .to_info(&*layouts_guard, subpass_info)
                .map_err(|err| anyhow!(err))?;
            drop(layouts_guard);

            self.register_graphics_pipeline(name, &info)
        }

        fn register_compute_pipeline_cfg(
            &self,
            name: impl Into<String>,
            cfg: &cfg::ComputePipelineCfg,
        ) -> Result<Handle<ComputePipeline>> {
            let name = name.into();
            let layout = {
                let layouts = self.compute_layouts.read().unwrap();
                *layouts
                    .get(&cfg.layout)
                    .ok_or_else(|| anyhow!("Unknown ComputePipelineLayout key: {}", cfg.layout))?
            };

            let info = ComputePipelineInfo {
                debug_name: &cfg.debug_name,
                layout,
            };

            self.register_compute_pipeline(name, &info)
        }
    }

    fn load_shader_words(cfg: &cfg::PipelineShaderCfg) -> Result<Vec<u32>> {
        if let Some(path) = &cfg.spirv_path {
            let bytes =
                fs::read(path).with_context(|| format!("reading SPIR-V file '{}'", path))?;
            if bytes.len() % 4 != 0 {
                bail!("SPIR-V file '{}' length must be a multiple of 4", path);
            }
            let mut words = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let mut word = [0u8; 4];
                word.copy_from_slice(chunk);
                words.push(u32::from_le_bytes(word));
            }
            return Ok(words);
        }

        if cfg.spirv.is_empty() {
            bail!("shader missing SPIR-V data or path");
        }

        Ok(cfg.spirv.clone())
    }
}
