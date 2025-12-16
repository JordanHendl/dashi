use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use anyhow::{bail, Context as _, Result};

use crate::gpu::vulkan::{BindGroupLayoutInfo, BindTableLayout, BindTableLayoutInfo, Context};
use crate::utils::Handle;

use super::BindingManager;

#[cfg(feature = "dashi-serde")]
use crate::gpu::vulkan::structs::cfg;

#[cfg(feature = "dashi-serde")]
use serde::{Deserialize, Serialize};

/// Manages authored bind group/table layouts that can be referenced by name.
///
/// This acts as the bridge between authored configuration (e.g. YAML files)
/// and runtime layout handles required when building pipeline layouts. Layouts
/// are cached by [`BindingManager`], guaranteeing that repeated authoring data
/// resolves to the same GPU objects regardless of where it was loaded.
pub struct BindingLayoutManager {
    binding_manager: Arc<BindingManager>,
    bind_table_layouts: RwLock<HashMap<String, Handle<BindTableLayout>>>,
}

impl BindingLayoutManager {
    pub fn new(ctx: *mut Context, frames_in_flight: usize, thread_count: usize) -> Self {
        Self {
            binding_manager: BindingManager::new(ctx, frames_in_flight, thread_count),
            bind_table_layouts: RwLock::new(HashMap::new()),
        }
    }

    pub fn binding_manager(&self) -> Arc<BindingManager> {
        Arc::clone(&self.binding_manager)
    }

    pub fn register_bind_group_layout(
        &self,
        name: impl Into<String>,
        info: &BindGroupLayoutInfo<'_>,
    ) -> Result<Handle<BindTableLayout>> {
        let info = BindTableLayoutInfo {
            debug_name: info.debug_name,
            shaders: info.shaders,
        };
        self.register_bind_table_layout(name, &info)
    }

    pub fn register_bind_table_layout(
        &self,
        name: impl Into<String>,
        info: &BindTableLayoutInfo<'_>,
    ) -> Result<Handle<BindTableLayout>> {
        let name = name.into();
        let handle = self
            .binding_manager
            .try_get_or_create_btl_from_info(info, |ctx, info| ctx.make_bind_table_layout(info))
            .with_context(|| format!("creating bind table layout '{}'", name))?;

        let mut map = self.bind_table_layouts.write().unwrap();
        if let Some(existing) = map.get(&name) {
            if *existing != handle {
                bail!(
                    "bind table layout '{}' already registered with a different definition",
                    name
                );
            }
        } else {
            map.insert(name.clone(), handle);
        }
        Ok(handle)
    }

    pub fn bind_table_layout(&self, name: &str) -> Option<Handle<BindTableLayout>> {
        self.bind_table_layouts.read().unwrap().get(name).copied()
    }

    pub fn bind_group_layout(&self, name: &str) -> Option<Handle<BindTableLayout>> {
        self.bind_table_layout(name)
    }

    #[cfg(feature = "dashi-serde")]
    pub fn register_bind_group_layout_cfg(
        &self,
        name: impl Into<String>,
        cfg: &cfg::BindGroupLayoutCfg,
    ) -> Result<Handle<BindTableLayout>> {
        let borrowed = cfg.borrow();
        let info = borrowed.info();
        self.register_bind_group_layout(name, &info)
    }

    #[cfg(feature = "dashi-serde")]
    pub fn register_bind_table_layout_cfg(
        &self,
        name: impl Into<String>,
        cfg: &cfg::BindTableLayoutCfg,
    ) -> Result<Handle<BindTableLayout>> {
        let borrowed = cfg.borrow();
        let info = borrowed.info();
        self.register_bind_table_layout(name, &info)
    }

    #[cfg(feature = "dashi-serde")]
    pub fn load_from_yaml(&self, yaml: &str) -> Result<()> {
        let cfg = BindingLayoutsCfg::from_yaml(yaml).context("parsing binding layouts YAML")?;
        self.load_from_cfg(&cfg)
    }

    #[cfg(feature = "dashi-serde")]
    pub fn load_from_yaml_file(&self, path: &str) -> Result<()> {
        let contents = cfg::load_text(path)
            .with_context(|| format!("loading binding layouts YAML file '{}'", path))?;
        self.load_from_yaml(&contents)
    }

    #[cfg(feature = "dashi-serde")]
    pub fn load_from_cfg(&self, cfg: &BindingLayoutsCfg) -> Result<()> {
        for entry in &cfg.bind_group_layouts {
            self.register_bind_group_layout_cfg(&entry.name, &entry.layout)
                .with_context(|| format!("registering bind table layout '{}'", entry.name))?;
        }

        for entry in &cfg.bind_table_layouts {
            self.register_bind_table_layout_cfg(&entry.name, &entry.layout)
                .with_context(|| format!("registering bind table layout '{}'", entry.name))?;
        }

        Ok(())
    }
}

impl BindingLayoutManager {
    /// Resolve bind table layouts referenced by a pipeline layout configuration.
    pub fn resolve_bind_table_layouts(
        &self,
        refs: &[Option<String>; 4],
    ) -> Result<[Option<Handle<BindTableLayout>>; 4]> {
        let mut resolved = [None; 4];
        for (dst, name) in resolved.iter_mut().zip(refs.iter()) {
            if let Some(name) = name {
                let handle = self
                    .bind_table_layout(name)
                    .ok_or_else(|| anyhow::anyhow!("unknown bind table layout '{}'", name))?;
                *dst = Some(handle);
            }
        }
        Ok(resolved)
    }
}

#[cfg(feature = "dashi-serde")]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BindingLayoutsCfg {
    #[serde(default)]
    pub bind_group_layouts: Vec<NamedBindGroupLayoutCfg>,
    #[serde(default)]
    pub bind_table_layouts: Vec<NamedBindTableLayoutCfg>,
}

#[cfg(feature = "dashi-serde")]
impl BindingLayoutsCfg {
    pub fn from_yaml(s: &str) -> Result<Self, serde_yaml::Error> {
        serde_yaml::from_str(s)
    }
}

#[cfg(feature = "dashi-serde")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedBindGroupLayoutCfg {
    pub name: String,
    pub layout: cfg::BindGroupLayoutCfg,
}

#[cfg(feature = "dashi-serde")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedBindTableLayoutCfg {
    pub name: String,
    pub layout: cfg::BindTableLayoutCfg,
}
