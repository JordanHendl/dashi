#[derive(Default, Clone, Copy, PartialEq)]
pub enum DeviceType {
    Dedicated,
    Integrated,
    #[default]
    Other,
}

#[derive(Default, Clone)]
pub struct DeviceInfo {
    pub(crate) name: String,
    pub(crate) kind: DeviceType,
    pub(crate) driver_version: u32,
    pub(crate) gpu_score: u32,
    /// Indicates support for bindless `BindTable` descriptors.
    pub(crate) bind_table_capable: bool,
    pub(crate) display_capable: bool,
    pub(crate) dashi_capable: bool,
}

pub struct SelectedDevice {
    pub(crate) device_id: usize,
    pub info: DeviceInfo,
}

impl std::fmt::Display for SelectedDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[Name [{}] -- Driver Ver [{}] -- PCIE ID [{}]]",
            self.info.name, self.info.driver_version, self.device_id
        )
    }
}

#[derive(Default, Clone)]
pub struct DeviceFilter {
    name: Option<String>,
    kind: Option<DeviceType>,
    driver_version: Option<u32>,
    /// Require devices to support bindless `BindTable` descriptors.
    bind_table_capable: Option<bool>,
    display_capable: Option<bool>,
}

impl DeviceFilter {
    /// Require the selected device's name to exactly match `name`.
    ///
    /// When used with [`DeviceSelector::select`], the selection will return
    /// [`None`] if no available device satisfies all required criteria.
    pub fn add_required_name(&mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self.clone()
    }

    /// Require the selected device to be of the specified `kind`.
    ///
    /// Selection fails with [`None`] if no device matches every requirement.
    pub fn add_required_type(&mut self, kind: DeviceType) -> Self {
        self.kind = Some(kind);
        self.clone()
    }

    /// Require the device driver version to equal `version`.
    ///
    /// [`DeviceSelector::select`] returns [`None`] when no device meets all
    /// prerequisites.
    pub fn add_required_driver_version(&mut self, version: u32) -> Self {
        self.driver_version = Some(version);
        self.clone()
    }

    /// Require support for bindless `BindTable` descriptors on the selected device.
    ///
    /// [`DeviceSelector::select`] will return [`None`] if this capability is
    /// not available on any device.
    pub fn require_bind_table(&mut self) -> Self {
        self.bind_table_capable = Some(true);
        self.clone()
    }

    /// Require that the device supports presentation to a display (swapchain).
    ///
    /// If no device fulfills all filter requirements, selection yields
    /// [`None`].
    pub fn require_display(&mut self) -> Self {
        self.display_capable = Some(true);
        self.clone()
    }
}

pub struct DeviceSelector {
    pub(crate) devices: Vec<DeviceInfo>,
}

impl DeviceSelector {
    pub(crate) fn with_devices(devices: Vec<DeviceInfo>) -> Self {
        Self { devices }
    }

    /// Retrieve the list of devices discovered during construction.
    pub fn devices(&self) -> &[DeviceInfo] {
        &self.devices
    }

    /// Select a device by its index in the discovered device list.
    ///
    /// # Panics
    ///
    /// Panics if `id` is out of range for `self.devices`.
    pub fn select_by_id(&self, id: usize) -> SelectedDevice {
        SelectedDevice {
            device_id: id,
            info: self.devices[id].clone(),
        }
    }

    fn matches_filter(device: &DeviceInfo, filter: &DeviceFilter) -> bool {
        if let Some(ref name) = filter.name {
            if device.name != *name {
                return false;
            }
        }
        if let Some(kind) = filter.kind {
            if device.kind != kind {
                return false;
            }
        }
        if let Some(driver_version) = filter.driver_version {
            if device.driver_version != driver_version {
                return false;
            }
        }
        if let Some(bind_table_capable) = filter.bind_table_capable {
            if device.bind_table_capable != bind_table_capable {
                return false;
            }
        }
        if let Some(display_capable) = filter.display_capable {
            if device.display_capable != display_capable {
                return false;
            }
        }

        true
    }

    /// Select the highest scoring device that satisfies the provided `filter`.
    ///
    /// Returns [`None`] when no device meets every required criterion.
    pub fn select(&self, filter: DeviceFilter) -> Option<SelectedDevice> {
        let mut max_score = 0;
        let mut best_device = None;
        for (id, device) in self.devices.iter().enumerate() {
            if !Self::matches_filter(device, &filter) {
                continue;
            }

            let score = device.gpu_score + u32::from(device.dashi_capable);
            if best_device.is_none() || score > max_score {
                max_score = score;
                best_device = Some(id);
            }
        }

        best_device.map(|id| SelectedDevice {
            device_id: id,
            info: self.devices[id].clone(),
        })
    }
}
