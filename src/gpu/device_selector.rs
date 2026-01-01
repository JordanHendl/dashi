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

    fn check<T: PartialEq>(a: T, b: Option<T>) -> bool {
        if let Some(c) = b {
            return c == a;
        }

        false
    }

    /// Select the highest scoring device that satisfies the provided `filter`.
    ///
    /// Returns [`None`] when no device meets every required criterion.
    pub fn select(&self, filter: DeviceFilter) -> Option<SelectedDevice> {
        let mut max_score = 0;
        let mut best_device = -1;
        for (id, device) in self.devices.iter().enumerate() {
            let mut score = 0;
            if Self::check(&device.name, filter.name.as_ref()) {
                score += 1;
            }
            if Self::check(&device.kind, filter.kind.as_ref()) {
                score += 1;
            }
            if Self::check(&device.display_capable, filter.display_capable.as_ref()) {
                score += 1;
            }
            if Self::check(&device.bind_table_capable, filter.bind_table_capable.as_ref()) {
                score += 1;
            }

            if device.dashi_capable {
                score += 1;
            }

            if score > max_score {
                max_score = score;
                best_device = id as i32;
            }
        }

        if best_device >= 0 {
            return Some(SelectedDevice {
                device_id: best_device as usize,
                info: self.devices[best_device as usize].clone(),
            });
        }

        None
    }
}
