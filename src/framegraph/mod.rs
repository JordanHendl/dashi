use std::collections::{VecDeque, HashMap};
use std::sync::Arc;

use ash::vk;

use crate::sync::{ResState};
use crate::sync::BarrierBuilder;
use crate::driver::types::Handle;
use crate::gpu::vulkan::{Image, Buffer};
use crate::sync::barrier_builder::ResourceLookup;

/// How a resource is used within a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseMode {
    /// Resource is read only
    Read,
    /// Resource is written to
    Write,
}

/// Image usage declaration for a pass.
#[derive(Debug, Clone, Copy)]
pub struct ImageUse {
    pub image: Handle<Image>,
    pub state: ResState,
    pub mode: UseMode,
}

/// Buffer usage declaration for a pass.
#[derive(Debug, Clone, Copy)]
pub struct BufferUse {
    pub buffer: Handle<Buffer>,
    pub state: ResState,
    pub mode: UseMode,
}

/// Type alias for a command recording callback.
pub type RecordFn = Arc<dyn Fn(vk::CommandBuffer) + Send + Sync>;

/// Description of a pass in the frame graph.
#[derive(Clone)]
pub struct PassDecl {
    pub label: &'static str,
    pub images: Vec<ImageUse>,
    pub buffers: Vec<BufferUse>,
    /// Function recording commands for this pass.
    pub record: RecordFn,
}

impl std::fmt::Debug for PassDecl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassDecl")
            .field("label", &self.label)
            .field("images", &self.images)
            .field("buffers", &self.buffers)
            .finish()
    }
}

impl Default for PassDecl {
    fn default() -> Self {
        Self {
            label: "",
            images: Vec::new(),
            buffers: Vec::new(),
            record: Arc::new(|_| {}),
        }
    }
}

/// Internal node representing a pass and its dependencies.
#[derive(Debug, Default, Clone)]
pub struct Node {
    pub pass: PassDecl,
    pub edges: Vec<usize>,
}

/// Frame graph consisting of a set of passes and their dependencies.
#[derive(Debug, Default)]
pub struct Graph {
    nodes: Vec<Node>,
}

impl Graph {
    /// Create an empty graph
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Add a pass to the graph and return its index.
    pub fn add_pass(&mut self, pass: PassDecl) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node { pass, edges: Vec::new() });
        id
    }

    /// Add a dependency edge from `src` to `dst`.
    pub fn add_edge(&mut self, src: usize, dst: usize) {
        if let Some(node) = self.nodes.get_mut(src) {
            node.edges.push(dst);
        }
    }

    /// Returns a topologically sorted order of the passes using Kahn's algorithm.
    pub fn topological_sort(&self) -> Vec<usize> {
        let mut indegree = vec![0u32; self.nodes.len()];
        for node in &self.nodes {
            for &e in &node.edges { indegree[e] += 1; }
        }
        let mut queue: VecDeque<usize> = indegree.iter().enumerate().filter_map(|(i,&d)| if d==0 {Some(i)} else {None}).collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(n) = queue.pop_front() {
            order.push(n);
            for &e in &self.nodes[n].edges {
                indegree[e]-=1;
                if indegree[e]==0 { queue.push_back(e); }
            }
        }
        order
    }

    /// Execute the frame graph.
    ///
    /// This allocates secondary command buffers for each pass, records them
    /// (allowing parallel recording by the caller if desired), stitches them
    /// into primaries that handle resource transitions, and finally schedules
    /// submissions on the provided queue using a timeline semaphore.
    pub fn execute<R: ResourceLookup>(
        &self,
        device: &ash::Device,
        lookup: &R,
        pool: vk::CommandPool,
        queue: vk::Queue,
        timeline: vk::Semaphore,
    ) -> Result<(), vk::Result> {
        use std::slice;

        let order = self.topological_sort();
        let mut img_states: HashMap<Handle<Image>, ResState> = HashMap::new();
        let mut buf_states: HashMap<Handle<Buffer>, ResState> = HashMap::new();

        let mut prev_value = 0u64;
        let mut value = 1u64;

        for idx in order {
            let node = &self.nodes[idx];

            // Record the pass into a secondary command buffer
            let alloc_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);
            let secondary = unsafe { device.allocate_command_buffers(&alloc_info)?[0] };

            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { device.begin_command_buffer(secondary, &begin_info)? };
            (node.pass.record)(secondary);
            unsafe { device.end_command_buffer(secondary)? };

            // Stitch with a primary that handles transitions
            let alloc_primary = vk::CommandBufferAllocateInfo::builder()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let primary = unsafe { device.allocate_command_buffers(&alloc_primary)?[0] };
            let begin_primary = vk::CommandBufferBeginInfo::builder();
            unsafe { device.begin_command_buffer(primary, &begin_primary)? };

            let mut builder = BarrierBuilder::new(lookup);
            for iu in &node.pass.images {
                let prev = img_states.get(&iu.image).cloned().unwrap_or_default();
                builder.image(iu.image, prev, iu.state);
                img_states.insert(iu.image, iu.state);
            }
            for bu in &node.pass.buffers {
                let prev = buf_states.get(&bu.buffer).cloned().unwrap_or_default();
                builder.buffer(bu.buffer, prev, bu.state);
                buf_states.insert(bu.buffer, bu.state);
            }
            unsafe { builder.emit(device, primary); }

            unsafe { device.cmd_execute_commands(primary, slice::from_ref(&secondary)); }
            unsafe { device.end_command_buffer(primary)? };

            // Submit and advance the timeline
            let wait_infos = if prev_value != 0 {
                vec![vk::SemaphoreSubmitInfo::builder()
                    .semaphore(timeline)
                    .value(prev_value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .build()]
            } else {
                Vec::new()
            };

            let signal_info = [vk::SemaphoreSubmitInfo::builder()
                .semaphore(timeline)
                .value(value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .build()];

            let cmd_info = [vk::CommandBufferSubmitInfo::builder()
                .command_buffer(primary)
                .build()];

            let submit = vk::SubmitInfo2::builder()
                .wait_semaphore_infos(&wait_infos)
                .signal_semaphore_infos(&signal_info)
                .command_buffer_infos(&cmd_info)
                .build();

            unsafe { device.queue_submit2(queue, slice::from_ref(&submit), vk::Fence::null())? };

            prev_value = value;
            value += 1;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topo_sort_orders_dependencies() {
        let mut g = Graph::new();
        let a = g.add_pass(PassDecl::default());
        let b = g.add_pass(PassDecl::default());
        let c = g.add_pass(PassDecl::default());
        g.add_edge(a, b);
        g.add_edge(b, c);
        let order = g.topological_sort();
        assert_eq!(order, vec![a, b, c]);
    }

    #[test]
    fn topo_sort_handles_branches() {
        let mut g = Graph::new();
        let a = g.add_pass(PassDecl::default());
        let b = g.add_pass(PassDecl::default());
        let c = g.add_pass(PassDecl::default());
        g.add_edge(a, c);
        g.add_edge(b, c);
        let order = g.topological_sort();
        assert_eq!(order.len(), 3);
        assert_eq!(order[2], c);
        assert!(order[0] == a || order[0] == b);
        assert!(order[1] == a || order[1] == b);
        assert_ne!(order[0], order[1]);
    }
}

