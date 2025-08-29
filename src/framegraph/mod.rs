use smallvec::SmallVec;
use rayon::prelude::*;

use crate::Handle;
use crate::sync::state::ResState;
use crate::gpu::vulkan::{Image, Buffer};

/// Describes usage of an image in a pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageUse {
    pub handle: Handle<Image>,
    pub state: ResState,
}

/// Describes usage of a buffer in a pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferUse {
    pub handle: Handle<Buffer>,
    pub state: ResState,
}

/// Declaration for a render/compute pass.
#[derive(Clone, Debug, Default)]
pub struct PassDecl {
    pub images: SmallVec<[ImageUse; 4]>,
    pub buffers: SmallVec<[BufferUse; 4]>,
}

impl PassDecl {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A node in the framegraph.
pub struct Node {
    pub decl: PassDecl,
    pub deps: SmallVec<[usize; 4]>,
    pub exec: Option<Box<dyn FnOnce() + Send + 'static>>, // placeholder for recording commands
}

impl Node {
    pub fn new<F>(decl: PassDecl, exec: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self { decl, deps: SmallVec::new(), exec: Some(Box::new(exec)) }
    }
}

/// Framegraph containing all nodes/passes.
pub struct Graph {
    pub nodes: Vec<Node>,
    order: Vec<usize>,
}

impl Graph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), order: Vec::new() }
    }

    /// Add a node to the graph and return its index.
    pub fn add_node(&mut self, node: Node) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        idx
    }

    /// Add a dependency `before -> after`.
    pub fn add_dependency(&mut self, after: usize, before: usize) {
        self.nodes[after].deps.push(before);
    }

    /// Topologically sort the graph using Kahn's algorithm.
    fn kahn_sort(&self) -> Vec<usize> {
        let node_count = self.nodes.len();
        let mut indegree = vec![0usize; node_count];
        let mut adj: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); node_count];
        for (idx, node) in self.nodes.iter().enumerate() {
            for &dep in &node.deps {
                indegree[idx] += 1;
                adj[dep].push(idx);
            }
        }
        let mut stack: SmallVec<[usize; 16]> = indegree
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d == 0 { Some(i) } else { None })
            .collect();
        let mut order = Vec::with_capacity(node_count);
        while let Some(n) = stack.pop() {
            order.push(n);
            for &m in &adj[n] {
                indegree[m] -= 1;
                if indegree[m] == 0 {
                    stack.push(m);
                }
            }
        }
        order
    }

    /// Execute nodes in topological order. Each node's closure will be
    /// executed in parallel as a stand-in for recording secondary command
    /// buffers. The execution order can be retrieved with `execution_order`.
    pub fn execute(&mut self) {
        self.order = self.kahn_sort();
        let mut tasks: Vec<Box<dyn FnOnce() + Send>> = Vec::new();
        for &idx in &self.order {
            if let Some(exec) = self.nodes[idx].exec.take() {
                tasks.push(exec);
            }
        }
        tasks.into_par_iter().for_each(|f| f());
    }

    /// Returns the execution order produced by the last `execute` call.
    pub fn execution_order(&self) -> &[usize] {
        &self.order
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_sort() {
        let mut g = Graph::new();
        let a = g.add_node(Node::new(PassDecl::new(), || {}));
        let b = g.add_node(Node::new(PassDecl::new(), || {}));
        g.add_dependency(b, a);
        g.execute();
        assert_eq!(g.execution_order(), &[0,1]);
    }
}

