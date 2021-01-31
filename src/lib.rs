use indexmap::{IndexMap, IndexSet};
use petgraph::{
    graph::{EdgeIndex, NodeIndex},
    stable_graph::StableGraph,
    visit::EdgeRef,
};
use rand::{
    distributions::{Distribution, WeightedIndex},
    RngCore,
};
use serde::{Deserialize, Serialize};

pub trait NodePattern {
    type N: std::fmt::Debug + Clone;
    fn is_match(&self, other: &Self::N) -> bool;
}
pub trait EdgePattern {
    type E: std::fmt::Debug + Clone + EdgeOverlap;
    fn is_match(&self, other: &Self::E) -> bool;
}

pub trait ReplacementNode: std::fmt::Debug {
    type N: std::fmt::Debug + Clone;
    fn create(&self) -> Self::N;
    fn apply(&self, target: &mut Self::N) -> bool;
}

pub trait EdgeOverlap {
    fn overlaps(&self, _other: &Self) -> bool {
        false
    }
}
pub trait ReplacementEdge: std::fmt::Debug {
    type E: std::fmt::Debug + Clone + EdgeOverlap;
    fn create(&self) -> Self::E;
    fn apply(&self, target: &mut Self::E) -> bool;
}

struct RewritingPatternMatch {
    nodes: IndexMap<NodeIndex, NodeIndex>,
    edges: IndexMap<EdgeIndex, EdgeIndex>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewritingPattern<NP, EP>(pub NodeIndex, pub StableGraph<NP, EP>);
#[derive(Clone)]
struct MatchProgress {
    consumed_world_nodes: IndexSet<NodeIndex>,
    consumed_world_edges: IndexSet<EdgeIndex>,
    consumed_pattern_nodes: IndexSet<NodeIndex>,
    consumed_pattern_edges: IndexSet<EdgeIndex>,
}
impl Default for MatchProgress {
    fn default() -> Self {
        MatchProgress {
            consumed_world_nodes: IndexSet::new(),
            consumed_world_edges: IndexSet::new(),
            consumed_pattern_nodes: IndexSet::new(),
            consumed_pattern_edges: IndexSet::new(),
        }
    }
}
impl<NP: NodePattern, EP: EdgePattern> RewritingPattern<NP, EP> {
    fn match_pattern(
        &self,
        node: NodeIndex,
        world: &StableGraph<NP::N, EP::E>,
    ) -> Option<RewritingPatternMatch> {
        if let Some(raw_match) = self.inner_match(self.0, node, world, MatchProgress::default()) {
            assert_eq!(
                raw_match.consumed_world_nodes.len(),
                raw_match.consumed_pattern_nodes.len()
            );
            assert_eq!(
                raw_match.consumed_world_edges.len(),
                raw_match.consumed_pattern_edges.len()
            );
            let nodes: IndexMap<_, _> = raw_match
                .consumed_pattern_nodes
                .into_iter()
                .zip(raw_match.consumed_world_nodes.into_iter())
                .collect();
            let edges: IndexMap<_, _> = raw_match
                .consumed_pattern_edges
                .into_iter()
                .zip(raw_match.consumed_world_edges.into_iter())
                .collect();
            Some(RewritingPatternMatch { nodes, edges })
        } else {
            None
        }
    }

    fn inner_match(
        &self,
        pattern_node: NodeIndex,
        world_node: NodeIndex,
        world: &StableGraph<NP::N, EP::E>,
        mut progress: MatchProgress,
    ) -> Option<MatchProgress> {
        let pattern_root = self.1.node_weight(pattern_node).unwrap();
        let world_root = world.node_weight(world_node).unwrap();
        if pattern_root.is_match(world_root) {
            let world_edges: Vec<_> = world
                .edges(world_node)
                .filter_map(|e| {
                    if progress.consumed_world_nodes.contains(&e.target())
                        || progress.consumed_world_edges.contains(&e.id())
                    {
                        None
                    } else {
                        Some(e)
                    }
                })
                .collect();
            let pattern_edges: Vec<_> = self
                .1
                .edges(pattern_node)
                .filter_map(|e| {
                    if progress.consumed_pattern_nodes.contains(&e.target())
                        || progress.consumed_pattern_edges.contains(&e.id())
                    {
                        None
                    } else {
                        Some(e)
                    }
                })
                .collect();
            if world_edges.len() < pattern_edges.len() {
                return None;
            }
            progress.consumed_pattern_nodes.insert(pattern_node);
            progress.consumed_world_nodes.insert(world_node);
            if pattern_edges.is_empty() {
                return Some(progress);
            }

            for pattern_edge in pattern_edges {
                progress.consumed_pattern_edges.insert(pattern_edge.id());
                let mut did_consume = false;
                for world_edge in &world_edges {
                    if pattern_edge.weight().is_match(world_edge.weight()) {
                        progress.consumed_world_edges.insert(world_edge.id());
                        let result = self.inner_match(
                            pattern_edge.target(),
                            world_edge.target(),
                            world,
                            progress.clone(),
                        );
                        if let Some(result) = result {
                            progress = result;
                            did_consume = true;
                            break;
                        } else {
                            progress.consumed_world_edges.remove(&world_edge.id());
                        }
                    }
                }
                if !did_consume {
                    return None;
                }
            }
            Some(progress)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RewritingReplacement<N, E> {
    pub output: StableGraph<N, E>,
    pub node_mapping: IndexMap<NodeIndex, NodeIndex>,
}

impl<N: ReplacementNode, E: ReplacementEdge> RewritingReplacement<N, E> {
    fn apply(
        &self,
        pattern_match: RewritingPatternMatch,
        world: &StableGraph<N::N, E::E>,
    ) -> Result<StableGraph<N::N, E::E>, ()> {
        let mut world: StableGraph<N::N, E::E> = world.clone();
        let surviving_nodes: IndexSet<_> = self.node_mapping.values().collect();
        for (pattern_node, world_node) in &pattern_match.nodes {
            if !surviving_nodes.contains(pattern_node) {
                world.remove_node(*world_node);
            }
        }

        let mut world_to_replacement = IndexMap::new();
        let mut nodes_created = IndexMap::new();
        for replacement_idx in self.output.node_indices() {
            let replacement = &self.output[replacement_idx];
            if let Some(pattern_idx) = self.node_mapping.get(&replacement_idx) {
                let world_idx = pattern_match.nodes.get(pattern_idx).unwrap();
                let world_node = world.node_weight_mut(*world_idx).unwrap();
                if !replacement.apply(world_node) {
                    return Err(());
                }
                world_to_replacement.insert(*world_idx, replacement_idx);
            } else {
                let new = world.add_node(replacement.create());
                nodes_created.insert(replacement_idx, new);
                world_to_replacement.insert(new, replacement_idx);
            }
        }

        for replacement_idx in self.output.node_indices() {
            let world_idx = if let Some(node_idx) = nodes_created.get(&replacement_idx) {
                node_idx
            } else {
                pattern_match
                    .nodes
                    .get(self.node_mapping.get(&replacement_idx).unwrap())
                    .unwrap()
            };
            let mut to_remove = IndexSet::new();
            let mut to_apply = IndexSet::new();
            for world_edge in world.edges(*world_idx) {
                if let Some(replacement_target_idx) = world_to_replacement.get(&world_edge.target())
                {
                    if let Some(replacement_edge) = self
                        .output
                        .find_edge(replacement_idx, *replacement_target_idx)
                    {
                        to_apply.insert((replacement_edge, world_edge.id()));
                    } else {
                        to_remove.insert(world_edge.id());
                    }
                }
            }
            for (replacement_idx, world_idx) in to_apply {
                let edge_weight = self.output.edge_weight(replacement_idx).unwrap();
                if !edge_weight.apply(world.edge_weight_mut(world_idx).unwrap()) {
                    return Err(());
                }
            }
            for idx in to_remove {
                world.remove_edge(idx);
            }
        }

        for edge in self.output.edge_indices() {
            let replacement_edge = self.output.edge_weight(edge).unwrap();
            let (replacement_source, replacement_target) =
                self.output.edge_endpoints(edge).unwrap();
            let source_world_idx = if let Some(node_idx) = nodes_created.get(&replacement_source) {
                node_idx
            } else {
                pattern_match
                    .nodes
                    .get(self.node_mapping.get(&replacement_source).unwrap())
                    .unwrap()
            };
            let target_world_idx = if let Some(node_idx) = nodes_created.get(&replacement_target) {
                node_idx
            } else {
                pattern_match
                    .nodes
                    .get(self.node_mapping.get(&replacement_target).unwrap())
                    .unwrap()
            };
            if let Some(world_edge) = world.find_edge(*source_world_idx, *target_world_idx) {
                let world_edge = world.edge_weight_mut(world_edge).unwrap();
                if !replacement_edge.apply(world_edge) {
                    return Err(());
                }
            } else {
                let e = replacement_edge.create();
                for existing in world.edges(*source_world_idx) {
                    if existing.weight().overlaps(&e) {
                        return Err(());
                    }
                }
                world.add_edge(*source_world_idx, *target_world_idx, e);
            }
        }
        Ok(world)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RewritingRule<
    NP: NodePattern,
    EP: EdgePattern,
    NR: ReplacementNode<N = NP::N>,
    ER: ReplacementEdge<E = EP::E>,
>(
    pub RewritingPattern<NP, EP>,
    pub RewritingReplacement<NR, ER>,
    pub f32,
);
pub struct RewritingRules<
    NP: NodePattern,
    EP: EdgePattern,
    NR: ReplacementNode<N = NP::N>,
    ER: ReplacementEdge<E = EP::E>,
>(pub Vec<RewritingRule<NP, EP, NR, ER>>);

fn weighted_shuffle(list: &[usize], weights: &[f32], rng: &mut impl RngCore) -> Vec<usize> {
    let mut result = Vec::with_capacity(list.len());
    let mut dist = WeightedIndex::new(weights).unwrap();
    let mut weights: Vec<_> = weights.iter().copied().collect();

    while result.len() < list.len() {
        let idx = dist.sample(rng);
        result.push(list[idx]);
        weights[idx] = 0.0;
        //dist.update_weights(&[(idx, &0.0)]);
        if let Ok(d) = WeightedIndex::new(&weights) {
            dist = d;
        } else {
            break;
        }
    }
    result
}

impl<
        NP: NodePattern,
        EP: EdgePattern,
        NR: ReplacementNode<N = NP::N>,
        ER: ReplacementEdge<E = EP::E>,
    > RewritingRules<NP, EP, NR, ER>
{
    fn new() -> Self {
        Self(vec![])
    }

    pub fn apply(
        &self,
        world: &StableGraph<NR::N, ER::E>,
    ) -> Result<StableGraph<NR::N, ER::E>, ()> {
        let mut rng = rand::thread_rng();
        let (rules, weights): (Vec<&RewritingRule<NP, EP, NR, ER>>, Vec<f32>) =
            self.0.iter().map(|r| (r, r.2)).unzip();
        let rule_idxs =
            weighted_shuffle(&(0..weights.len()).collect::<Vec<_>>(), &weights, &mut rng);

        let (nodes, weights): (Vec<_>, Vec<_>) = world
            .node_indices()
            .map(|n| (n, 100.0 / (world.edges(n).count() as f32 + 0.1)))
            .unzip();
        let node_idxs =
            weighted_shuffle(&(0..weights.len()).collect::<Vec<_>>(), &weights, &mut rng);

        for idx in rule_idxs {
            let rule = &rules[idx];
            for node_idx in &node_idxs {
                let node = &nodes[*node_idx];
                if let Some(rule_match) = rule.0.match_pattern(*node, world) {
                    if let Ok(new_world) = rule.1.apply(rule_match, world) {
                        return Ok(new_world);
                    }
                }
            }
        }
        Err(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    enum Node {
        A,
        B,
        C,
    }
    impl NodePattern for Node {
        type N = Self;
        fn is_match(&self, other: &Node) -> bool {
            self == other
        }
    }
    impl ReplacementNode for Node {
        type N = Self;
        fn create(&self) -> Node {
            *self
        }

        fn apply(&self, target: &mut Node) -> bool {
            *target = *self;
            true
        }
    }
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    struct Edge;
    impl EdgePattern for Edge {
        type E = Self;
        fn is_match(&self, other: &Edge) -> bool {
            self == other
        }
    }
    impl EdgeOverlap for Edge {}
    impl ReplacementEdge for Edge {
        type E = Self;
        fn create(&self) -> Edge {
            *self
        }

        fn apply(&self, target: &mut Edge) -> bool {
            *target = *self;
            true
        }
    }

    fn small_test_world() -> (NodeIndex, StableGraph<Node, Edge>) {
        let mut world = StableGraph::new();
        let root = world.add_node(Node::A);
        let other = world.add_node(Node::B);
        world.add_edge(root, other, Edge);
        world.add_edge(other, root, Edge);
        (root, world)
    }

    #[test]
    fn basic_match_functionality() {
        let (root, world) = small_test_world();

        let mut pattern = StableGraph::new();
        let pattern_root = pattern.add_node(Node::A);
        let other = pattern.add_node(Node::B);
        pattern.add_edge(pattern_root, other, Edge);

        let pattern = RewritingPattern(pattern_root, pattern);
        assert!(pattern.match_pattern(root, &world).is_some());
    }

    #[test]
    fn too_few_nodes() {
        let (root, world) = small_test_world();

        let mut pattern = StableGraph::new();
        let pattern_root = pattern.add_node(Node::A);
        let other = pattern.add_node(Node::B);
        let other2 = pattern.add_node(Node::C);
        pattern.add_edge(pattern_root, other, Edge);
        pattern.add_edge(pattern_root, other2, Edge);

        let pattern = RewritingPattern(pattern_root, pattern);
        assert!(pattern.match_pattern(root, &world).is_none());
    }

    #[test]
    fn mismatched_node() {
        let (root, world) = small_test_world();

        let mut pattern = StableGraph::new();
        let pattern_root = pattern.add_node(Node::A);
        let other = pattern.add_node(Node::C);
        pattern.add_edge(pattern_root, other, Edge);

        let pattern = RewritingPattern(pattern_root, pattern);
        assert!(pattern.match_pattern(root, &world).is_none());
    }

    #[test]
    fn simple_replacement() {
        let (root, mut world) = small_test_world();

        let mut pattern: StableGraph<Node, Edge> = StableGraph::new();
        let pattern_root = pattern.add_node(Node::A);
        let pattern = RewritingPattern(pattern_root, pattern);

        let mut replacement: StableGraph<Node, Edge> = StableGraph::new();
        let replacement_root = replacement.add_node(Node::B);
        let mut node_mapping = IndexMap::new();
        node_mapping.insert(replacement_root, pattern_root);
        let replacement = RewritingReplacement {
            output: replacement,
            node_mapping,
        };

        let pattern_match = pattern.match_pattern(root, &world).unwrap();
        replacement.apply(pattern_match, &mut world);
        match world[root] {
            Node::B => (),
            _ => panic!(),
        };
    }

    #[test]
    fn simple_replacement_with_reused_edge() {
        let (root, mut world) = small_test_world();

        let mut pattern: StableGraph<Node, Edge> = StableGraph::new();
        let pattern_root = pattern.add_node(Node::A);
        let pattern_other = pattern.add_node(Node::B);
        pattern.add_edge(pattern_root, pattern_other, Edge);
        pattern.add_edge(pattern_other, pattern_root, Edge);
        let pattern = RewritingPattern(pattern_root, pattern);

        let mut replacement: StableGraph<Node, Edge> = StableGraph::new();
        let replacement_root = replacement.add_node(Node::B);
        let other = replacement.add_node(Node::C);
        replacement.add_edge(replacement_root, other, Edge);
        replacement.add_edge(other, replacement_root, Edge);

        let mut node_mapping = IndexMap::new();
        node_mapping.insert(replacement_root, pattern_root);
        node_mapping.insert(other, pattern_other);
        let replacement = RewritingReplacement {
            output: replacement,
            node_mapping,
        };

        let pattern_match = pattern.match_pattern(root, &world).unwrap();
        replacement.apply(pattern_match, &mut world);
        let mut found_other = false;
        for other in world.neighbors(root) {
            match world[other] {
                Node::C => found_other = true,
                _ => (),
            };
        }
        assert!(found_other);
    }

    #[test]
    fn remove_node() {
        let (root, mut world) = small_test_world();

        let mut pattern: StableGraph<Node, Edge> = StableGraph::new();
        let pattern_root = pattern.add_node(Node::A);
        let pattern_other = pattern.add_node(Node::B);
        pattern.add_edge(pattern_root, pattern_other, Edge);
        pattern.add_edge(pattern_other, pattern_root, Edge);
        let pattern = RewritingPattern(pattern_root, pattern);

        let mut replacement: StableGraph<Node, Edge> = StableGraph::new();
        let replacement_root = replacement.add_node(Node::B);

        let mut node_mapping = IndexMap::new();
        node_mapping.insert(replacement_root, pattern_root);
        let replacement = RewritingReplacement {
            output: replacement,
            node_mapping,
        };

        let pattern_match = pattern.match_pattern(root, &world).unwrap();
        replacement.apply(pattern_match, &mut world);
        assert_eq!(world.node_count(), 1);
    }

    #[test]
    fn add_a_node() {
        let (root, mut world) = small_test_world();

        let mut pattern: StableGraph<Node, Edge> = StableGraph::new();
        let pattern_root = pattern.add_node(Node::A);
        let pattern_other = pattern.add_node(Node::B);
        pattern.add_edge(pattern_root, pattern_other, Edge);
        pattern.add_edge(pattern_other, pattern_root, Edge);
        let pattern = RewritingPattern(pattern_root, pattern);

        let mut replacement: StableGraph<Node, Edge> = StableGraph::new();
        let replacement_root = replacement.add_node(Node::A);
        let other = replacement.add_node(Node::B);
        let new = replacement.add_node(Node::C);
        replacement.add_edge(replacement_root, new, Edge);
        replacement.add_edge(new, replacement_root, Edge);
        replacement.add_edge(other, new, Edge);
        replacement.add_edge(new, other, Edge);

        let mut node_mapping = IndexMap::new();
        node_mapping.insert(replacement_root, pattern_root);
        node_mapping.insert(other, pattern_other);
        let replacement = RewritingReplacement {
            output: replacement,
            node_mapping,
        };

        let pattern_match = pattern.match_pattern(root, &world).unwrap();
        replacement.apply(pattern_match, &mut world);
        let mut f = File::create("/tmp/test.dot").unwrap();
        assert_eq!(world.node_count(), 3);
        let ns: IndexSet<_> = world.neighbors_undirected(root).collect();
        assert_eq!(ns.len(), 1);
    }

    #[test]
    fn add_edges() {
        let (root, mut world) = small_test_world();

        let mut pattern: StableGraph<Node, Edge> = StableGraph::new();
        let pattern_root = pattern.add_node(Node::A);
        let pattern_other = pattern.add_node(Node::B);
        pattern.add_edge(pattern_root, pattern_other, Edge);
        pattern.add_edge(pattern_other, pattern_root, Edge);
        let pattern = RewritingPattern(pattern_root, pattern);

        let mut replacement: StableGraph<Node, Edge> = StableGraph::new();
        let replacement_root = replacement.add_node(Node::A);
        let other1 = replacement.add_node(Node::B);
        let other2 = replacement.add_node(Node::C);
        replacement.add_edge(replacement_root, other1, Edge);
        replacement.add_edge(other1, replacement_root, Edge);
        replacement.add_edge(other1, other2, Edge);
        replacement.add_edge(other2, other1, Edge);

        let mut node_mapping = IndexMap::new();
        node_mapping.insert(replacement_root, pattern_root);
        node_mapping.insert(other2, pattern_other);
        let replacement = RewritingReplacement {
            output: replacement,
            node_mapping,
        };

        let pattern_match = pattern.match_pattern(root, &world).unwrap();
        replacement.apply(pattern_match, &mut world);
        assert_eq!(world.node_count(), 3);
        assert_eq!(world.neighbors(root).count(), 1);
    }
}
