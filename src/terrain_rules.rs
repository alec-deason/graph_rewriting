use rand::seq::{IteratorRandom, SliceRandom};
use serde::{Deserialize, Serialize};
use std::{any::Any, cmp::Ordering, ops::Add};

use indexmap::IndexMap;
use petgraph::{
    algo::dijkstra,
    graph::{DiGraph, NodeIndex},
    stable_graph::StableGraph,
};

use super::{
    graph_rewriter::{
        EdgeOverlap, EdgePattern, NodePattern, ReplacementEdge, ReplacementNode, RewritingPattern,
        RewritingReplacement, RewritingRule, RewritingRules,
    },
    Direction, Distance, World, Slope,
};
use crate::{
    ecology::{Ecology, LocalBiome, ZonoBiome},
    Collectable, Exit, Location,
};

#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum PatternNode {
    Void,
    Mountain,
    Hill,
    Plain,
    Valley,
    Cavern,
    Lake,
}

impl PatternNode {
    fn height(&self) -> i32 {
        match self {
            PatternNode::Void => 0,
            PatternNode::Mountain => 100,
            PatternNode::Hill => 10,
            PatternNode::Valley => -5,
            PatternNode::Cavern => -50,
            PatternNode::Lake => 0,
            PatternNode::Plain => 0,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            PatternNode::Mountain => "mountain".into(),
            PatternNode::Hill => "hill".into(),
            PatternNode::Valley => "valley".into(),
            PatternNode::Lake => "lake".into(),
            PatternNode::Cavern => "cavern".into(),
            PatternNode::Plain => "plain".into(),
            _ => unreachable!(),
        }
    }
}

impl NodePattern for PatternNode {
    type N = Self;
    fn is_match(&self, other: &PatternNode) -> bool {
        self == other
    }
}

impl ReplacementNode for PatternNode {
    type N = Self;
    fn create(&self) -> PatternNode {
        *self
    }

    fn apply(&self, target: &mut PatternNode) -> bool {
        *target = *self;
        true
    }
}
#[derive(Default, Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
struct PatternEdge(Option<Direction>);
impl EdgePattern for PatternEdge {
    type E = RealizedEdge;
    fn is_match(&self, other: &Self::E) -> bool {
        if let Some(direction) = self.0 {
            direction == other.0
        } else {
            true
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct RealizedEdge(pub Direction);
impl EdgeOverlap for RealizedEdge {
    fn overlaps(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl ReplacementEdge for PatternEdge {
    type E = RealizedEdge;
    fn create(&self) -> Self::E {
        RealizedEdge(self.0.unwrap())
    }

    fn apply(&self, target: &mut Self::E) -> bool {
        if let Some(direction) = self.0 {
            target.0 = direction;
        }
        true
    }
}

#[typetag::serde]
impl Collectable for RewritingRule<PatternNode, PatternEdge, PatternNode, PatternEdge> {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn ex_nihilo() -> Vec<RewritingRule<PatternNode, PatternEdge, PatternNode, PatternEdge>> {
    let mut rules = vec![];
    for (thing, weight) in &[
        (PatternNode::Mountain, 1.0),
        //(PatternNode::Cavern, 0.01),
    ] {
        let mut pattern: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let pattern_root = pattern.add_node(PatternNode::Void);
        let mut replacement: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let replacement_root = replacement.add_node(*thing);
        let mut mapping = IndexMap::new();
        mapping.insert(replacement_root, pattern_root);

        rules.push(RewritingRule(
            RewritingPattern(pattern_root, pattern),
            RewritingReplacement {
                output: replacement,
                node_mapping: mapping,
            },
            *weight,
        ));
    }
    rules
}

fn mountain_range() -> Vec<RewritingRule<PatternNode, PatternEdge, PatternNode, PatternEdge>> {
    let mut rules = vec![];

    //Start a range
    for direction in &Direction::all_flat() {
        let mut pattern: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let pattern_root = pattern.add_node(PatternNode::Mountain);
        let mut replacement: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let replacement_root = replacement.add_node(PatternNode::Mountain);
        let replacement_other = replacement.add_node(PatternNode::Mountain);
        replacement.add_edge(
            replacement_root,
            replacement_other,
            PatternEdge(Some(*direction)),
        );
        replacement.add_edge(
            replacement_other,
            replacement_root,
            PatternEdge(Some(direction.opposite())),
        );

        let mut mapping = IndexMap::new();
        mapping.insert(replacement_root, pattern_root);

        rules.push(RewritingRule(
            RewritingPattern(pattern_root, pattern),
            RewritingReplacement {
                output: replacement,
                node_mapping: mapping,
            },
            0.1,
        ));
    }

    //Extend a range
    for direction in &Direction::all_flat() {
        let mut pattern: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let pattern_root = pattern.add_node(PatternNode::Mountain);
        let pattern_other = pattern.add_node(PatternNode::Mountain);
        pattern.add_edge(pattern_root, pattern_other, PatternEdge(Some(*direction)));
        pattern.add_edge(
            pattern_other,
            pattern_root,
            PatternEdge(Some(direction.opposite())),
        );
        let mut replacement: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let replacement_root = replacement.add_node(PatternNode::Mountain);
        let replacement_other = replacement.add_node(PatternNode::Mountain);
        let replacement_new = replacement.add_node(PatternNode::Mountain);
        replacement.add_edge(
            replacement_root,
            replacement_new,
            PatternEdge(Some(*direction)),
        );
        replacement.add_edge(
            replacement_new,
            replacement_root,
            PatternEdge(Some(direction.opposite())),
        );
        replacement.add_edge(
            replacement_other,
            replacement_new,
            PatternEdge(Some(*direction)),
        );
        replacement.add_edge(
            replacement_new,
            replacement_other,
            PatternEdge(Some(direction.opposite())),
        );

        let mut mapping = IndexMap::new();
        mapping.insert(replacement_root, pattern_root);
        mapping.insert(replacement_other, pattern_other);

        rules.push(RewritingRule(
            RewritingPattern(pattern_root, pattern),
            RewritingReplacement {
                output: replacement,
                node_mapping: mapping,
            },
            0.5,
        ));
    }

    rules
}

fn hills() -> Vec<RewritingRule<PatternNode, PatternEdge, PatternNode, PatternEdge>> {
    let mut rules = vec![];

    for direction in &Direction::all_flat() {
        let mut pattern: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let pattern_root = pattern.add_node(PatternNode::Mountain);
        let mut replacement: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let replacement_root = replacement.add_node(PatternNode::Mountain);
        let replacement_new = replacement.add_node(PatternNode::Hill);
        replacement.add_edge(
            replacement_root,
            replacement_new,
            PatternEdge(Some(*direction)),
        );
        replacement.add_edge(
            replacement_new,
            replacement_root,
            PatternEdge(Some(direction.opposite())),
        );

        let mut mapping = IndexMap::new();
        mapping.insert(replacement_root, pattern_root);

        rules.push(RewritingRule(
            RewritingPattern(pattern_root, pattern),
            RewritingReplacement {
                output: replacement,
                node_mapping: mapping,
            },
            0.2,
        ));
    }

    rules
}

fn valley() -> Vec<RewritingRule<PatternNode, PatternEdge, PatternNode, PatternEdge>> {
    let mut rules = vec![];

    //Intermountain valley
    for direction in &Direction::all_flat() {
        let mut pattern: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let pattern_root = pattern.add_node(PatternNode::Mountain);
        let pattern_other = pattern.add_node(PatternNode::Mountain);
        pattern.add_edge(pattern_root, pattern_other, PatternEdge(Some(*direction)));
        pattern.add_edge(
            pattern_other,
            pattern_root,
            PatternEdge(Some(direction.opposite())),
        );
        let mut replacement: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let replacement_root = replacement.add_node(PatternNode::Mountain);
        let replacement_other = replacement.add_node(PatternNode::Mountain);
        let replacement_new = replacement.add_node(PatternNode::Valley);
        replacement.add_edge(
            replacement_root,
            replacement_new,
            PatternEdge(Some(*direction)),
        );
        replacement.add_edge(
            replacement_new,
            replacement_root,
            PatternEdge(Some(direction.opposite())),
        );
        replacement.add_edge(
            replacement_other,
            replacement_new,
            PatternEdge(Some(*direction)),
        );
        replacement.add_edge(
            replacement_new,
            replacement_other,
            PatternEdge(Some(direction.opposite())),
        );

        let mut mapping = IndexMap::new();
        mapping.insert(replacement_root, pattern_root);
        mapping.insert(replacement_other, pattern_other);

        rules.push(RewritingRule(
            RewritingPattern(pattern_root, pattern),
            RewritingReplacement {
                output: replacement,
                node_mapping: mapping,
            },
            0.5,
        ));
    }

    rules
}

fn lake() -> Vec<RewritingRule<PatternNode, PatternEdge, PatternNode, PatternEdge>> {
    let mut rules = vec![];

    for direction in &Direction::all_flat() {
        let mut pattern: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let pattern_root = pattern.add_node(PatternNode::Mountain);
        let pattern_other = pattern.add_node(PatternNode::Hill);
        pattern.add_edge(pattern_root, pattern_other, PatternEdge(Some(*direction)));
        pattern.add_edge(
            pattern_other,
            pattern_root,
            PatternEdge(Some(direction.opposite())),
        );
        let mut replacement: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let replacement_root = replacement.add_node(PatternNode::Mountain);
        let replacement_other = replacement.add_node(PatternNode::Hill);
        let replacement_new = replacement.add_node(PatternNode::Lake);
        replacement.add_edge(
            replacement_root,
            replacement_new,
            PatternEdge(Some(*direction)),
        );
        replacement.add_edge(
            replacement_new,
            replacement_root,
            PatternEdge(Some(direction.opposite())),
        );
        replacement.add_edge(
            replacement_other,
            replacement_new,
            PatternEdge(Some(*direction)),
        );
        replacement.add_edge(
            replacement_new,
            replacement_other,
            PatternEdge(Some(direction.opposite())),
        );

        let mut mapping = IndexMap::new();
        mapping.insert(replacement_root, pattern_root);
        mapping.insert(replacement_other, pattern_other);

        rules.push(RewritingRule(
            RewritingPattern(pattern_root, pattern),
            RewritingReplacement {
                output: replacement,
                node_mapping: mapping,
            },
            0.5,
        ));
    }

    rules
}

fn cavern() -> Vec<RewritingRule<PatternNode, PatternEdge, PatternNode, PatternEdge>> {
    let mut rules = vec![];

    //Start a stub cavern
    let mut pattern: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
    let pattern_root = pattern.add_node(PatternNode::Mountain);
    let mut replacement: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
    let replacement_root = replacement.add_node(PatternNode::Mountain);
    let replacement_other = replacement.add_node(PatternNode::Cavern);
    replacement.add_edge(
        replacement_root,
        replacement_other,
        PatternEdge(Some(Direction::Below)),
    );
    replacement.add_edge(
        replacement_other,
        replacement_root,
        PatternEdge(Some(Direction::Above)),
    );

    let mut mapping = IndexMap::new();
    mapping.insert(replacement_root, pattern_root);

    rules.push(RewritingRule(
        RewritingPattern(pattern_root, pattern),
        RewritingReplacement {
            output: replacement,
            node_mapping: mapping,
        },
        0.1,
    ));

    //Start a loop cavern
    for direction in &Direction::all_flat() {
        let mut pattern: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let pattern_root = pattern.add_node(PatternNode::Mountain);
        let pattern_other = pattern.add_node(PatternNode::Mountain);
        pattern.add_edge(pattern_root, pattern_other, PatternEdge(Some(*direction)));
        pattern.add_edge(
            pattern_other,
            pattern_root,
            PatternEdge(Some(direction.opposite())),
        );

        let mut replacement: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let entrance = replacement.add_node(PatternNode::Mountain);
        let cavern = replacement.add_node(PatternNode::Cavern);
        let exit = replacement.add_node(PatternNode::Mountain);
        replacement.add_edge(entrance, cavern, PatternEdge(Some(Direction::Below)));
        replacement.add_edge(cavern, entrance, PatternEdge(Some(Direction::Above)));
        replacement.add_edge(entrance, exit, PatternEdge(Some(*direction)));
        replacement.add_edge(exit, entrance, PatternEdge(Some(direction.opposite())));
        replacement.add_edge(cavern, exit, PatternEdge(Some(Direction::Above)));
        replacement.add_edge(exit, cavern, PatternEdge(Some(Direction::Below)));

        let mut mapping = IndexMap::new();
        mapping.insert(entrance, pattern_root);
        mapping.insert(exit, pattern_other);

        rules.push(RewritingRule(
            RewritingPattern(pattern_root, pattern),
            RewritingReplacement {
                output: replacement,
                node_mapping: mapping,
            },
            0.1,
        ));
    }

    //Extend a cavern
    for direction in Direction::all_flat()
        .iter()
        .chain(std::iter::once(&Direction::Below))
    {
        let mut pattern: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let pattern_root = pattern.add_node(PatternNode::Cavern);
        let mut replacement: StableGraph<PatternNode, PatternEdge> = StableGraph::new();
        let replacement_root = replacement.add_node(PatternNode::Cavern);
        let replacement_new = replacement.add_node(PatternNode::Cavern);
        replacement.add_edge(
            replacement_root,
            replacement_new,
            PatternEdge(Some(*direction)),
        );
        replacement.add_edge(
            replacement_new,
            replacement_root,
            PatternEdge(Some(direction.opposite())),
        );

        let mut mapping = IndexMap::new();
        mapping.insert(replacement_root, pattern_root);

        rules.push(RewritingRule(
            RewritingPattern(pattern_root, pattern),
            RewritingReplacement {
                output: replacement,
                node_mapping: mapping,
            },
            0.5,
        ));
    }

    rules
}

fn over_world_rules() -> RewritingRules<PatternNode, PatternEdge, PatternNode, PatternEdge> {
    let mut rules = vec![];

    rules.extend(ex_nihilo());
    rules.extend(mountain_range());
    //rules.extend(cavern());
    rules.extend(valley());
    rules.extend(hills());
    rules.extend(lake());

    RewritingRules(rules)
}

fn detail_rules() -> RewritingRules<PatternNode, PatternEdge, PatternNode, PatternEdge> {
    let mut rules = vec![];


    RewritingRules(rules)
}

fn assign_plants(
    ctx: &World,
    world: &StableGraph<PatternNode, RealizedEdge>,
) -> IndexMap<NodeIndex, Ecology> {
    let mut result = IndexMap::new();
    let mut rng = rand::thread_rng();

    let zonobiome = ZonoBiome::Tropical; //*ZonoBiome::all().choose(&mut rng).unwrap();
    let environment = zonobiome.base_environment();
    let local_biome_weights = zonobiome.local_biome_weights();

    for node in world.node_indices() {
        let biome = LocalBiome::all().choose(&mut rng).unwrap().clone();
        let mut environment = environment.clone();
        biome.adjust_environment(&mut environment);
        let ecology = Ecology::new(environment, &ctx.species);
        result.insert(node, ecology);
    }

    result
}

fn make_terrain(_ctx: &World) -> (StableGraph<PatternNode, RealizedEdge>, StableGraph<PatternNode, RealizedEdge>) {
    let rules = over_world_rules();

    let mut over_world: StableGraph<PatternNode, RealizedEdge> = StableGraph::new();
    let _root = over_world.add_node(PatternNode::Void);

    for _i in 0..40 {
        if let Ok(new_over_world) = rules.apply(&mut over_world) {
            over_world = new_over_world;
        } else {
            break;
        }
    }

    let rules = detail_rules();
    let mut world = over_world.clone();
    for _i in 0..40 {
        if let Ok(new_world) = rules.apply(&mut world) {
            world = new_world;
        } else {
            break;
        }
    }

    (over_world, world)
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum Visibility {
    Dominant,
    Obvious,
    Visible,
    Obscured,
    Hidden,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct ObjectVisibility {
    pub direction: Direction,
    pub distance: Distance,
    pub visibility: Visibility,
    pub object: PatternNode,
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
struct Vector(f32, f32);
impl Add for Vector {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }
}
impl PartialOrd for Vector {
    fn partial_cmp(&self, other: &Vector) -> Option<Ordering> {
        let a = (self.0.powf(2.0) + self.1.powf(2.0)).sqrt();
        let b = (other.0.powf(2.0) + other.1.powf(2.0)).sqrt();
        a.partial_cmp(&b)
    }
}
impl Vector {
    fn direction(&self) -> Direction {
        unimplemented!()
    }

    fn from_direction_and_distance(direction: Direction, distance: Distance) -> Self {
        let a = direction.as_radians() + std::f32::consts::PI / 2.0;
        let d = distance.to_f32();
        /*
        match direction {
            Direction::Above | Direction::Below => d = std::f32::INFINITY,
            _ => (),
        }
        */
        Self(a.cos() * d, a.sin() * d)
    }
}
fn placement(world: &StableGraph<PatternNode, RealizedEdge>) -> IndexMap<NodeIndex, (i32, i32)> {
    let mut result = IndexMap::new();
    let root = world
        .node_indices()
        .choose(&mut rand::thread_rng())
        .unwrap();
    result.insert(root, (0, 0));

    for (node_idx, vector) in dijkstra(world, root, None, |e| {
        Vector::from_direction_and_distance(e.weight().0, Distance::AcrossTheStreet)
    }) {
        if vector.0.is_finite() && vector.1.is_finite() {
            result.insert(node_idx, (vector.0 as i32, vector.1 as i32));
        }
    }

    result
}

fn visibility(
    world: &StableGraph<PatternNode, RealizedEdge>,
) -> IndexMap<NodeIndex, Vec<(NodeIndex, ObjectVisibility)>> {
    let mut result = IndexMap::new();

    let mut placement = placement(world);
    let mut min_x = std::i32::MAX;
    let mut min_y = std::i32::MAX;
    let mut max_x = std::i32::MIN;
    let mut max_y = std::i32::MIN;
    for (x, y) in placement.values().copied() {
        if x < min_x {
            min_x = x;
        }
        if y < min_y {
            min_y = y;
        }
        if x > max_x {
            max_x = x;
        }
        if y > max_y {
            max_y = y;
        }
    }
    let x_scale = (max_x - min_x) as f32 / 100.0;
    let y_scale = (max_y - min_y) as f32 / 100.0;
    let mut grid = vec![vec![vec![]; 101]; 101];
    for (idx, (x, y)) in placement.iter_mut() {
        *x -= min_x;
        *x = (*x as f32 / x_scale) as i32;
        *y -= min_y;
        *y = (*y as f32 / y_scale) as i32;
        grid[*x as usize][*y as usize].push(*idx);
    }

    for observer_idx in world.node_indices() {
        if let PatternNode::Cavern = world.node_weight(observer_idx).unwrap() {
            continue;
        }
        if !placement.contains_key(&observer_idx) {
            continue;
        }
        let mut visibilities = vec![];
        for other_idx in world.node_indices() {
            if let PatternNode::Cavern = world.node_weight(other_idx).unwrap() {
                continue;
            }
            if !placement.contains_key(&other_idx) {
                continue;
            }
            let (ix, iy) = placement[&observer_idx];
            let iz = world.node_weight(observer_idx).unwrap().height();
            let mut x = ix as f32;
            let mut y = iy as f32;
            let mut z = iz as f32;
            let (tx, ty) = placement[&other_idx];
            let tz = world.node_weight(other_idx).unwrap().height();
            let dx = (tx - ix) as f32;
            let dy = (ty - iy) as f32;
            let dz = (tz - iz) as f32;
            let mut dist = 0.0;
            let mut hit = true;
            'outer: while dist < 1.0 {
                dist += 0.01;
                x = ix as f32 + dist * dx;
                y = iy as f32 + dist * dy;
                z = iz as f32 + dist * dz;
                if (x as i32, y as i32) == (ix, iy) {
                    continue;
                }
                for idx in &grid[x as usize][y as usize] {
                    if *idx == other_idx {
                        break;
                    }
                    let lz = world.node_weight(*idx).unwrap().height();
                    if lz as f32 > z {
                        hit = false;
                        break 'outer;
                    }
                }
            }
            if hit {
                let dist = dist
                    * ((dx * x_scale).powf(2.0) + (dy * y_scale).powf(2.0) + dz.powf(2.0)) * 2.0;
                let distance = Distance::from_f32(dist);
                visibilities.push((
                    other_idx,
                    ObjectVisibility {
                        direction: Direction::from_radians(
                            (dy).atan2(dx) - std::f32::consts::PI / 2.0,
                        ),
                        distance,
                        visibility: Visibility::Obvious,
                        object: world.node_weight(other_idx).unwrap().clone(),
                    },
                ));
            }
        }
        result.insert(observer_idx, visibilities);
    }

    result
}

pub fn make_level(ctx: &World) -> DiGraph<Location, Exit> {
    let (over_world, world) = make_terrain(ctx);
    terrain_to_locations(ctx, world)
}

pub fn terrain_to_locations(ctx: &World, world: StableGraph<PatternNode, RealizedEdge>) -> DiGraph<Location, Exit> {
    let plants = assign_plants(ctx, &world);
    let visibility = visibility(&world);
    world
        .map(
            |idx, n| {
                let ecology = plants[&idx].clone();
                let mut description: String = format!(
                    "a {} {} {}",
                    ecology.environment.zonobiome.name(),
                    ecology.environment.localbiome.name(),
                    n.name()
                );
                description.push_str(&format!("({})", idx.index()));
                let mut local_visibility = vec![];
                if let Some(objects) = visibility.get(&idx) {
                    if objects.len() > 1 {
                        description.push_str("\nFrom here you can see:\n");
                    }
                    for (other_idx, vis) in objects {
                        if *other_idx == idx {
                            continue;
                        }
                        local_visibility.push((*other_idx, vis.clone()));
                        let w = world.node_weight(*other_idx).unwrap();
                        let dist_and_dir = match vis.distance {
                            Distance::OverTheHorizon => {
                                format!("on the horizon to the {}", vis.direction.name())
                            }
                            Distance::DaysWalk => {
                                format!("very far to the {}", vis.direction.name())
                            }
                            Distance::LongWalk => format!("far to the {}", vis.direction.name()),
                            Distance::ShortWalk => {
                                format!("a ways off to the {}", vis.direction.name())
                            }
                            Distance::DownTheBlock => {
                                format!("a ways off to the {}", vis.direction.name())
                            }
                            Distance::AcrossTheStreet => format!("to the {}", vis.direction.name()),
                            Distance::AcrossTheRoom => format!("to the {}", vis.direction.name()),
                            Distance::AtHand => format!("to the {}", vis.direction.name()),
                            Distance::OnTopOf => format!("just to the {}", vis.direction.name()),
                        };
                        description.push_str(&format!(
                            "a {} {}({})\n",
                            w.name(),
                            dist_and_dir,
                            other_idx.index()
                        ));
                    }
                }
                let local_height = world.node_weight(idx).unwrap().height();
                for dir in &[Direction::North, Direction::West, Direction::South, Direction::East] {
                }
                let mut max_height_diff = [0i32; 4];
                let mut slope_dir = Direction::North;
                for other_idx in world.neighbors(idx) {
                    let dir = world.edge_weight(world.find_edge(idx, other_idx).unwrap()).unwrap().0;
                    let i = if dir >= Direction::NorthWest && dir < Direction::NorthEast {
                        0
                    } else if dir >= Direction::NorthEast && dir < Direction::SouthEast {
                        1
                    } else if dir >= Direction::SouthEast && dir < Direction::SouthWest {
                        2
                    } else {
                        3
                    };
                    let other_height = world.node_weight(other_idx).unwrap().height();
                    if (local_height - other_height).abs() > max_height_diff[i].abs() {
                        max_height_diff[i] = local_height - other_height;
                    }
                }

                let mut slope = [(Slope::Flat, false); 4];
                for (i, max_height_diff) in max_height_diff.into_iter().enumerate() {
                    let s = if max_height_diff.abs() < 5 {
                        Slope::Flat
                    } else if max_height_diff.abs() < 20 {
                        Slope::Gentle
                    } else if max_height_diff.abs() < 50 {
                        Slope::Steep
                    } else {
                        Slope::Cliff
                    };
                    slope[i] = (s, *max_height_diff > 0);
                }
                Location {
                    description,
                    ecology,
                    distant_objects: local_visibility,
                    terrain_type: *world.node_weight(idx).unwrap(),
                    slope,
                }
            },
            |idx, e| {
                let description = world
                    .node_weight(world.edge_endpoints(idx).unwrap().1)
                    .unwrap().name();
                Exit {
                    description: format!("path to {} to the {}", description, e.0.name()).into(),
                    direction: e.0,
                }
            },
        )
        .into()
}
