//what the tests ask of a tree: its arenas as the references and the pools see them, and its reachable part as a Tree
use super::*;
use crate::search::{Node as TreeNode, UNLINKED, link_up};
use std::collections::{HashMap, HashSet};

//what a tree holds: nodes, edges, lists' links (not link 0) and solid slots
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct Sizes {
    pub nodes: usize,
    pub edges: usize,
    pub links: usize,
    pub slots: usize,
}

//ranges of an arena that are in use or given back must not overlap
fn assert_apart(mut runs: Vec<(usize, usize)>, what: &str) {
    runs.sort_unstable();
    for pair in runs.windows(2) {
        assert!(pair[0].0 + pair[0].1 <= pair[1].0, "{what}: {:?} and {:?} overlap", pair[0], pair[1]);
    }
}

impl SharedTree {
    //every node the root reaches once, in breadth first order
    fn reachable_ids(&self) -> Vec<usize> {
        let mut order = vec![self.root];
        let mut seen: HashSet<usize> = order.iter().copied().collect();
        let mut index = 0;
        while index < order.len() {
            let node = self.nodes.get(order[index]);
            for (_, link) in self.links.linked(node) {
                let child = self.links.child(link) as usize;
                if seen.insert(child) {
                    order.push(child);
                }
            }
            index += 1;
        }
        order
    }

    //what the root reaches
    pub fn reachable(&self) -> Sizes {
        let mut sizes = Sizes { nodes: 0, edges: 0, links: 0, slots: 0 };
        for id in self.reachable_ids() {
            let node = self.nodes.get(id);
            sizes.nodes += 1;
            sizes.edges += node.edges().len();
            match node.children() {
                Children::List(_) => sizes.links += self.links.linked(node).count(),
                Children::Solid(_) => sizes.slots += node.edges().len(),
            }
        }
        sizes
    }

    //what is in use, dead nodes and unfreed lists included
    pub fn in_use(&self) -> Sizes {
        Sizes { nodes: self.nodes.used(), edges: self.edges.used(), links: self.links.used() - 1, slots: self.links.slots_used() }
    }

    //nodes, edges, links and slots: the bytes of the arenas and those of them in use
    pub fn arena_bytes(&self) -> [(usize, usize); 4] {
        [
            (self.nodes.len() * NODE_BYTES, self.nodes.used() * NODE_BYTES),
            (self.edges.len() * EDGE_BYTES, self.edges.taken() * EDGE_BYTES),
            (self.links.len() * LINK_BYTES, self.links.used() * LINK_BYTES),
            (self.links.slots() * SLOT_BYTES, self.links.slots_taken() * SLOT_BYTES),
        ]
    }

    pub fn table_entries(&self) -> Option<usize> {
        self.graph.as_ref().map(|graph| graph.entries.load(Ordering::Relaxed))
    }

    //the references of the nodes from the moves' (links and slots) and the root, for a tree made without the protocol
    pub fn count_references(&self) {
        let freed: HashSet<u32> = self.nodes.freed().into_iter().collect();
        let mut counts: HashMap<usize, u32> = HashMap::from([(self.root, 1)]);
        for id in (0..self.nodes.len()).filter(|&id| !freed.contains(&(id as u32))) {
            for (_, link) in self.links.linked(self.nodes.get(id)) {
                *counts.entry(self.links.child(link) as usize).or_default() += 1;
            }
        }
        for id in (0..self.nodes.len()).filter(|&id| !freed.contains(&(id as u32))) {
            self.nodes.get(id).set_references(counts.get(&id).copied().unwrap_or(0));
        }
    }

    //every reference is the number of moves that name the node (plus the root's), nothing given back is still in use, and no
    //two things share room; for a tree no thread is working in
    pub fn check_refs(&self) {
        let freed_nodes = self.nodes.freed();
        let freed: HashSet<u32> = freed_nodes.iter().copied().collect();
        assert_eq!(freed.len(), freed_nodes.len(), "a node given back twice");
        let mut counts: HashMap<usize, u32> = HashMap::from([(self.root, 1)]);
        let (mut links, mut edge_runs, mut slot_runs) = (HashSet::new(), Vec::new(), Vec::new());
        let alive: Vec<usize> = (0..self.nodes.len()).filter(|&id| !freed.contains(&(id as u32))).collect();
        assert!(!freed.contains(&(self.root as u32)), "the root was freed");
        for &id in &alive {
            let node = self.nodes.get(id);
            assert!(!node.is_freed(), "node {id} is not given back but marked as freed");
            let edges = node.edges();
            if !edges.is_empty() {
                edge_runs.push((edges.start, edges.len()));
            }
            //select_edge stops at the first move without a node
            let priors: Vec<f32> = edges.clone().map(|edge| self.edges.prior(edge)).collect();
            assert!(priors.is_sorted_by(|before, after| before >= after), "the priors of node {id}");
            let places: Vec<usize> = self.links.linked(node).map(|(place, _)| place).collect();
            assert!(places.iter().copied().eq(0..places.len()), "node {id}'s moves with a node are {places:?}");
            match node.children() {
                Children::List(mut link) => {
                    while link != NO_LINK {
                        assert!(links.insert(link), "link {link} in two lists");
                        link = self.links.get(link).next();
                    }
                }
                Children::Solid(first) => slot_runs.push(((first & !SOLID) as usize, edges.len())),
            }
            for (_, link) in self.links.linked(node) {
                let child = self.links.child(link) as usize;
                assert!(!freed.contains(&(child as u32)), "node {id} has the freed node {child} as a child");
                *counts.entry(child).or_default() += 1;
            }
        }
        let garbage = self.garbage.lock().expect("the garbage");
        for &head in self.retired.lock().expect("the retired lists").iter().chain(&garbage.lists) {
            let mut link = head;
            while link != NO_LINK {
                assert!(links.insert(link), "link {link} in two lists");
                link = self.links.get(link).next();
            }
        }
        for &id in &alive {
            let references = self.nodes.get(id).references();
            assert_eq!(references, counts.get(&id).copied().unwrap_or(0), "the references of node {id}");
        }
        for link in self.links.freed_links() {
            assert!(links.insert(link), "link {link} given back while in use or twice");
        }
        assert!(!links.contains(&NO_LINK), "link 0 is in use or given back");
        edge_runs.extend(self.edges.freed_runs());
        slot_runs.extend(self.links.freed_slot_runs());
        assert_apart(edge_runs, "edges");
        assert_apart(slot_runs, "slots");
        if let Some(graph) = &self.graph {
            let mut entries = 0;
            for shard in graph.shards.iter() {
                let shard = shard.lock().expect("a table shard");
                entries += shard.len();
                for (&entry, &id) in shard.iter() {
                    assert!(!freed.contains(&id), "an entry of {entry:x} names the freed node {id}");
                }
            }
            assert_eq!(entries, graph.entries.load(Ordering::Relaxed), "the table's entries");
            for &id in &alive {
                let node = self.nodes.get(id);
                let entry = graph_key(node.key(), u32::from(node.halfmoves()));
                let named = graph.shard(entry).lock().expect("a table shard").get(&entry).copied();
                match (node.references(), named) {
                    (0, Some(_)) => {}
                    (_, Some(named)) if named as usize == id => {}
                    (references, named) => panic!("node {id} with {references} references, its entry names {named:?}"),
                }
            }
        }
    }

    //the same tree in the same arenas: every node, edge, link and slot alike, and the table's entries; for trees no thread works in
    pub fn assert_same_arenas(&self, other: &SharedTree) {
        assert_eq!((self.root, &self.keys, self.is_full()), (other.root, &other.keys, other.is_full()), "the root, game or fullness");
        assert_eq!((self.arena_bytes(), self.in_use()), (other.arena_bytes(), other.in_use()), "the arenas");
        let children = |node: &Node| match node.children() {
            Children::List(head) => (false, head),
            Children::Solid(first) => (true, first),
        };
        let numbers = |node: &Node| {
            let proof = node.proof().map(|proof| (proof.outcome, proof.value.to_bits(), proof.plies));
            let sums = (node.value_sum().to_bits(), node.visited_policy_sqrt().to_bits());
            (node.counts(), sums, node.edges(), proof, node.key(), node.halfmoves(), node.references(), node.refreshes(), children(node))
        };
        for id in 0..self.nodes.len() {
            let (node, other_node) = (self.nodes.get(id), other.nodes.get(id));
            assert_eq!(numbers(node), numbers(other_node), "node {id}");
            for edge in node.edges() {
                assert_eq!(self.edges.chess_move(edge).bits(), other.edges.chess_move(edge).bits(), "edge {edge}'s move");
                assert_eq!(self.edges.prior(edge).to_bits(), other.edges.prior(edge).to_bits(), "edge {edge}'s prior");
            }
            match node.children() {
                Children::List(mut link) => {
                    while link != NO_LINK {
                        let (one, another) = (self.links.get(link), other.links.get(link));
                        let numbers = |link: &Link| (link.child(), link.visits().load(Ordering::Relaxed), link.place(), link.next());
                        assert_eq!(numbers(one), numbers(another), "link {link}");
                        link = one.next();
                    }
                }
                Children::Solid(first) => {
                    let slots = |tree: &SharedTree| -> Vec<(u32, u32)> {
                        let slots = tree.links.solid_slots(first, node.edges().len());
                        slots.iter().map(|slot| (slot.child(), slot.visits().load(Ordering::Relaxed))).collect()
                    };
                    assert_eq!(slots(self), slots(other), "node {id}'s slots");
                }
            }
        }
        match (&self.graph, &other.graph) {
            (None, None) => {}
            (Some(graph), Some(other_graph)) => {
                for (shard, other_shard) in graph.shards.iter().zip(other_graph.shards.iter()) {
                    assert!(*shard.lock().expect("a table shard") == *other_shard.lock().expect("a table shard"), "a table shard");
                }
                assert_eq!(graph.entries.load(Ordering::Relaxed), other_graph.entries.load(Ordering::Relaxed), "the table's entries");
            }
            _ => panic!("one tree has transpositions and the other not"),
        }
    }

    //the part the root reaches as a tree numbered as Tree::compact numbers it (breadth first), whatever the arenas' order
    pub fn to_reachable_tree(&self) -> Tree {
        let order = self.reachable_ids();
        let numbers: HashMap<usize, u32> = order.iter().enumerate().map(|(number, &id)| (id, number as u32)).collect();
        let (mut moves, mut priors) = (Vec::new(), Vec::new());
        let (mut links, mut slots) = (vec![UNLINKED], Vec::new());
        let (mut nodes, mut node_keys, mut halfmoves) = (Vec::new(), Vec::new(), Vec::new());
        for (number, &id) in order.iter().enumerate() {
            let node = self.nodes.get(id);
            let (visit_count, in_flight) = node.counts();
            let edges = node.edges();
            let mut copy = TreeNode {
                visit_count,
                in_flight,
                value_sum: node.value_sum(),
                visited_policy_sqrt: node.visited_policy_sqrt(),
                edges_at: if edges.is_empty() { 0 } else { (moves.len() as u64) << 8 | edges.len() as u64 },
                proof: node.proof().filter(|_| number > 0),
                children: NO_LINK,
            };
            for edge in edges {
                moves.push(self.edges.chess_move(edge));
                priors.push(self.edges.prior(edge));
            }
            let children = self
                .links
                .linked(node)
                .map(|(place, link)| (place, numbers[&(self.links.child(link) as usize)], self.links.visits(link).load(Ordering::Relaxed)));
            link_up(&mut copy, &mut links, &mut slots, visit_count >= SOLID_TREE_THRESHOLD, children);
            nodes.push(copy);
            node_keys.push(node.key());
            halfmoves.push(node.halfmoves());
        }
        let graph = self.graph.as_ref().map(|_| {
            let table = node_keys.iter().zip(&halfmoves).enumerate().map(|(number, (&key, &count))| (graph_key(key, u32::from(count)), number as u32));
            Graph { table: table.collect(), halfmoves }
        });
        Tree {
            board: self.board.clone(),
            keys: self.keys.clone(),
            nodes,
            root: 0,
            node_keys,
            moves,
            priors,
            noised_priors: Vec::new(),
            links,
            slots,
            memory_limit: usize::MAX,
            pending: Vec::new(),
            logits: Vec::new(),
            full: self.is_full(),
            graph,
        }
    }
}
