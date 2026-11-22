//what the tests ask of a tree: its arenas as the references and the pools see them, and its reachable part as a Tree. With the
//cargo feature `audit` the checks that need no test's network are Searches.audit's, for the tree a real search left
#![cfg_attr(not(test), allow(dead_code))]
use super::arena::units;
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
            sizes.edges += units(node.edges().len());
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

    //the counts that are no move's: the root's, and a held root's
    fn own_counts(&self) -> HashMap<usize, u32> {
        let held = self.garbage.lock().expect("the garbage").held;
        std::iter::once(self.root).chain(held.map(|held| held as usize)).map(|id| (id, 1)).collect()
    }

    //the references of the nodes from the moves' (links and slots) and the root, for a tree made without the protocol
    pub fn count_references(&self) {
        let freed: HashSet<u32> = self.nodes.freed().into_iter().collect();
        let mut counts = self.own_counts();
        for id in (0..self.nodes.len()).filter(|&id| !freed.contains(&(id as u32))) {
            for (_, link) in self.links.linked(self.nodes.get(id)) {
                *counts.entry(self.links.child(link) as usize).or_default() += 1;
            }
        }
        for id in (0..self.nodes.len()).filter(|&id| !freed.contains(&(id as u32))) {
            self.nodes.get(id).set_references(counts.get(&id).copied().unwrap_or(0));
        }
    }

    //every count of descents in flight is the waiting leaves that count them: a node's of those that wait at it
    //(in_flight_nodes), a move's of those that wait through it, and none where no leaf waits; for a tree no thread is working
    //in. It would catch a count that a gathering call left behind or took off twice: a run's descents it did not get to, a
    //collision's, a leaf's that another call took
    pub fn assert_waiting_counts(&self) {
        let (mut at_nodes, mut at_moves): (HashMap<usize, u32>, HashMap<u32, u32>) = Default::default();
        for leaf in self.pending.lock().expect("the pending leaves").values().flatten() {
            for &node in self.in_flight_nodes(&leaf.path, true) {
                *at_nodes.entry(node).or_default() += 1;
            }
            for (&node, &link) in leaf.path.iter().zip(&leaf.edges) {
                *at_moves.entry(self.current_link(node, link as u32)).or_default() += 1;
            }
        }
        let freed: HashSet<u32> = self.nodes.freed().into_iter().collect();
        let (mut nodes, mut moves) = (0, 0);
        for id in (0..self.nodes.len()).filter(|&id| !freed.contains(&(id as u32))) {
            let node = self.nodes.get(id);
            let waiting = at_nodes.get(&id).copied().unwrap_or(0);
            assert_eq!(node.counts().1, waiting, "node {id}: the descents in flight at it against the leaves that wait at it");
            nodes += usize::from(waiting > 0);
            for (place, link) in self.links.linked(node) {
                let waiting = at_moves.get(&link).copied().unwrap_or(0);
                assert_eq!(self.links.slot(link).state().pending(), waiting, "node {id}, move {place}: the descents in flight through it");
                moves += usize::from(waiting > 0);
            }
        }
        assert_eq!((nodes, moves), (at_nodes.len(), at_moves.len()), "a waiting leaf's node or move is no longer the tree's");
    }

    //the audits of a tree that a search has left and no thread works in, whatever network evaluated it: the references and the
    //arenas (check_refs), the counts of what waits (assert_waiting_counts) and what the moves hold against their nodes
    //(check_moves). Each panics with what it found. Returns the nodes and the moves that check_moves looked at
    pub fn audit(&self) -> (usize, usize) {
        self.check_refs();
        self.assert_waiting_counts();
        self.check_moves()
    }

    //with transpositions, for every node the root reaches: a move has no more visits than its node; a visited move's mean is a
    //score; a move that holds its node as lost leads to a node proven lost and holds its score; a move's filter is its node's;
    //a node's visits are its own and its moves'; a proven node's sum is its visits at its proven score; an unproven node with
    //moves has a visit of its own at least, and its own visits' scores (its sum less its moves' visits at the means they hold)
    //are scores. The network's values are not known here, so no more can be said of a node's own scores: the tests, which
    //know their network's, say it (assert_move_means). The root's sum is left out: a root proven before the game came to it
    //keeps the sum its proof wrote
    fn check_moves(&self) -> (usize, usize) {
        let ids = self.reachable_ids();
        if self.graph.is_none() {
            return (ids.len(), 0);
        }
        let mut moves = 0;
        for &id in &ids {
            let node = self.nodes.get(id);
            let visits = node.visits();
            let (mut moves_visits, mut moves_sum) = (0u64, 0.0);
            for (place, link) in self.links.linked(node) {
                let record = self.links.record(link);
                let child = self.nodes.get(record.state.child() as usize);
                let what = format!("node {id}, move {place}");
                assert!(record.visits <= child.visits(), "{what}: {} visits, its node {}", record.visits, child.visits());
                assert!(record.visits == 0 || (0.0..=1.0).contains(&record.mean), "{what}: a mean of {}", record.mean);
                let lost = child.proof().is_some_and(|proof| proof.outcome == Outcome::Loss && record.mean == proof.value as f32);
                assert!(!record.state.lost() || lost, "{what}: held as lost, its mean {} and its node's proof {:?}", record.mean, child.proof().map(|proof| proof.outcome));
                assert_eq!(record.state.repeats(), repeat_filter(child.key(), child.halfmoves()), "{what}: its filter of its node's repeats");
                moves_visits += u64::from(record.visits);
                moves_sum += f64::from(record.visits) * (1.0 - f64::from(record.mean));
                moves += 1;
            }
            assert!(u64::from(visits) >= moves_visits, "node {id}: {visits} visits, its moves {moves_visits}");
            let (own_visits, tolerance) = (f64::from(visits) - moves_visits as f64, 1e-6 * f64::from(visits.max(1)));
            match node.proof() {
                _ if id == self.root => {}
                Some(proof) => {
                    let sum = f64::from(visits) * proof.value;
                    assert!((node.value_sum() - sum).abs() <= tolerance, "node {id}: a proven node's sum {} for {visits} visits", node.value_sum());
                }
                None if node.has_edges() => {
                    let own_sum = node.value_sum() - moves_sum;
                    let scores = own_visits >= 1.0 && (-tolerance..=own_visits + tolerance).contains(&own_sum);
                    assert!(scores, "node {id}: sum {} less its moves' {moves_sum} over {own_visits} visits of its own", node.value_sum());
                }
                None => {}
            }
        }
        (ids.len(), moves)
    }

    //every reference is the number of moves that name the node (plus the root's and a held root's), nothing given back is still
    //in use, and no two things share room; for a tree no thread is working in
    pub fn check_refs(&self) {
        self.assert_waiting_counts();
        let freed_nodes = self.nodes.freed();
        let freed: HashSet<u32> = freed_nodes.iter().copied().collect();
        assert_eq!(freed.len(), freed_nodes.len(), "a node given back twice");
        let mut counts = self.own_counts();
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
            //select_edge stops at the head's first move without a node and the tail's
            let priors: Vec<f32> = edges.clone().map(|edge| self.edges.prior(edges.clone(), edge)).collect();
            let (head, tail) = priors.split_at(priors.len().min(HEAD_MOVES));
            assert!(
                head.is_sorted_by(|before, after| before >= after) && tail.is_sorted_by(|before, after| before >= after),
                "the priors of node {id}"
            );
            //the moves that link_known_moves gave nodes are any of the node's
            let places: Vec<usize> = self.links.linked(node).map(|(place, _)| place).collect();
            assert!(node.known_passes() > 0 || first_of_head_and_tail(&places), "node {id}'s moves with a node are {places:?}");
            match node.children() {
                Children::List(mut link) => {
                    while link != NO_LINK {
                        assert!(links.insert(link), "link {link} in two lists");
                        link = self.links.get(link).next();
                    }
                    //one link a move, in the moves' order, wherever link_known_moves put some in
                    assert!(places.is_sorted_by(|before, after| before < after), "node {id}'s list has its links at {places:?}");
                    assert!(places.last().is_none_or(|&last| last < edges.len()), "node {id}'s list has a link past its moves");
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
        assert_eq!(self.graph.is_some(), other.graph.is_some(), "one tree with transpositions and the other without");
        assert_eq!((self.arena_bytes(), self.in_use()), (other.arena_bytes(), other.in_use()), "the arenas");
        let children = |node: &Node| match node.children() {
            Children::List(head) => (false, head),
            Children::Solid(first) => (true, first),
        };
        //all that a move holds, its mean by its bits
        let count = |tree: &SharedTree, link: u32| {
            let record = tree.links.record(link);
            (record.visits, record.mean.to_bits(), record.state)
        };
        let numbers = |node: &Node| {
            let proof = node.proof().map(|proof| (proof.outcome, proof.value.to_bits(), proof.plies));
            let sums = (node.value_sum().to_bits(), node.visited_policy_sqrt().to_bits());
            let (known, names) = ((node.known_passes(), node.known_to()), (node.references(), children(node)));
            (node.counts(), sums, node.edges(), proof, known, node.key(), node.halfmoves(), node.refreshes(), names)
        };
        for id in 0..self.nodes.len() {
            let (node, other_node) = (self.nodes.get(id), other.nodes.get(id));
            assert_eq!(numbers(node), numbers(other_node), "node {id}");
            assert_eq!(self.edges.run(node.edges()), other.edges.run(node.edges()), "node {id}'s edges");
            match node.children() {
                Children::List(mut link) => {
                    while link != NO_LINK {
                        let (one, another) = (self.links.get(link), other.links.get(link));
                        let numbers = |link: &Link| (link.place(), link.next());
                        assert_eq!((numbers(one), count(self, link)), (numbers(another), count(other, link)), "link {link}");
                        link = one.next();
                    }
                }
                Children::Solid(first) => {
                    let slots = |tree: &SharedTree| -> Vec<_> { (first..).take(node.edges().len()).map(|link| count(tree, link)).collect() };
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
                known_passes: node.known_passes() as u8,
                known_to: node.known_to() as u8,
            };
            for edge in edges.clone() {
                moves.push(self.edges.chess_move(edge));
                priors.push(self.edges.prior(edges.clone(), edge));
            }
            let children = self.links.linked(node).map(|(place, link)| {
                let slot = slot_of(self.links.record(link));
                (place, crate::search::Slot { child: numbers[&(slot.child as usize)], ..slot })
            });
            let solid = copy.solid_in_a_copy();
            link_up(&mut copy, &mut links, &mut slots, solid, children);
            nodes.push(copy);
            node_keys.push(node.key());
            halfmoves.push(node.halfmoves());
        }
        let graph = self.graph.as_ref().map(|_| {
            let table = node_keys.iter().zip(&halfmoves).enumerate().map(|(number, (&key, &count))| (graph_key(key, u32::from(count)), number as u32));
            Graph { table: table.collect(), halfmoves, held: None }
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
