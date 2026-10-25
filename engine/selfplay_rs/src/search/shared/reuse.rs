//tree reuse for a SharedTree, Tree's (search.rs) line for line: the root moved down in place, the leaves that a rule scored
//only from the old root taken out, the part kept copied out of the rest. A tree here belongs to one thread (`&mut self`)
//or is only read by the threads of stale_candidates; no search runs in it
use super::*;

//nodes a walk or copy that may be stopped goes through between looks at its flag
const ABORT_CHECKS: usize = 4096;

//parts of the tree a thread of stale_candidates gets on the mean, so that a big part at the end leaves none idle long
const PARTS_A_THREAD: usize = 16;

//a part of the tree for one of stale_candidates' threads: a subtree (its node, and the places of the moves down to it) or a
//candidate leaf
enum Part {
    Subtree(usize, Vec<u16>),
    Leaf(Vec<u16>),
}

//a tree's storage, as compact returns the old one to free
pub type Storage = (Nodes, Edges, Links);

impl SharedTree {
    //plays a line of moves from the root (edges, each of the node the last one leads to) and makes the node it ends on the
    //root where it is, without a copy: the rest of the tree stays in the arenas until compact. The new root's proof is
    //dropped, as a root is never proven
    pub fn move_root(&mut self, line: &[usize]) {
        assert_eq!(self.pending(), 0, "a move is played once the search's leaves are back");
        for &edge in line {
            let child = self.child_index(self.nodes.get(self.root), edge).expect("a line through the tree's nodes");
            self.board.play_unchecked(self.edges.chess_move(edge).to_move(&self.board));
            self.keys.push(position_key(&self.board));
            self.root = child;
        }
        if !line.is_empty() {
            self.nodes.get(self.root).clear_proof();
            *self.full.get_mut() = false;
        }
    }

    //move_root and compact, the old tree freed on a thread of its own
    pub fn play_line(&mut self, line: &[usize]) {
        self.move_root(line);
        let old = self.compact(None);
        let _ = std::thread::Builder::new().spawn(move || drop(old));
    }

    //the root's part of the tree copied out of the rest as a tree of its own, in breadth-first order from the root, as
    //Tree::compact has it, each node's links made again (solid with the threshold's visits). With transpositions a node
    //that several moves lead to is copied once (Tree::compact_graph's), with the moves' visits, and the table made again of
    //the copied nodes. Returns the old storage to free, or None once abort is set (the tree then as it was)
    pub fn compact(&mut self, abort: Option<&AtomicBool>) -> Option<Storage> {
        let (nodes, edges, links) = (Nodes::new(), Edges::new(), Links::new());
        let mut children = Vec::new();
        let mut order = vec![self.root];
        //with transpositions each copied node's new number
        let mut copied_as = if self.graph.is_some() { vec![NO_NODE; self.nodes.len()] } else { Vec::new() };
        if let Some(first) = copied_as.get_mut(self.root) {
            *first = 0;
        }
        let mut index = 0;
        while index < order.len() {
            if index.is_multiple_of(ABORT_CHECKS) && abort.is_some_and(|abort| abort.load(Ordering::Relaxed)) {
                return None;
            }
            let old = self.nodes.get(order[index]);
            assert_eq!(nodes.push(old.key()), index);
            let node = nodes.get(index);
            old.copy_to(node);
            let old_edges = old.edges();
            if !old_edges.is_empty() {
                let first = edges.alloc(old_edges.len());
                for (place, edge) in old_edges.clone().enumerate() {
                    edges.set(first + place, self.edges.chess_move(edge), self.edges.prior(edge));
                }
                children.clear();
                for (place, link) in self.links.linked(old) {
                    let child = self.links.get(link).child() as usize;
                    let child = if copied_as.is_empty() {
                        order.push(child);
                        (order.len() - 1) as u32
                    } else {
                        if copied_as[child] == NO_NODE {
                            copied_as[child] = order.len() as u32;
                            order.push(child);
                        }
                        copied_as[child]
                    };
                    children.push((place, child, self.links.get(link).visits().load(Ordering::Relaxed)));
                }
                set_links(&links, node, old_edges.len(), old.visits() >= self.solid_threshold, children.iter().copied());
                node.publish_edges(first, old_edges.len());
            }
            index += 1;
        }
        if let Some(graph) = &mut self.graph {
            let fresh = SharedGraph::new();
            for number in 0..order.len() {
                let node = nodes.get(number);
                let entry = graph_key(node.key(), u32::from(node.halfmoves()));
                fresh.shard(entry).lock().expect("a table shard").insert(entry, number as u32);
                fresh.entries.fetch_add(1, Ordering::Relaxed);
            }
            *graph = fresh;
        }
        self.root = 0;
        //what was queued for solid links: compact made them
        self.to_solidify.get_mut().expect("the solid queue").clear();
        *self.queued.get_mut() = 0;
        Some((std::mem::replace(&mut self.nodes, nodes), std::mem::replace(&mut self.edges, edges), std::mem::replace(&mut self.links, links)))
    }

    //Tree::drop_stale_leaves: a kept tree under a new root, the leaves a rule scored that held only from the old root
    //taken out, their visits off their ancestors. With transpositions nothing is taken out (Tree's)
    pub fn drop_stale_leaves(&mut self, settings: &SearchSettings, tables: Option<&Tables>) -> Result<(), SyzygyError> {
        if self.graph.is_some() {
            return Ok(());
        }
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        let mut stale = Vec::new();
        let mut keys = self.keys.clone();
        self.find_stale_leaves(&mut vec![self.root], &mut Vec::new(), &mut keys, settings, tables, root_covered, &mut stale)?;
        for path in stale {
            self.take_out(&path);
        }
        Ok(())
    }

    //the paths to the stale leaves under the path's last node, which the line's moves (its edges) reach from the root;
    //keys end with the path's keys
    #[allow(clippy::too_many_arguments)]
    fn find_stale_leaves(
        &self,
        path: &mut Vec<usize>,
        line: &mut Vec<usize>,
        keys: &mut Vec<u64>,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        root_covered: bool,
        stale: &mut Vec<Vec<usize>>,
    ) -> Result<(), SyzygyError> {
        let node = *path.last().expect("the root at least");
        let edges = self.nodes.get(node).edges();
        if edges.is_empty() {
            if node != self.root && self.nodes.get(node).visits() > 0 && self.is_stale(path, line, keys, settings, tables, root_covered)? {
                stale.push(path.clone());
            }
            return Ok(());
        }
        for (place, link) in self.links.linked(self.nodes.get(node)) {
            let child = self.links.get(link).child() as usize;
            if self.nodes.get(child).proof().is_some() {
                continue;
            }
            path.push(child);
            line.push(edges.start + place);
            keys.push(self.nodes.get(child).key());
            let found = self.find_stale_leaves(path, line, keys, settings, tables, root_covered, stale);
            keys.pop();
            line.pop();
            path.pop();
            found?;
        }
        Ok(())
    }

    //whether the leaf at the end of path, which the line's moves (edges) reach from the root, is no longer what gather scored
    //it as; keys end with the path's keys
    fn is_stale(
        &self,
        path: &[usize],
        line: &[usize],
        keys: &[u64],
        settings: &SearchSettings,
        tables: Option<&Tables>,
        root_covered: bool,
    ) -> Result<bool, SyzygyError> {
        //as gather scores a leaf
        let mut board = self.board.clone();
        for &edge in line {
            board.play_unchecked(self.edges.chess_move(edge).to_move(&board));
        }
        let legal_moves = board.legal_moves();
        let node_tables = tables.filter(|_| !root_covered || board.halfmoves() == 0);
        let over = game_result(&board, keys, &legal_moves, settings.draw, settings.repetitions_for_a_draw, node_tables)?;
        let (&leaf_key, path_keys) = keys[keys.len() - path.len()..].split_last().expect("the leaf's key");
        Ok(over.is_none() && !path_keys.contains(&leaf_key))
    }

    //the leaves drop_stale_leaves may find stale once the root has moved further down, found before the moves are known
    //(Searches' advance, while the opponent thinks): every leaf below the root with visits, no moves and no proof of its
    //own, in the walk's order, each as the places of its line's moves among their nodes' moves, which hold across
    //compact. The walk runs on `threads` threads, each taking the next of the tree's parts, their candidates joined in
    //the parts' order. None once abort is set. With transpositions none: nothing is taken out (drop_stale_leaves)
    pub fn stale_candidates(&self, threads: usize, abort: &AtomicBool) -> Option<Vec<Vec<u16>>> {
        if self.graph.is_some() {
            return (!abort.load(Ordering::Relaxed)).then(Vec::new);
        }
        let share = (self.root().visits() / (threads * PARTS_A_THREAD).max(1) as u32).max(1);
        let mut parts = Vec::new();
        self.split_for_walk(self.root, &mut Vec::new(), share, &mut parts);
        let next = AtomicUsize::new(0);
        let walk = || {
            let (mut walked, mut looked) = (Vec::new(), 0);
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                let Some(part) = parts.get(index) else { return Some(walked) };
                let mut candidates = Vec::new();
                match part {
                    Part::Leaf(places) => candidates.push(places.clone()),
                    Part::Subtree(node, places) => {
                        if !self.find_candidates(*node, &mut places.clone(), &mut candidates, &mut looked, abort) {
                            return None;
                        }
                    }
                }
                walked.push((index, candidates));
            }
        };
        let walked: Vec<Option<_>> = std::thread::scope(|scope| {
            let workers: Vec<_> = (1..threads).map(|_| scope.spawn(walk)).collect();
            let mut walked = vec![walk()];
            walked.extend(workers.into_iter().map(|worker| worker.join().expect("a walk's thread")));
            walked
        });
        let mut walked: Vec<(usize, Vec<Vec<u16>>)> = walked.into_iter().collect::<Option<Vec<_>>>()?.into_iter().flatten().collect();
        walked.sort_unstable_by_key(|&(index, _)| index);
        Some(walked.into_iter().flat_map(|(_, candidates)| candidates).collect())
    }

    //the parts of the tree below node (its line's moves at `places`) for stale_candidates' threads, in the walk's order: a
    //subtree of at most `share` visits is one part, a bigger one is split into its moves'; a leaf met on the way that is a
    //candidate is a part of its own
    fn split_for_walk(&self, node: usize, places: &mut Vec<u16>, share: u32, parts: &mut Vec<Part>) {
        if node != self.root && self.nodes.get(node).visits() <= share {
            parts.push(Part::Subtree(node, places.clone()));
            return;
        }
        for (place, link) in self.links.linked(self.nodes.get(node)) {
            let child = self.links.get(link).child() as usize;
            places.push(place as u16);
            let node = self.nodes.get(child);
            if node.has_edges() {
                self.split_for_walk(child, places, share, parts);
            } else if node.visits() > 0 && node.proof().is_none() {
                parts.push(Part::Leaf(places.clone()));
            }
            places.pop();
        }
    }

    fn find_candidates(
        &self,
        node: usize,
        places: &mut Vec<u16>,
        candidates: &mut Vec<Vec<u16>>,
        looked: &mut usize,
        abort: &AtomicBool,
    ) -> bool {
        *looked += 1;
        if looked.is_multiple_of(ABORT_CHECKS) && abort.load(Ordering::Relaxed) {
            return false;
        }
        for (place, link) in self.links.linked(self.nodes.get(node)) {
            let child = self.links.get(link).child() as usize;
            places.push(place as u16);
            let node = self.nodes.get(child);
            if node.has_edges() {
                if !self.find_candidates(child, places, candidates, looked, abort) {
                    return false;
                }
            } else if node.visits() > 0 && node.proof().is_none() {
                candidates.push(places.clone());
            }
            places.pop();
        }
        true
    }

    //drop_stale_leaves from the candidates found at the root the moves at `line`'s places moved it down from: of those
    //below the root, the leaves its walk would find stale, taken out in its order
    pub fn drop_stale_candidates(
        &mut self,
        line: &[u16],
        candidates: &[Vec<u16>],
        settings: &SearchSettings,
        tables: Option<&Tables>,
    ) -> Result<(), SyzygyError> {
        //with transpositions nothing is taken out (drop_stale_leaves)
        if self.graph.is_some() {
            return Ok(());
        }
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        let mut keys = self.keys.clone();
        let root_keys = keys.len();
        let mut stale = Vec::new();
        'candidates: for places in candidates.iter().filter(|places| places.len() > line.len() && places.starts_with(line)) {
            let (mut path, mut edges) = (vec![self.root], Vec::new());
            for &place in &places[line.len()..] {
                let parent = self.nodes.get(*path.last().expect("the root at least"));
                let edge = parent.edges().start + usize::from(place);
                let child = self.child_index(parent, edge).expect("a candidate's moves have nodes");
                //the walk passes over a proven node and what is below it
                if self.nodes.get(child).proof().is_some() {
                    continue 'candidates;
                }
                path.push(child);
                edges.push(edge);
            }
            let leaf = self.nodes.get(*path.last().expect("a leaf below the root"));
            if !leaf.has_edges() && leaf.visits() > 0 {
                keys.extend(path[1..].iter().map(|&node| self.nodes.get(node).key()));
                let found = self.is_stale(&path, &edges, &keys, settings, tables, root_covered);
                keys.truncate(root_keys);
                if found? {
                    stale.push(path);
                }
            }
        }
        for path in stale {
            self.take_out(&path);
        }
        Ok(())
    }

    //takes a leaf's visits out of the tree: a visit's score s went to the leaf, 1 - s to its parent and so on up, and the
    //leaf is as if never visited
    fn take_out(&mut self, path: &[usize]) {
        let (&leaf, ancestors) = path.split_last().expect("a leaf below the root");
        let leaf_node = self.nodes.get(leaf);
        let (visits, value_sum) = (leaf_node.visits(), leaf_node.value_sum());
        for (distance, &ancestor) in ancestors.iter().rev().enumerate() {
            let ancestor = self.nodes.get(ancestor);
            ancestor.take_visits(visits);
            ancestor.add_value(-(if distance % 2 == 0 { f64::from(visits) - value_sum } else { value_sum }));
        }
        leaf_node.reset();
        self.refresh_visited_policy(*ancestors.last().expect("a leaf below the root"));
    }
}
