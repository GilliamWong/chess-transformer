//Tree's tree reuse (search.rs) for a SharedTree, line for line
use super::*;

impl SharedTree {
    //the rest of the tree is left dead for the sweepers; a root is never proven, so the new root's proof is dropped. The line's
    //last node is the root and keeps a count of its own, which the nodes above it lose with their moves when they are freed
    pub fn move_root(&mut self, line: &[usize]) {
        assert_eq!(self.pending(), 0, "a move is played once the search's leaves are back");
        let old_root = self.root;
        for &edge in line {
            let child = self.child_index(self.nodes.get(self.root), edge).expect("a line through the tree's nodes");
            self.board.play_unchecked(self.edges.chess_move(edge).to_move(&self.board));
            self.keys.push(position_key(&self.board));
            self.root = child;
        }
        if !line.is_empty() {
            let root = self.nodes.get(self.root);
            assert!(root.acquire(), "the line's last node is dead");
            self.drop_reference(old_root);
            root.clear_proof();
            *self.full.get_mut() = false;
            //nothing can be freed yet, so the nodes queued still have their lists
            let queued = std::mem::take(self.to_solidify.get_mut().expect("the solid queue"));
            for id in queued {
                self.make_solid(id as usize);
            }
            *self.queued.get_mut() = 0;
        }
        //no leaf waits, so no list made solid is read any more
        self.release_retired_lists();
    }

    //takes out the leaves scored by a rule that held only from the old root (Tree::drop_stale_leaves); with transpositions none, and
    //the nodes the game has since made a draw are proven
    pub fn drop_stale_leaves(&mut self, settings: &SearchSettings, tables: Option<&Tables>) {
        if self.graph.is_some() {
            self.prove_the_games_third_times(settings);
            return;
        }
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        let mut stale = Vec::new();
        let mut keys = self.keys.clone();
        self.find_stale_leaves(&mut vec![self.root], &mut Vec::new(), &mut keys, settings, tables, root_covered, &mut stale);
        for path in stale {
            self.take_out(&path);
        }
    }

    //Tree::prove_the_games_third_times; the nodes the root no longer reaches are those not freed yet
    fn prove_the_games_third_times(&mut self, settings: &SearchSettings) {
        let Some(graph) = &self.graph else { return };
        let halfmoves = self.board.halfmoves();
        let node_of = |entry: u64| graph.shard(entry).lock().expect("a table shard").get(&entry).copied();
        let nodes_of = |key| (halfmoves + 1..FIFTY_MOVE_PLIES).filter_map(move |count| node_of(graph_key(key, count)));
        let thirds: Vec<u32> = one_short_of_threefold(&self.keys, halfmoves).into_iter().flat_map(nodes_of).collect();
        for id in thirds {
            //prove keeps a proof the node has, and the rules' draw comes before it
            self.nodes.get(id as usize).clear_proof();
            self.prove(&[id as usize], Proof { outcome: Outcome::Draw, value: 1.0 - settings.draw, plies: 0 });
        }
    }

    //pushes the paths of the stale leaves below the path's last node; line is the path's edges from the root, keys end with its keys
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
    ) {
        let node = *path.last().expect("the root at least");
        let edges = self.nodes.get(node).edges();
        if edges.is_empty() {
            if node != self.root && self.nodes.get(node).visits() > 0 && self.is_stale(path, line, keys, settings, tables, root_covered) {
                stale.push(path.clone());
            }
            return;
        }
        for (place, link) in self.links.linked(self.nodes.get(node)) {
            let child = self.links.child(link) as usize;
            if self.nodes.get(child).proof().is_some() {
                continue;
            }
            path.push(child);
            line.push(edges.start + place);
            keys.push(self.nodes.get(child).key());
            self.find_stale_leaves(path, line, keys, settings, tables, root_covered, stale);
            keys.pop();
            line.pop();
            path.pop();
        }
    }

    //whether the path's leaf is no longer what gather scored; line and keys as in find_stale_leaves
    fn is_stale(
        &self,
        path: &[usize],
        line: &[usize],
        keys: &[u64],
        settings: &SearchSettings,
        tables: Option<&Tables>,
        root_covered: bool,
    ) -> bool {
        let mut board = self.board.clone();
        for &edge in line {
            board.play_unchecked(self.edges.chess_move(edge).to_move(&board));
        }
        let legal_moves = board.legal_moves();
        let node_tables = tables.filter(|_| !root_covered || board.halfmoves() == 0);
        let over = game_result(&board, keys, &legal_moves, settings.draw, node_tables);
        let (&leaf_key, path_keys) = keys[keys.len() - path.len()..].split_last().expect("the leaf's key");
        over.is_none() && !path_keys.contains(&leaf_key)
    }

    //the leaves drop_stale_leaves may find stale once the root has moved further down: those with visits, no moves and no proof,
    //as the places of their line's moves (stable across compact), in the walk's order. None once abort is set; none with transpositions
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

    fn split_for_walk(&self, node: usize, places: &mut Vec<u16>, share: u32, parts: &mut Vec<Part>) {
        if node != self.root && self.nodes.get(node).visits() <= share {
            parts.push(Part::Subtree(node, places.clone()));
            return;
        }
        for (place, link) in self.links.linked(self.nodes.get(node)) {
            let child = self.links.child(link) as usize;
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

    //false once abort is set
    fn find_candidates(
        &self,
        node: usize,
        places: &mut Vec<u16>,
        candidates: &mut Vec<Vec<u16>>,
        looked: &mut usize,
        abort: &AtomicBool,
    ) -> bool {
        *looked += 1;
        if aborted(*looked, Some(abort)) {
            return false;
        }
        for (place, link) in self.links.linked(self.nodes.get(node)) {
            let child = self.links.child(link) as usize;
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

    //drop_stale_leaves from candidates found at the old root, `line` being the places of the moves down from it; takes out in
    //the walk's order
    pub fn drop_stale_candidates(&mut self, line: &[u16], candidates: &[Vec<u16>], settings: &SearchSettings, tables: Option<&Tables>) {
        if self.graph.is_some() {
            self.prove_the_games_third_times(settings);
            return;
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
                //the walk skips a proven node and what is below it
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
                if found {
                    stale.push(path);
                }
            }
        }
        for path in stale {
            self.take_out(&path);
        }
    }

    //a visit's score s went to the leaf, 1 - s to its parent and so on up
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
