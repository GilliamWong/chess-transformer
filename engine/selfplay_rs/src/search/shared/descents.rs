//the reference for SharedTree::gather: a call's descents made one after another, each from the root, as gather made them
//before it sent them down in runs. On one thread gather must leave the tree this leaves, number for number
use super::*;

impl SharedTree {
    //SharedTree::gather by gather_in_by_descents
    pub fn gather_by_descents(
        &self,
        round: &Round,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        cache: Option<&SharedCache>,
        boards: &mut Vec<u8>,
    ) -> Vec<PendingLeaf> {
        let mut hold = self.hold();
        let mut call = Call::new(Vec::with_capacity(LEAVES_AT_ONCE.min(round.leaves as usize)), None);
        self.gather_in_by_descents(&mut hold, round, &Together::default(), settings, tables, cache, boards, &mut call);
        self.end_call(call)
    }

    //one simulation down to a leaf; the path ends on the node it reached. Without transpositions it counts itself in flight at
    //every node on the way; with them at every move it goes on through, and at the node it ends at (arrive). With
    //transpositions `edges` gets the links of the moves between the path's nodes, and the descent may stop above a shared
    //node (a repeat or a move's first visit, as Tree::gather's), that node the path's last but counted in flight neither
    //itself nor at the move to it; a move it passes takes its node's mean. The moves are not played on `board` but left in
    //`unplayed`, for where the position is needed. `gives`: the descent may give a list's moves links (link_known_moves),
    //which take room it has not reserved
    #[allow(clippy::too_many_arguments)]
    fn descend(
        &self,
        settings: &SearchSettings,
        board: &mut Chess,
        unplayed: &mut Vec<usize>,
        keys: &mut Vec<u64>,
        path: &mut Vec<usize>,
        edges: &mut Vec<usize>,
        gives: bool,
    ) -> Arrival {
        let path_start = keys.len() - 1;
        let mut node = self.root;
        path.push(node);
        loop {
            match self.arrive(node) {
                Arrival::Inner => {
                    //a descent that comes to a node due a pass gives its moves the nodes their positions have; a list's
                    //moves, which take links for them, only if the descent has room to give (`gives`)
                    if self.graph.is_some() {
                        let (children, visits) = (self.nodes.get(node).children(), self.nodes.get(node).visits());
                        //and only while half the links are left: the descents count on the rest for their own (free_descents)
                        let has_room = matches!(children, Children::Solid(_)) || gives && self.links.room() > Links::CAPACITY / 2;
                        if has_room && self.begins_known_pass(self.nodes.get(node), visits) {
                            self.play_path(board, unplayed);
                            self.link_known_moves(node, children, visits, board);
                        }
                    }
                    let repeats = self.graph.as_ref().map(|_| Repeats::new(keys, path_start, u32::from(self.nodes.get(node).halfmoves())));
                    let (edge, selected) = self.select_edge(node, settings, repeats.as_ref());
                    unplayed.push(edge);
                    let (child, link) = self.child_node(node, edge, selected, board, unplayed);
                    node = child;
                    path.push(node);
                    let child = self.nodes.get(node);
                    let mut stop = None;
                    if let Some(repeats) = &repeats {
                        edges.push(link as usize);
                        //a repeat of this path's position, whatever the node is to other paths, or a node that knows
                        //more than its move
                        if repeats.repeats(child.key(), child.halfmoves()) {
                            stop = Some(Arrival::Repeat);
                        } else if child.has_edges() && child.proof().is_none() {
                            stop = self.meet(path[path.len() - 2], link, child).then_some(Arrival::FirstVisit);
                        }
                    }
                    keys.push(child.key());
                    if let Some(stop) = stop {
                        return stop;
                    }
                    if self.graph.is_some() {
                        self.links.slot(link).add_pending(1);
                    }
                }
                other => return other,
            }
        }
    }

    //a descent that ended at a leaf without a backup is in flight no more: at the nodes it counted itself at, and at its
    //path's moves (`links`)
    fn take_off(&self, path: &[usize], links: &[usize]) {
        for &node in self.in_flight_nodes(path, true) {
            self.nodes.get(node).take_off(1);
        }
        for (&node, &link) in path.iter().zip(links) {
            self.links.slot(self.current_link(node, link as u32)).take_pending(1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gather_in_by_descents<'a>(
        &'a self,
        hold: &mut Hold<'a>,
        round: &Round,
        together: &Together,
        settings: &SearchSettings,
        tables: Option<&Tables>,
        cache: Option<&SharedCache>,
        boards: &mut Vec<u8>,
        call: &mut Call,
    ) {
        let root_covered = tables.is_some_and(|tables| tables.covers(&self.board));
        let mut keys = self.keys.clone();
        let root_keys = keys.len();
        let mut path = Vec::new();
        let mut edges = Vec::new();
        let mut unplayed = Vec::new();
        let mut priors = Vec::new();
        let mut free_descents = self.free_descents(round.memory_limit);
        self.hold_back_freed_room(round.memory_limit);
        loop {
            //a writer let in here may have ended the search and closed the tree (end's quiet), so the room is looked at
            //after: a leaf taken in a closed tree belongs to a search that is gone
            hold.give_way();
            self.solidify_queued(hold);
            let short = together.leaves.load(Ordering::Acquire) < round.leaves
                && together.simulations.load(Ordering::Acquire) < round.simulations
                && !together.ended.load(Ordering::Acquire);
            if !(short && self.has_room(round.visit_limit)) {
                break;
            }
            if let Some((alone, call_for_help)) = call.help
                && call.simulations >= alone
            {
                call.help = None;
                call_for_help();
            }
            let reserved = free_descents == 0;
            if reserved {
                //the root's own evaluation always fits, so that a full search still has moves to report
                if !self.reserve(round.memory_limit, self.root().has_edges()) {
                    self.full.store(true, Ordering::Relaxed);
                    break;
                }
            } else {
                free_descents -= 1;
            }
            let mut board = self.board.clone();
            keys.truncate(root_keys);
            path.clear();
            edges.clear();
            unplayed.clear();
            //near the memory limit, where every descent reserves its room, no list's moves are given links: a pass's links
            //are in no reservation
            let arrival = self.descend(settings, &mut board, &mut unplayed, &mut keys, &mut path, &mut edges, !reserved);
            let give_back = || self.release(round.memory_limit, reserved);
            let node = *path.last().expect("the root at least");
            let owned = match arrival {
                Arrival::Inner => unreachable!("a descent ends on a leaf"),
                Arrival::Proven(proof) => {
                    give_back();
                    call.simulations += 1;
                    together.simulations.fetch_add(1, Ordering::AcqRel);
                    self.back_up(&path, &edges, 1.0 - proof.value, false);
                    //with transpositions this parent may not have met the proof, made on another path
                    if self.graph.is_some() && settings.certainty && self.proves_parent(&path) {
                        self.exclusive(hold, || self.propagate_proofs(&path));
                    }
                    self.learn_proofs(&path, &edges);
                    continue;
                }
                Arrival::Repeat | Arrival::FirstVisit => {
                    give_back();
                    call.simulations += 1;
                    together.simulations.fetch_add(1, Ordering::AcqRel);
                    let shared = path.pop().expect("the shared node");
                    match arrival {
                        //the move's first visit is in its node's sum since the descent stopped (meet)
                        Arrival::FirstVisit => {
                            let (&parent, above) = path.split_last().expect("the node the descent stopped below");
                            self.back_up_means(above, &edges[..above.len()], parent, (0, 1));
                        }
                        //a repeat is a visit of the node too, at a draw's score for the side that moved into it
                        //(Tree::gather's)
                        _ => {
                            self.add_own_score(shared, 1.0 - settings.draw);
                            self.back_up_means(&path, &edges, shared, (0, 0));
                        }
                    }
                    continue;
                }
                Arrival::Collision => {
                    give_back();
                    if call.collisions < round.collisions as usize {
                        call.collide(&path, &edges, self.in_flight_nodes(&path, true), &mut None);
                        continue;
                    }
                    self.take_off(&path, &edges);
                    break;
                }
                Arrival::Leaf { owned } => owned,
            };

            self.play_path(&mut board, &mut unplayed);
            let mut moves = board.legal_moves();
            let node_tables = tables.filter(|_| node != self.root && (!root_covered || board.halfmoves() == 0));
            //the root is searched whatever the rules say of it: Searches.start refuses only a mate or stalemate
            let game_over = if node == self.root { None } else { game_result(&board, &keys, &moves, settings.draw, node_tables) };
            //a repeat inside the search is a draw, but not a proven one: the game need not repeat
            let (&leaf_key, path_keys) = keys[root_keys - 1..].split_last().expect("the root's key");
            //with transpositions a leaf is no repeat inside the search (Tree::gather's)
            let value = game_over.or_else(|| path_keys.contains(&leaf_key).then_some(settings.draw));
            //no game's end, and the node is another descent's to evaluate: Tree::gather's collision
            if value.is_none() && !owned {
                give_back();
                if call.collisions < round.collisions as usize {
                    call.collide(&path, &edges, self.in_flight_nodes(&path, true), &mut None);
                    continue;
                }
                self.take_off(&path, &edges);
                break;
            }
            call.simulations += 1;
            together.simulations.fetch_add(1, Ordering::AcqRel);
            if let Some(value) = value {
                give_back();
                match game_over {
                    //the result is for the side to move, the proof for the side that moved here. A win or loss with moves
                    //left is the tables', after any mate
                    Some(result) if settings.certainty && node != self.root => {
                        let (outcome, plies) = if result == 0.0 {
                            (Outcome::Win, if moves.is_empty() { 0 } else { TABLEBASE_PLIES })
                        } else if result == 1.0 {
                            (Outcome::Loss, TABLEBASE_PLIES)
                        } else {
                            (Outcome::Draw, 0)
                        };
                        let proof = Proof { outcome, value: 1.0 - result, plies };
                        let leaf = self.nodes.get(node);
                        //with transpositions a path's repeat may visit the leaf beside this, at a draw's score: that is a drawn
                        //leaf's proven score, and the position right after a capture or pawn move repeats none
                        let scored_alike = self.graph.is_none() || outcome == Outcome::Draw || leaf.halfmoves() == 0;
                        if leaf.visits() == 0 && scored_alike {
                            //a leaf nobody has visited, and nobody can before its proof is seen: it rewrites no sum
                            leaf.set_proof(proof);
                            self.back_up(&path, &edges, value, true);
                            if self.proves_parent(&path) {
                                self.exclusive(hold, || self.propagate_proofs(&path));
                            }
                        } else {
                            self.exclusive(hold, || {
                                self.prove(&path, proof);
                                self.back_up(&path, &edges, value, true);
                                self.propagate_proofs(&path);
                            });
                        }
                    }
                    _ => {
                        self.back_up(&path, &edges, value, true);
                        if settings.certainty && self.proves_parent(&path) {
                            self.exclusive(hold, || self.propagate_proofs(&path));
                        }
                    }
                }
                if settings.certainty {
                    self.learn_proofs(&path, &edges);
                }
                continue;
            }
            sort_moves_in_python_order(&board, &mut moves);
            //a hash collision would give another number of moves
            let cached = cache.and_then(|cache| cache.get(leaf_key, &mut priors)).filter(|_| priors.len() == moves.len());
            match cached {
                Some(value) => {
                    give_back();
                    self.expand(node, &moves, &priors);
                    self.back_up(&path, &edges, f64::from(value), true);
                }
                None => {
                    //the round's last leaf may have gone to another of its calls meanwhile: this one is left for a later round
                    if together.leaves.fetch_add(1, Ordering::AcqRel) >= round.leaves {
                        together.leaves.fetch_sub(1, Ordering::AcqRel);
                        self.take_off(&path, &edges);
                        give_back();
                        break;
                    }
                    encode_board(&board, boards);
                    self.waiting.fetch_add(1, Ordering::AcqRel);
                    call.leaves.push(PendingLeaf { path: path.clone(), edges: edges.clone(), key: leaf_key, moves, turn: board.turn() });
                    give_back();
                }
            }
        }
    }
}
