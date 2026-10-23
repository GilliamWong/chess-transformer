//Searches with concurrent_descents: the searches are SharedTrees (search/shared.rs), so a call holds the state lock only to
//find them and to queue or take a round, and the threads that drive the networks descend, finish and back up in one
//tree at once. The plain mode's search and its locks (lib.rs) are untouched
use super::*;
use search::{SharedCache, SharedTree};

//a running search of concurrent mode; the tree is shared with the calls in it
pub(crate) struct SharedSearch {
    pub(crate) id: u64,
    pub(crate) fen: String,
    pub(crate) moves: Vec<String>,
    pub(crate) simulations: u32,
    pub(crate) stopped: bool,
    pub(crate) tree: Arc<SharedTree>,
}

impl SharedSearch {
    fn done(&self) -> bool {
        (self.stopped || self.tree.root().visits() >= self.simulations || self.tree.is_full()) && self.tree.pending() == 0
    }

    //as player.search_move reports it
    fn report(&self) -> SearchReport {
        let tree = &self.tree;
        let line = tree.principal_variation();
        //one simulation visits no child: then the root's score for its side to move
        let score = match tree.child(line[0]) {
            Some(chosen) if chosen.visits() > 0 => chosen.mean_score(chosen.visits()),
            _ => 1.0 - tree.root().mean_score(tree.root().visits()),
        };
        let moves = |edges: &[usize]| -> Vec<String> { edges.iter().map(|&edge| uci(tree.chess_move(edge))).collect() };
        let root_moves = tree.edges_by_visits().into_iter().map(|edge| (uci(tree.chess_move(edge)), tree.visits(edge)));
        (self.id, moves(&line), score, root_moves.collect(), tree.root().visits())
    }
}

//the root's score of a running search (root_score_of's)
fn shared_root_score(search: &SharedSearch) -> f64 {
    let tree = &search.tree;
    match tree.principal_variation().first().and_then(|&edge| tree.child(edge)) {
        Some(chosen) if chosen.visits() > 0 => chosen.mean_score(chosen.visits()),
        _ => 1.0 - tree.root().mean_score(tree.root().visits()),
    }
}

impl Searches {
    //the running search `id`'s tree
    pub(crate) fn shared_tree(&self, id: u64) -> PyResult<Option<Arc<SharedTree>>> {
        Ok(self.lock()?.shared_searches.iter().find(|search| search.id == id).map(|search| search.tree.clone()))
    }

    pub(crate) fn shared_start(&self, id: u64, fen: &str, moves: Vec<String>, simulations: u32) -> PyResult<()> {
        if simulations == 0 {
            return Err(PyValueError::new_err("a search needs at least one simulation"));
        }
        let mut state = self.lock()?;
        if state.shared_searches.iter().any(|search| search.id == id) {
            return Err(PyValueError::new_err(format!("search {id} is already running")));
        }
        let (board, keys) = play_out(fen, &moves)?;
        if board.legal_moves().is_empty() {
            return Err(PyValueError::new_err(format!("the game is over, there is nothing to search: {fen} {moves:?}")));
        }
        let tree = self.shared_reused_tree(&mut state, id, fen, &moves).unwrap_or_else(|| SharedTree::new(board, keys));
        let tree = Arc::new(tree);
        state.shared_searches.push(SharedSearch { id, fen: fen.to_string(), moves, simulations, stopped: false, tree });
        Ok(())
    }

    //descend_round without the state lock around the gathers: the boards of up to `leaves` new leaves of every search
    pub(crate) fn shared_descend(
        &self,
        leaves: u32,
        collisions: u32,
        simulations: Option<u32>,
        round: Option<u64>,
        boards: &mut Vec<u8>,
        mut columns: Option<&mut Vec<u16>>,
    ) -> PyResult<()> {
        let (number, searches) = {
            let mut state = self.lock()?;
            if let Some(name) = round
                && state.rounds.iter().any(|out| out.name == Some(name))
            {
                return Err(PyValueError::new_err(format!("round {name} is still waiting for its answers")));
            }
            let number = state.rounds_descended;
            state.rounds_descended += 1;
            let searches: Vec<_> =
                state.shared_searches.iter().filter(|search| !search.stopped).map(|search| (search.id, search.simulations, search.tree.clone())).collect();
            (number, searches)
        };
        let mut round_leaves = Vec::new();
        for (id, visit_limit, tree) in searches {
            let limits = Round {
                id: number,
                leaves,
                simulations: simulations.unwrap_or(leaves),
                collisions,
                visit_limit,
                memory_limit: self.memory_limit,
            };
            let queued = tree
                .gather(&limits, &self.settings, self.tables.as_ref(), self.shared_cache.as_ref(), boards)
                .map_err(tablebase_error)?;
            if !queued.is_empty() {
                round_leaves.push((id, queued.len() as u32));
            }
            if let Some(columns) = columns.as_deref_mut() {
                let score = (self.answer_width - 1) as u16;
                for leaf in &queued {
                    if leaf.moves.len() > LEGAL_SLOTS {
                        return Err(PyRuntimeError::new_err(format!(
                            "a leaf with {} legal moves, past a lean row's {LEGAL_SLOTS}",
                            leaf.moves.len()
                        )));
                    }
                    columns.extend(leaf.moves.iter().map(|&chess_move| self.move_ids.id(chess_move, leaf.turn) as u16));
                    columns.extend(std::iter::repeat_n(score, LEAN_COLUMNS - leaf.moves.len()));
                }
            }
            tree.queue(number, queued);
        }
        self.lock()?.rounds.push_back(RoundOut { number, name: round, leaves: round_leaves, lean: columns.is_some() });
        Ok(())
    }

    //finish_round without the state lock around the finishes. Returns the reports of the searches that are done
    pub(crate) fn shared_finish<N: Number>(&self, answers: &[N], round: Option<u64>) -> PyResult<Vec<SearchReport>> {
        let (out, trees) = {
            let mut state = self.lock()?;
            let index = match round {
                None => (!state.rounds.is_empty()).then_some(0).ok_or_else(|| PyRuntimeError::new_err("no round is waiting for its answers")),
                Some(name) => state
                    .rounds
                    .iter()
                    .position(|out| out.name == Some(name))
                    .ok_or_else(|| PyRuntimeError::new_err(format!("round {name} is not waiting for its answers"))),
            }?;
            let boards = state.rounds[index].leaves.iter().map(|&(_, leaves)| leaves as usize).sum();
            let width = if state.rounds[index].lean { LEAN_COLUMNS } else { self.answer_width };
            answers_for(answers, boards, width)?;
            let out = state.rounds.remove(index).expect("the round");
            let trees: Vec<Arc<SharedTree>> = out
                .leaves
                .iter()
                .map(|&(id, _)| state.shared_searches.iter().find(|search| search.id == id).map(|search| search.tree.clone()))
                .collect::<Option<_>>()
                .ok_or_else(|| PyRuntimeError::new_err("a search of the round is gone"))?;
            (out, trees)
        };
        let width = if out.lean { LEAN_COLUMNS } else { self.answer_width };
        let mut rows = answers.chunks_exact(width);
        //a lean answer put back in its columns of a whole row, where finish_leaf reads it
        let mut row = if out.lean { vec![0.0; self.answer_width] } else { Vec::new() };
        let mut logits = Vec::new();
        for (&(_, count), tree) in out.leaves.iter().zip(&trees) {
            let _hold = tree.hold();
            let leaves = tree.take_round(out.number);
            assert_eq!(leaves.len(), count as usize, "the round's leaves in the tree");
            for (leaf, answer) in leaves.into_iter().zip(rows.by_ref()) {
                if out.lean {
                    let columns = leaf.moves.iter().map(|&chess_move| self.move_ids.id(chess_move, leaf.turn));
                    for (column, number) in
                        columns.chain(std::iter::once(self.answer_width - 1)).zip(answer.iter().take(leaf.moves.len()).chain(answer.last()))
                    {
                        row[column] = number.number();
                    }
                }
                let lean = out.lean;
                tree.finish_leaf(
                    leaf,
                    |column| if lean { row[column] } else { answer[column].number() },
                    self.answer_width,
                    &self.move_ids,
                    self.shared_cache.as_ref(),
                    &self.settings,
                    self.memory_limit,
                    &mut logits,
                );
            }
        }
        let mut state = self.lock()?;
        Ok(self.shared_reports(&mut state))
    }

    //removes the searches that are done, the reports of those that were searched; each is reported once no call is in
    //its tree
    pub(crate) fn shared_reports(&self, state: &mut State) -> Vec<SearchReport> {
        let mut reports = Vec::new();
        for search in state.shared_searches.extract_if(.., |search| search.done()) {
            reports.push(search.tree.quiet(|| search.report()));
            if self.tree_reuse {
                state.shared_kept.insert(search.id, KeptTree { fen: search.fen, moves: search.moves, tree: Kept::Searched(search.tree) });
            }
        }
        reports
    }

    //stop's: the search's report if no leaf of it waits. With `if_running` a search that is gone is not an error: another
    //thread's finish has ended it, and its report is on its way
    fn shared_stop_with(&self, id: u64, if_running: bool) -> PyResult<Option<SearchReport>> {
        let mut state = self.lock()?;
        let Some(search) = state.shared_searches.iter_mut().find(|search| search.id == id) else {
            return if if_running { Ok(None) } else { Err(PyValueError::new_err(format!("no search {id}"))) };
        };
        if search.tree.root().visits() == 0 && search.tree.pending() == 0 {
            return Err(PyRuntimeError::new_err(format!("search {id} was stopped before its first simulation")));
        }
        search.stopped = true;
        Ok(self.shared_reports(&mut state).into_iter().next())
    }

    pub(crate) fn shared_stop(&self, id: u64) -> PyResult<Option<SearchReport>> {
        self.shared_stop_with(id, false)
    }

    //the round loop's stop, in either mode
    pub(crate) fn stop_running(&self, id: u64) -> PyResult<Option<SearchReport>> {
        if self.concurrent_descents { self.shared_stop_with(id, true) } else { self.stop_search(&mut *self.lock()?, id) }
    }

    //the root's score of a running search, None once it is gone
    pub(crate) fn shared_root_score_if_running(&self, id: u64) -> PyResult<Option<f64>> {
        let state = self.lock()?;
        Ok(state.shared_searches.iter().find(|search| search.id == id).map(shared_root_score))
    }

    //root_numbers of a running search, None when it is not running
    pub(crate) fn shared_root_numbers(&self, id: u64) -> PyResult<Option<RootNumbers>> {
        Ok(self.shared_tree(id)?.map(|tree| {
            let (most, second) = tree.two_most_visits();
            (tree.root().visits(), most, second)
        }))
    }

    pub(crate) fn shared_root_score(&self, id: u64) -> PyResult<f64> {
        let state = self.lock()?;
        let search =
            state.shared_searches.iter().find(|search| search.id == id).ok_or_else(|| PyValueError::new_err(format!("no search {id}")))?;
        Ok(shared_root_score(search))
    }

    pub(crate) fn shared_report_now(&self, id: u64) -> PyResult<Option<SearchReport>> {
        let state = self.lock()?;
        let search =
            state.shared_searches.iter().find(|search| search.id == id).ok_or_else(|| PyValueError::new_err(format!("no search {id}")))?;
        Ok(search.tree.root().has_edges().then(|| search.report()))
    }

    //the tree of a running search, or the error of every call that asks for one
    pub(crate) fn shared_running(&self, id: u64) -> PyResult<Arc<SharedTree>> {
        self.shared_tree(id)?.ok_or_else(|| PyValueError::new_err(format!("no search {id}")))
    }

    //a new game under the id (uci's ucinewgame): its first search starts a fresh tree, and the network's answers are forgotten
    pub(crate) fn shared_new_game(&self, id: u64) -> PyResult<()> {
        if let Some(kept) = self.lock()?.shared_kept.remove(&id) {
            kept.tree.abandon();
        }
        if let Some(cache) = self.shared_cache.as_ref() {
            cache.clear();
        }
        Ok(())
    }

    //the id's kept tree under the position after `moves`, when it is the same game and holds the moves played since, as
    //reused_tree does for a Tree. A tree is reused only once the calls of its last search have left it (it is then the
    //kept slot's alone)
    fn shared_reused_tree(&self, state: &mut State, id: u64, fen: &str, moves: &[String]) -> Option<SharedTree> {
        let kept = state.shared_kept.remove(&id)?;
        if kept.fen != fen || !moves.starts_with(&kept.moves) {
            kept.tree.abandon();
            return None;
        }
        let since = &moves[kept.moves.len()..];
        let (settings, tables) = (&self.settings, self.tables.as_ref());
        match kept.tree.ready()? {
            Kept::Searched(tree) => {
                let mut tree = Arc::into_inner(tree)?;
                let (line, _) = shared_line_to(&tree, since)?;
                if !line.is_empty() {
                    tree.play_line(&line);
                    tree.drop_stale_leaves(settings, tables).ok()?;
                }
                Some(tree)
            }
            Kept::Advanced(tree, candidates) => {
                let mut tree = Arc::into_inner(tree)?;
                let (line, places) = shared_line_to(&tree, since)?;
                tree.move_root(&line);
                match candidates {
                    Some(candidates) => tree.drop_stale_candidates(&places, &candidates, settings, tables).ok()?,
                    None => tree.drop_stale_leaves(settings, tables).ok()?,
                }
                Some(tree)
            }
            Kept::Advancing(..) => unreachable!("ready waits for the thread"),
        }
    }

    //advance for a SharedTree: the root moves down in place, and on a thread of its own the tree finds the leaves the next
    //root may make stale and copies the kept part out of the rest
    pub(crate) fn shared_advance(&self, id: u64, fen: &str, moves: Vec<String>) -> PyResult<bool> {
        let mut state = self.lock()?;
        if state.shared_searches.iter().any(|search| search.id == id) {
            return Err(PyValueError::new_err(format!("search {id} is running")));
        }
        let Some(kept) = state.shared_kept.remove(&id) else { return Ok(false) };
        if kept.fen != fen || !moves.starts_with(&kept.moves) {
            kept.tree.abandon();
            return Ok(false);
        }
        let Some(Kept::Searched(tree) | Kept::Advanced(tree, _)) = kept.tree.ready() else { return Ok(false) };
        let Some(mut tree) = Arc::into_inner(tree) else { return Ok(false) };
        let Some((line, _)) = shared_line_to(&tree, &moves[kept.moves.len()..]) else { return Ok(false) };
        tree.move_root(&line);
        let (stops, threads) = (Arc::new(Stops::default()), self.advance_threads);
        let thread_stops = stops.clone();
        let carry = move || {
            let candidates = tree.stale_candidates(threads, &thread_stops.walk);
            if candidates.is_some() {
                drop(tree.compact(Some(&thread_stops.copy)));
            }
            (Arc::new(tree), candidates)
        };
        //without a thread the tree is not kept
        let Ok(thread) = std::thread::Builder::new().spawn(carry) else { return Ok(false) };
        state.shared_kept.insert(id, KeptTree { fen: fen.to_string(), moves, tree: Kept::Advancing(thread, stops) });
        Ok(true)
    }
}

//the moves since (uci) as the edges from the tree's root down to the position after them, and as their places among their
//nodes' moves, when every move has its node and the last node has moves (line_to's)
fn shared_line_to(tree: &SharedTree, moves: &[String]) -> Option<(Vec<usize>, Vec<u16>)> {
    let (mut line, mut places, mut root) = (Vec::new(), Vec::new(), tree.root());
    for played in moves {
        let edges = root.edges();
        let edge = edges.clone().find(|&edge| uci(tree.chess_move(edge)) == *played)?;
        root = tree.child(edge)?;
        line.push(edge);
        places.push((edge - edges.start) as u16);
    }
    (!root.edges().is_empty()).then_some((line, places))
}
