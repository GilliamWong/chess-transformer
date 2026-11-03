//Searches with concurrent_descents: the searches are SharedTrees (search/shared.rs) that the networks' threads descend, finish
//and back up in at once, so a call holds the state lock only to find them and to queue or take a round
use super::*;
use search::SharedTree;

pub(crate) struct SharedSearch {
    id: u64,
    fen: String,
    moves: Vec<String>,
    simulations: u32,
    pub(crate) stopped: bool,
    tree: Arc<SharedTree>,
}

impl SharedSearch {
    fn done(&self) -> bool {
        (self.stopped || self.tree.root().visits() >= self.simulations || self.tree.is_full()) && self.tree.pending() == 0
    }

    //done is checked again with no call in the tree: a thread that was gathering when it looked done may have a leaf out by now
    fn end(&self) -> Option<SearchReport> {
        if !self.done() {
            return None;
        }
        self.tree.quiet(|| {
            self.done().then(|| {
                self.tree.close();
                self.report()
            })
        })
    }

    //the chosen move's score; the root's for its side to move while the move has no visit (one simulation)
    fn score(&self, best: Option<usize>) -> f64 {
        let tree = &self.tree;
        match best.and_then(|edge| tree.child(tree.root(), edge)) {
            Some(chosen) if chosen.visits() > 0 => chosen.mean_score(chosen.visits()),
            _ => 1.0 - tree.root().mean_score(tree.root().visits()),
        }
    }

    fn root_score(&self) -> f64 {
        self.score(self.tree.principal_variation().first().copied())
    }

    fn report(&self) -> SearchReport {
        let tree = &self.tree;
        let line = tree.principal_variation();
        let score = self.score(Some(line[0]));
        let moves = line.iter().map(|&edge| uci(tree.chess_move(edge)));
        let root_moves = tree.edges_by_visits().into_iter().map(|edge| (uci(tree.chess_move(edge)), tree.visits(tree.root(), edge)));
        (self.id, moves.collect(), score, root_moves.collect(), tree.root().visits())
    }
}

impl State {
    fn shared_search(&self, id: u64) -> Option<&SharedSearch> {
        self.shared_searches.iter().find(|search| search.id == id)
    }
}

fn no_search(id: u64) -> PyErr {
    PyValueError::new_err(format!("no search {id}"))
}

impl Searches {
    pub(crate) fn shared_tree(&self, id: u64) -> PyResult<Option<Arc<SharedTree>>> {
        Ok(self.lock()?.shared_search(id).map(|search| search.tree.clone()))
    }

    pub(crate) fn shared_start(&self, id: u64, fen: &str, moves: Vec<String>, simulations: u32) -> PyResult<()> {
        if simulations == 0 {
            return Err(PyValueError::new_err("a search needs at least one simulation"));
        }
        let mut state = self.lock()?;
        if state.shared_search(id).is_some() {
            return Err(PyValueError::new_err(format!("search {id} is already running")));
        }
        let (board, keys) = play_out(fen, &moves)?;
        if board.legal_moves().is_empty() {
            return Err(PyValueError::new_err(format!("the game is over, there is nothing to search: {fen} {moves:?}")));
        }
        let tree = self
            .shared_reused_tree(&mut state, id, fen, &moves)
            .unwrap_or_else(|| SharedTree::new(board, keys).with_transpositions(self.settings.transpositions));
        state.shared_searches.push(SharedSearch { id, fen: fen.to_string(), moves, simulations, stopped: false, tree: Arc::new(tree) });
        Ok(())
    }

    //descend_round without the state lock around the gathers
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
            let searches: Vec<_> = state
                .shared_searches
                .iter()
                .filter(|search| !search.stopped)
                .map(|search| (search.id, search.simulations, search.tree.clone()))
                .collect();
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
            let queued =
                tree.gather(&limits, &self.settings, self.tables.as_ref(), self.shared_cache.as_ref(), boards).map_err(tablebase_error)?;
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

    //finish_round without the state lock around the finishes; the reports of the searches that are done
    pub(crate) fn shared_finish<N: Number>(&self, answers: &[N], round: Option<u64>) -> PyResult<Vec<SearchReport>> {
        let (out, trees, width) = {
            let mut state = self.lock()?;
            let index = match round {
                None => {
                    (!state.rounds.is_empty()).then_some(0).ok_or_else(|| PyRuntimeError::new_err("no round is waiting for its answers"))
                }
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
                .map(|&(id, _)| state.shared_search(id).map(|search| search.tree.clone()))
                .collect::<Option<_>>()
                .ok_or_else(|| PyRuntimeError::new_err("a search of the round is gone"))?;
            (out, trees, width)
        };
        let mut rows = answers.chunks_exact(width);
        //a lean answer put back in its columns of a whole row, where finish_leaf reads it
        let mut row = if out.lean { vec![0.0; self.answer_width] } else { Vec::new() };
        let mut logits = Vec::new();
        for (&(_, count), tree) in out.leaves.iter().zip(&trees) {
            let mut hold = tree.hold();
            let leaves = tree.take_round(out.number);
            assert_eq!(leaves.len(), count as usize, "the round's leaves in the tree");
            for (leaf, answer) in leaves.into_iter().zip(rows.by_ref()) {
                hold.give_way();
                if out.lean {
                    let columns = leaf.moves.iter().map(|&chess_move| self.move_ids.id(chess_move, leaf.turn));
                    for (column, number) in
                        columns.chain(std::iter::once(self.answer_width - 1)).zip(answer.iter().take(leaf.moves.len()).chain(answer.last()))
                    {
                        row[column] = number.number();
                    }
                }
                tree.finish_leaf(
                    leaf,
                    |column| if out.lean { row[column] } else { answer[column].number() },
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

    //removes the searches that are done and reports them, each once no call is in its tree
    fn shared_reports(&self, state: &mut State) -> Vec<SearchReport> {
        let mut reports = Vec::new();
        let ended = state.shared_searches.extract_if(.., |search| match search.end() {
            Some(report) => {
                reports.push(report);
                true
            }
            None => false,
        });
        for search in ended {
            if self.tree_reuse {
                state.shared_kept.insert(search.id, KeptTree { fen: search.fen, moves: search.moves, tree: Kept::Searched(search.tree) });
            }
        }
        reports
    }

    //with `if_running` a search that is gone is no error: another thread's finish has ended it and reports it
    fn shared_stop_with(&self, id: u64, if_running: bool) -> PyResult<Option<SearchReport>> {
        let mut state = self.lock()?;
        let Some(search) = state.shared_searches.iter_mut().find(|search| search.id == id) else {
            return if if_running { Ok(None) } else { Err(no_search(id)) };
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

    pub(crate) fn shared_root_score_if_running(&self, id: u64) -> PyResult<Option<f64>> {
        Ok(self.lock()?.shared_search(id).map(SharedSearch::root_score))
    }

    pub(crate) fn shared_root_numbers(&self, id: u64) -> PyResult<Option<RootNumbers>> {
        Ok(self.shared_tree(id)?.map(|tree| {
            let (most, second) = tree.two_most_visits();
            (tree.root().visits(), most, second)
        }))
    }

    pub(crate) fn shared_root_score(&self, id: u64) -> PyResult<f64> {
        let state = self.lock()?;
        Ok(state.shared_search(id).ok_or_else(|| no_search(id))?.root_score())
    }

    pub(crate) fn shared_report_now(&self, id: u64) -> PyResult<Option<SearchReport>> {
        let state = self.lock()?;
        let search = state.shared_search(id).ok_or_else(|| no_search(id))?;
        Ok(search.tree.root().has_edges().then(|| search.report()))
    }

    pub(crate) fn shared_running(&self, id: u64) -> PyResult<Arc<SharedTree>> {
        self.shared_tree(id)?.ok_or_else(|| no_search(id))
    }

    //ucinewgame: the id's next search starts a fresh tree, and the network's answers are forgotten
    pub(crate) fn shared_new_game(&self, id: u64) -> PyResult<()> {
        if let Some(kept) = self.lock()?.shared_kept.remove(&id) {
            kept.tree.abandon();
        }
        if let Some(cache) = self.shared_cache.as_ref() {
            cache.clear();
        }
        Ok(())
    }

    //reused_tree for a SharedTree: None while a call of its last search still holds the tree
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
                tree.reopen();
                let (line, _) = shared_line_to(&tree, since)?;
                if !line.is_empty() {
                    tree.play_line(&line);
                    tree.drop_stale_leaves(settings, tables).ok()?;
                }
                Some(tree)
            }
            Kept::Advanced(tree, candidates) => {
                let mut tree = Arc::into_inner(tree)?;
                tree.reopen();
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

    //advance for a SharedTree: the root moves down in place, and a thread finds the leaves the next root may make stale and
    //copies the kept part out of the rest
    pub(crate) fn shared_advance(&self, id: u64, fen: &str, moves: Vec<String>) -> PyResult<bool> {
        let mut state = self.lock()?;
        if state.shared_search(id).is_some() {
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
        let stops = Arc::new(Stops::default());
        let (thread_stops, threads) = (stops.clone(), self.advance_threads);
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

//the moves since (uci) as the edges from the tree's root down to the position after them and their places among their
//nodes' moves; None unless every move has its node and the last node has moves
fn shared_line_to(tree: &SharedTree, moves: &[String]) -> Option<(Vec<usize>, Vec<u16>)> {
    let (mut line, mut places, mut root) = (Vec::new(), Vec::new(), tree.root());
    for played in moves {
        let edges = root.edges();
        let edge = edges.clone().find(|&edge| uci(tree.chess_move(edge)) == *played)?;
        root = tree.child(root, edge)?;
        line.push(edge);
        places.push((edge - edges.start) as u16);
    }
    (!root.edges().is_empty()).then_some((line, places))
}
