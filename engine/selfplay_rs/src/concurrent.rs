//the state lock is held only to find the searches and to queue or take a round, never around a gather or a finish
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

    //done again with no call in the tree: a thread gathering when it first looked done may have a leaf out by now
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

    //the root's score for its side to move while the chosen move has no visit (a one-simulation search), a half with none on the root
    fn score(&self, best: Option<usize>) -> f64 {
        let tree = &self.tree;
        let root_visits = tree.root().visits();
        let score = match best.and_then(|edge| tree.child(tree.root(), edge)) {
            Some(chosen) if chosen.visits() > 0 => chosen.mean_score(chosen.visits()),
            _ if root_visits > 0 => 1.0 - tree.root().mean_score(root_visits),
            _ => 0.5,
        };
        //a mean read beside backups may be off by their values
        score.clamp(0.0, 1.0)
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

    pub(crate) fn drop_shared_search(&mut self, id: u64) {
        self.shared_searches.extract_if(.., |search| search.id == id).for_each(|search| drop_later(search.tree));
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
        let tree = Arc::new(tree);
        tree.start_sweepers(self.gc_threads);
        state.shared_searches.push(SharedSearch { id, fen: fen.to_string(), moves, simulations, stopped: false, tree });
        Ok(())
    }

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
            let queued = tree.gather(&limits, &self.settings, self.tables.as_ref(), self.shared_cache.as_ref(), boards);
            if !queued.is_empty() {
                round_leaves.push((id, queued.len() as u32));
            }
            if let Some(columns) = columns.as_deref_mut() {
                for leaf in &queued {
                    self.lean_columns(columns, &leaf.moves, leaf.turn)?;
                }
            }
            tree.queue(number, queued);
        }
        self.lock()?.rounds.push_back(RoundOut { number, name: round, leaves: round_leaves, lean: columns.is_some() });
        Ok(())
    }

    pub(crate) fn shared_finish<N: Number>(&self, answers: &[N], round: Option<u64>) -> PyResult<Vec<SearchReport>> {
        let (out, trees, width) = {
            let mut state = self.lock()?;
            let index = round_index(&state.rounds, round)?;
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
        let mut logits = Vec::new();
        for (&(_, count), tree) in out.leaves.iter().zip(&trees) {
            let mut hold = tree.hold();
            let leaves = tree.take_round(out.number);
            assert_eq!(leaves.len(), count as usize, "the round's leaves in the tree");
            for (leaf, answer) in leaves.into_iter().zip(rows.by_ref()) {
                hold.give_way();
                let (cache, settings) = (self.shared_cache.as_ref(), &self.settings);
                if out.lean {
                    //the legal moves' logits in the leaf's move order, then the score
                    logits.clear();
                    logits.extend(answer[..leaf.moves.len()].iter().map(N::number));
                    let value = answer[LEAN_COLUMNS - 1].number();
                    tree.finish_leaf_in_order(leaf, &mut logits, value, cache, settings);
                } else {
                    let answer_at = |column: usize| answer[column].number();
                    tree.finish_leaf(leaf, answer_at, self.answer_width, &self.move_ids, cache, settings, &mut logits);
                }
            }
        }
        let mut state = self.lock()?;
        Ok(self.shared_reports(&mut state))
    }

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

    //with `if_running`, a search that is gone is no error: another thread's finish has ended and reported it
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
        self.shared_root_score_if_running(id)?.ok_or_else(|| no_search(id))
    }

    pub(crate) fn shared_report_now(&self, id: u64) -> PyResult<Option<SearchReport>> {
        let state = self.lock()?;
        let search = state.shared_search(id).ok_or_else(|| no_search(id))?;
        Ok(search.tree.root().has_edges().then(|| search.report()))
    }

    pub(crate) fn shared_running(&self, id: u64) -> PyResult<Arc<SharedTree>> {
        self.shared_tree(id)?.ok_or_else(|| no_search(id))
    }

    //ucinewgame: the id's next search starts a fresh tree, and the cached answers go
    pub(crate) fn shared_new_game(&self, id: u64) -> PyResult<()> {
        if let Some(kept) = self.lock()?.shared_kept.remove(&id) {
            kept.tree.abandon();
        }
        if let Some(cache) = self.shared_cache.as_ref() {
            cache.clear();
        }
        Ok(())
    }

    //None while a call of the id's last search still holds the tree. The root is moved down the line, which costs no more than the
    //line is long; the dead rest of the tree is freed beside the search, or at once with no gc threads
    fn shared_reused_tree(&self, state: &mut State, id: u64, fen: &str, moves: &[String]) -> Option<SharedTree> {
        let kept = state.shared_kept.remove(&id)?;
        if kept.fen != fen || !moves.starts_with(&kept.moves) {
            kept.tree.abandon();
            return None;
        }
        let since = &moves[kept.moves.len()..];
        let (settings, tables) = (&self.settings, self.tables.as_ref());
        //the walk's candidates of an advance, the inner None if it was stopped: without an advance, no walk was made
        let (tree, candidates) = match kept.tree.ready()? {
            Kept::Searched(tree) => (tree, None),
            Kept::Advanced(tree, candidates) => (tree, Some(candidates)),
            Kept::Advancing(..) => unreachable!("ready waits for the thread"),
        };
        let mut tree = into_owned(tree)?;
        tree.reopen();
        if tree.sprawls(self.memory_limit) {
            drop_later(tree);
            return None;
        }
        let Some((line, places)) = shared_line_to(&tree, since) else {
            drop_later(tree);
            return None;
        };
        tree.move_root(&line);
        match candidates {
            None if line.is_empty() => {}
            Some(Some(candidates)) => tree.drop_stale_candidates(&places, &candidates, settings, tables),
            _ => tree.drop_stale_leaves(settings, tables),
        }
        if self.gc_threads == 0 {
            tree.drain();
        }
        Some(tree)
    }

    //moves the kept tree on towards the next search while the opponent thinks. A tree with transpositions is moved at the next
    //search's start, which costs no more than the line is long; one without has its stale leaves looked for by a thread
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
        let Some((line, _)) = shared_line_to(&tree, &moves[kept.moves.len()..]) else {
            drop_later(tree);
            return Ok(false);
        };
        if tree.has_transpositions() {
            state.shared_kept.insert(id, KeptTree { fen: kept.fen, moves: kept.moves, tree: Kept::Searched(tree) });
            return Ok(true);
        }
        let Some(mut tree) = into_owned(tree) else { return Ok(false) };
        tree.move_root(&line);
        let stops = Arc::new(Stops::default());
        let (thread_stops, threads) = (stops.clone(), self.advance_threads);
        let carry = move || {
            let candidates = tree.stale_candidates(threads, &thread_stops.walk);
            (Arc::new(tree), candidates)
        };
        let Ok(thread) = std::thread::Builder::new().spawn(carry) else { return Ok(false) };
        state.shared_kept.insert(id, KeptTree { fen: fen.to_string(), moves, tree: Kept::Advancing(thread, stops) });
        Ok(true)
    }
}

//the tree to itself again, once the sweepers of its last search have left: None while a call of that search still holds it
fn into_owned(tree: Arc<SharedTree>) -> Option<SharedTree> {
    tree.join_sweepers();
    Arc::into_inner(tree)
}

//the edges down from the root along `moves` (uci) and each one's place among its node's moves; None unless every move has its
//node and the last node has moves
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::MOVE_ID_TABLE_LENGTH;
    use crate::search::{Round, SearchSettings};

    const ANSWER_WIDTH: usize = 1859;

    //a score is between 0 and 1 whatever was read: before the root's first visit, or beside backups that count their value first
    #[test]
    fn a_score_is_a_score_whatever_was_read() {
        let board = Chess::default();
        let tree = Arc::new(SharedTree::new(board.clone(), vec![position_key(&board)]));
        let search = SharedSearch { id: 0, fen: String::new(), moves: Vec::new(), simulations: 100, stopped: false, tree: tree.clone() };
        assert_eq!((search.root_score(), search.score(None)), (0.5, 0.5));

        let settings = SearchSettings {
            c_puct: 0.9,
            c_puct_factor: 1.9,
            c_puct_base: 38739.0,
            fpu_reduction: 0.16,
            draw: 0.5,
            value_sharpness: 1.0,
            certainty: false,
            transpositions: false,
        };
        let move_ids = MoveIds::new((0..MOVE_ID_TABLE_LENGTH).map(|index| (index % (ANSWER_WIDTH - 1)) as i16).collect());
        for id in 0..2 {
            let round = Round { id, leaves: 4, simulations: 4, collisions: 0, visit_limit: 100, memory_limit: usize::MAX };
            let queued = tree.gather(&round, &settings, None, None, &mut Vec::new());
            tree.queue(id, queued);
            let _hold = tree.hold();
            for leaf in tree.take_round(id) {
                tree.finish_leaf(leaf, |_| 0.25, ANSWER_WIDTH, &move_ids, None, &settings, &mut Vec::new());
            }
        }
        let edge = tree.principal_variation()[0];
        let chosen = tree.child(tree.root(), edge).expect("the most visited move has a node");
        assert!(chosen.visits() > 0);
        chosen.set_counts(2, 0, 3.0);
        assert_eq!(search.score(Some(edge)), 1.0);
        chosen.set_counts(2, 0, -1.0);
        assert_eq!(search.score(Some(edge)), 0.0);
    }
}
