//a root held over a move (hold_root): the rest of the tree is kept for one move, and a search that comes to one of its positions
//again finds the node with its visits
use super::*;

fn entry(tree: &Tree, node: usize) -> u64 {
    graph_key(tree.node_keys[node], u32::from(tree.graph.as_ref().expect("the graph").halfmoves[node]))
}

//every visited node's visits and the bits of its sum, by its table entry
fn numbers_by_entry(tree: &Tree) -> HashMap<u64, (u32, u64)> {
    let visited = (0..tree.nodes.len()).filter(|&node| tree.nodes[node].visit_count > 0);
    visited.map(|node| (entry(tree, node), (tree.nodes[node].visit_count, tree.nodes[node].value_sum.to_bits()))).collect()
}

fn reached_by_the_root(tree: &Tree) -> Vec<bool> {
    let mut reached = vec![false; tree.nodes.len()];
    for (node, _) in nodes_with_boards(tree) {
        reached[node] = true;
    }
    reached
}

//check_graph's look at the moves' own records, for every move in the vectors, those of a held root's part too: no descent waits
//through one, each holds its node's repeat filter, and none holds as lost a node that is not proven so
fn check_every_record(tree: &Tree) {
    let graph = tree.graph.as_ref().expect("the graph");
    for node in 0..tree.nodes.len() {
        for (place, link) in tree.linked(node) {
            let slot = tree.slot(link);
            let child = slot.child as usize;
            assert_eq!(slot.pending, 0, "node {node}'s move {place} has simulations waiting through it");
            assert_eq!(slot.repeats, repeat_filter(tree.node_keys[child], graph.halfmoves[child]), "node {node}'s move {place}: its filter");
            let lost = tree.nodes[child].proof.is_some_and(|proof| proof.outcome == Outcome::Loss);
            assert!(!slot.lost || lost, "node {node}'s move {place} holds an unproven node as lost");
        }
    }
}

fn move_and_reply(tree: &Tree) -> Vec<usize> {
    reusable_lines(tree).into_iter().find(|line| line.len() == 2).expect("a move and its reply, both with moves")
}

//a compact after hold_root keeps the old tree whole: the root's part as a compact without leaves it, then the rest, every visited
//node with the numbers it had. A search that goes on comes to some of the rest again, and those nodes have their visits; the
//sums hold throughout. The compact after the next move drops what the search did not come to, and keeps what it did
#[test]
fn a_held_roots_rest_is_kept_for_a_move_and_its_positions_come_back_with_their_visits() {
    let network = Network::new();
    let mut came_back = 0;
    for (fen, moves, visits) in [(LOCKED_PAWNS, &[][..], 6_000), (START, OPENING, 6_000), (ROOKS, &[][..], 4_000)] {
        let (mut tree, settings) = graph_tree(fen, moves);
        search_without_cache(&mut tree, visits, &settings, &network);
        let before = numbers_by_entry(&tree);
        let line = move_and_reply(&tree);
        let mut dropped = copied(&tree);
        dropped.play_line(&line);
        assert_eq!(nodes_with_boards(&dropped).len(), dropped.nodes.len(), "{fen}: a compact without a held root keeps what the root reaches");
        tree.hold_root();
        tree.play_line(&line);

        let root_part = dropped.nodes.len();
        assert!(tree.nodes.len() > root_part, "{fen}: nothing but the root's part was kept");
        let numbers = |tree: &Tree, node: usize| (tree.node_keys[node], tree.nodes[node].visit_count, tree.nodes[node].value_sum.to_bits());
        for node in 0..root_part {
            assert_eq!(numbers(&tree, node), numbers(&dropped, node), "{fen}: node {node} of the root's part");
        }
        let node_of = |tree: &Tree, entry: &u64| tree.graph.as_ref().expect("the graph").table.get(entry).map(|&node| node as usize);
        for (entry, &(node_visits, sum)) in &before {
            let node = node_of(&tree, entry).expect("a visited node of the old tree is kept");
            assert_eq!((tree.nodes[node].visit_count, tree.nodes[node].value_sum.to_bits()), (node_visits, sum), "{fen}: a kept node's numbers");
        }
        check_every_record(&tree);
        check_graph(&tree);
        assert_move_means(&tree, &settings);

        let kept_visits: Vec<u32> = tree.nodes.iter().map(|node| node.visit_count).collect();
        let target = tree.root().visit_count + visits;
        search_without_cache(&mut tree, target, &settings, &network);
        let reached = reached_by_the_root(&tree);
        for node in (root_part..kept_visits.len()).filter(|&node| reached[node] && kept_visits[node] > 0) {
            came_back += 1;
            assert!(tree.nodes[node].visit_count >= kept_visits[node], "{fen}: node {node} came back with fewer visits than it had");
        }
        check_graph(&tree);
        assert_move_means(&tree, &settings);

        let entries: Vec<(u64, bool)> = (0..tree.nodes.len()).map(|node| (entry(&tree, node), reached[node])).collect();
        let line = move_and_reply(&tree);
        tree.hold_root();
        tree.play_line(&line);
        for (entry, was_reached) in entries {
            assert_eq!(node_of(&tree, &entry).is_some(), was_reached, "{fen}: the root held now keeps what it reached, and no more");
        }
        check_every_record(&tree);
        check_graph(&tree);
        assert_move_means(&tree, &settings);
    }
    assert!(came_back > 0, "no search came to a kept position again");
}

//a_move_forgets_a_proven_loss_once_the_game_has_made_its_node_a_draw with the root held: the old tree is kept whole, and every
//move to the node forgets the loss, those of the held root's part too, in both trees
#[test]
fn the_moves_of_a_held_roots_rest_forget_a_loss_the_game_has_made_a_draw() {
    let settings = graph_settings(true);
    let (mut tree, third_time) = searched_to_the_third_time(&settings);
    let moves_to = |tree: &Tree| -> Vec<u32> {
        let node = tree.graph.as_ref().expect("a tree with transpositions").table[&third_time];
        let links = (0..tree.nodes.len()).flat_map(|parent| tree.linked(parent).map(|(_, link)| link));
        links.filter(|&link| tree.link_child(link) == node).collect()
    };
    let node = tree.graph.as_ref().expect("a tree with transpositions").table[&third_time] as usize;
    tree.nodes[node].proof = Some(Proof { outcome: Outcome::Loss, value: 0.0, plies: 3 });
    let marked = moves_to(&tree);
    for &link in &marked {
        tree.slot_mut(link).lost = true;
    }
    let line = edges_of_uci(&tree, &SHUFFLE[2..4]);
    let line_moves: Vec<PackedMove> = line.iter().map(|&edge| tree.chess_move(edge)).collect();
    let mut shared = SharedTree::from_tree(&tree);
    shared.hold_root(usize::MAX);
    shared.move_root(&edges_of(&shared, &line_moves));
    shared.drop_stale_leaves(&settings, None);
    tree.hold_root();
    tree.play_line(&line);
    tree.drop_stale_leaves(&settings, None);
    let unreached = reached_by_the_root(&tree).iter().filter(|&&reached| !reached).count();
    assert!(unreached > 0, "the held root kept nothing the root does not reach");
    for (name, kept) in [("the tree", &tree), ("the shared tree", &shared.to_tree())] {
        let to_the_node = moves_to(kept);
        assert_eq!(to_the_node.len(), marked.len(), "{name}: the moves to the node, the held root's part's too");
        assert!(to_the_node.iter().all(|&link| !kept.slot(link).lost) && lost_moves(kept) == 0, "{name}: a move still holds the loss");
        check_every_record(kept);
    }
}

//the held root's part is kept while the tree's vectors take at most one part in HELD_SHARE of the memory limit
#[test]
fn a_tree_past_its_share_of_the_memory_limit_keeps_no_held_roots_rest() {
    let (mut tree, settings) = graph_tree(START, OPENING);
    search_without_cache(&mut tree, 6_000, &settings, &Network::new());
    let line = move_and_reply(&tree);
    for (short_by, kept) in [(0, true), (1, false)] {
        let mut moved = copied(&tree);
        moved.memory_limit = moved.allocated_bytes() * HELD_SHARE - short_by;
        moved.hold_root();
        moved.play_line(&line);
        assert_eq!(nodes_with_boards(&moved).len() < moved.nodes.len(), kept, "a limit {short_by} short of {HELD_SHARE} times the tree");
    }
}

//Tree against SharedTree on one thread, both holding the root over the line: the SharedTree's old root is not dead, nothing
//of it is freed, and the search that goes on makes the Tree's tree, number for number, with nodes in it that carry the visits of
//the tree before the move. A drain then frees all that the root does not reach
#[test]
fn a_shared_tree_that_holds_its_root_searches_on_as_a_tree_that_does() {
    let tables = tables();
    let (mut continued, mut carried) = (0, 0);
    for (fen, moves, simulations, certainty, tables) in cases(&tables).into_iter().take(4) {
        let settings = SearchSettings { transpositions: true, ..settings(certainty) };
        let mut tree = tree_for(fen, moves, &settings);
        search(&mut tree, simulations, usize::MAX, &settings, tables, &mut Network::new());
        for line in reusable_lines(&tree).into_iter().take(2) {
            let line_moves: Vec<PackedMove> = line.iter().map(|&edge| tree.chess_move(edge)).collect();
            let mut old = copied(&tree);
            old.hold_root();
            old.play_line(&line);
            old.drop_stale_leaves(&settings, tables);
            search(&mut old, simulations + 3_000, usize::MAX, &settings, tables, &mut Network::new());
            old.compact(1, None);

            let mut shared = shared_tree_as_tree(fen, moves, &settings);
            shared_in_flight(&shared, simulations, 1, &settings, tables, &SharedNetwork::new(), usize::MAX);
            let in_use = shared.in_use();
            shared.hold_root(usize::MAX);
            shared.move_root(&edges_of(&shared, &line_moves));
            shared.drop_stale_leaves(&settings, tables);
            //what is dead is the lists made solid since the last move, and no node
            shared.sweep_now();
            assert_eq!(shared.in_use().nodes, in_use.nodes, "{fen}: a held root leaves no node dead");
            shared.check_refs();
            shared_in_flight(&shared, simulations + 3_000, 1, &settings, tables, &SharedNetwork::new(), usize::MAX);
            shared.check_refs();
            let reachable = shared.to_reachable_tree();
            assert_same_trees(&old, &reachable);
            carried += visits_without_a_move(&reachable);
            shared.move_root(&[]);
            shared.drain();
            check_drained(&shared);
            continued += 1;
        }
    }
    assert!(continued >= 8 && carried > 0, "{continued} continued searches, {carried} nodes that carry old visits");
}

//gc_workers_beside_searching_threads_lose_no_live_node with every root held: the sweepers free what the move before left while
//the threads take nodes of what this move left, and every count and sum holds
#[test]
fn threads_searching_beside_a_held_roots_rest_keep_every_count_and_sum() {
    let network = SharedNetwork::new();
    let (mut shared_nodes, mut reclaimed, mut moves) = (0, 0, 0);
    for certainty in [false, true] {
        let settings = SearchSettings { transpositions: true, ..settings(certainty) };
        for threads in [2, 8, 24] {
            let mut tree = shared_tree_for(START, OPENING, &settings);
            let mut root_was_proven = false;
            for _ in 0..8 {
                let searching = Arc::new(tree);
                searching.start_sweepers(2);
                let target = searching.root().visits() + 1_500;
                search_on_threads(&searching, threads, target, 2, 16, &settings, &network, usize::MAX);
                tree = owned_again(searching);
                shared_nodes += check_after_search(&tree, &settings, root_was_proven);
                let Some(edge) = best_edge(&tree) else { break };
                root_was_proven = leads_to_a_proven_node(&tree, edge);
                tree.hold_root(usize::MAX);
                tree.move_root(&[edge]);
                tree.drop_stale_leaves(&settings, None);
                tree.check_refs();
                moves += 1;
            }
            reclaimed += tree.counts.reclaimed.load(Ordering::Relaxed);
            tree.drain();
            check_drained(&tree);
        }
    }
    assert!(shared_nodes > 0 && reclaimed > 0 && moves >= 48, "{shared_nodes} shared nodes, {reclaimed} freed, {moves} moves");
}

//what a held root kept is the first to go when room is short: a search that fills a memory limit set after the root was held
//frees it all, and has as much room as one whose root was not held
#[test]
fn a_search_short_of_room_frees_what_a_held_root_kept() {
    let settings = graph_settings(true);
    let filled = |held: bool| {
        let (mut tree, network) = (graph_shared_tree_after(START, OPENING), SharedNetwork::new());
        shared_in_flight(&tree, 3_000, 2, &settings, None, &network, usize::MAX);
        let (limit, nodes) = (tree.allocated_bytes() * 5 / 4, tree.in_use().nodes);
        if held {
            tree.hold_root(usize::MAX);
        }
        tree.move_root(&[best_edge(&tree).expect("a move")]);
        tree.sweep_now();
        assert_eq!(tree.in_use().nodes == nodes, held, "the old root is dead, and its rest freed, unless it is held");
        shared_in_flight(&tree, u32::MAX >> 1, 2, &settings, None, &network, limit);
        assert!(tree.is_full() && tree.allocated_bytes() <= limit, "{} bytes of {limit}", tree.allocated_bytes());
        assert!(!tree.release_held() && !tree.has_garbage(), "a full tree with a root held or garbage left");
        tree.check_refs();
        (tree.in_use().nodes, tree.counts.assists.load(Ordering::Relaxed))
    };
    let ((held_nodes, assists), (dropped_nodes, _)) = (filled(true), filled(false));
    assert!(assists > 0, "no descent freed what the held root kept");
    assert!(held_nodes * 10 >= dropped_nodes * 9, "{held_nodes} nodes in the memory with the root held, {dropped_nodes} without");
}

//without transpositions no move finds a dead node again: nothing is held
#[test]
fn a_tree_without_transpositions_holds_no_root() {
    let (settings, network) = (settings(true), SharedNetwork::new());
    let mut tree = shared_tree_after(START, OPENING);
    shared_in_flight(&tree, 1_000, 2, &settings, None, &network, usize::MAX);
    tree.hold_root(usize::MAX);
    tree.move_root(&[best_edge(&tree).expect("a move")]);
    assert!(tree.has_garbage() && !tree.release_held(), "the old root is dead at once");
}

//threads short of room beside sweepers: a search is full only once the held root's rest is given up too, whichever thread was
//freeing the last of the garbage when another looked for some
#[test]
fn threads_short_of_room_never_end_a_search_with_a_root_still_held() {
    let settings = graph_settings(true);
    for _ in 0..30 {
        let (mut tree, network) = (graph_shared_tree_after(START, OPENING), SharedNetwork::new());
        shared_in_flight(&tree, 2_000, 2, &settings, None, &network, usize::MAX);
        let limit = tree.allocated_bytes() * 5 / 4;
        for _ in 0..2 {
            let Some(edge) = best_edge(&tree) else { break };
            tree.hold_root(usize::MAX);
            tree.move_root(&[edge]);
            let searching = Arc::new(tree);
            searching.start_sweepers(2);
            search_on_threads(&searching, 8, u32::MAX >> 1, 2, 16, &settings, &network, limit);
            tree = owned_again(searching);
            assert!(tree.is_full() && tree.allocated_bytes() <= limit, "{} bytes of {limit}", tree.allocated_bytes());
            assert!(!tree.has_garbage() && !tree.release_held(), "a search ended full with garbage left or a root still held");
            tree.check_refs();
        }
    }
}

//a tree that takes more than half its memory limit holds no root: what a search took of the rest would leave it less room than
//the tree has had
#[test]
fn a_tree_past_half_its_memory_limit_holds_no_root() {
    let (settings, network) = (graph_settings(true), SharedNetwork::new());
    for (short_by, held) in [(0, true), (1, false)] {
        let mut tree = graph_shared_tree_after(START, OPENING);
        shared_in_flight(&tree, 1_000, 2, &settings, None, &network, usize::MAX);
        let nodes = tree.in_use().nodes;
        tree.hold_root(2 * tree.allocated_bytes() - short_by);
        tree.move_root(&[best_edge(&tree).expect("a move")]);
        tree.sweep_now();
        assert_eq!(tree.in_use().nodes == nodes, held, "a limit {short_by} bytes short of twice the tree");
    }
}

//under a memory limit a Tree past its share keeps nothing of a held root, so its search to the limit is the search of a tree
//that held none, number for number. One within its share has the same room, of which the held root's part takes its own
#[test]
fn a_trees_search_to_its_memory_limit_is_the_same_search_once_it_is_past_its_share() {
    let (mut tree, settings) = graph_tree(START, OPENING);
    search(&mut tree, 6_000, usize::MAX, &settings, None, &mut Network::new());
    let line = move_and_reply(&tree);
    let filled = |held: bool, limit_for: fn(usize) -> usize| {
        let mut moved = copied(&tree);
        let limit = limit_for(moved.allocated_bytes());
        moved.memory_limit = limit;
        if held {
            moved.hold_root();
        }
        moved.play_line(&line);
        let kept = moved.nodes.len();
        search(&mut moved, u32::MAX >> 1, limit, &settings, None, &mut Network::new());
        assert!(moved.full && moved.allocated_bytes() <= limit, "{} bytes of {limit}", moved.allocated_bytes());
        (moved, kept)
    };
    let past_its_share = |bytes: usize| bytes * HELD_SHARE - 1;
    assert_same_trees(&filled(true, past_its_share).0, &filled(false, past_its_share).0);
    let within = |bytes: usize| bytes * HELD_SHARE;
    let ((with, kept_with), (without, kept_without)) = (filled(true, within), filled(false, within));
    assert!(kept_with > kept_without, "the held root's part was not kept");
    let (fit, fit_without) = (with.nodes.len(), without.nodes.len());
    assert!(fit * 20 >= fit_without * 19, "{fit} nodes fit in the memory with a held root's part, {fit_without} without");
    let (visits, visits_without) = (with.root().visit_count, without.root().visit_count);
    assert!(visits * 10 >= visits_without * 9, "{visits} visits before the memory was full with a held root's part, {visits_without} without");
}

//a game under a memory limit as Searches plays it with no gc threads, the root held at every move if `held`: a search to
//`visits` at the root, or until the memory is full without; the move, with a copy once the arenas give out freed room. Returns
//the visits every search made and whether it ended full, and the moves that left a root held
fn game_under_a_limit(held: bool, limit: usize, visits: Option<u32>, moves: usize) -> (Vec<(u32, bool)>, usize) {
    let (settings, network) = (graph_settings(true), SharedNetwork::new());
    let mut tree = graph_shared_tree_after(START, OPENING);
    let (mut searches, mut held_moves) = (Vec::new(), 0);
    while searches.len() < moves {
        let reused = tree.root().visits();
        shared_in_flight(&tree, visits.unwrap_or(u32::MAX >> 1), 2, &settings, None, &network, limit);
        assert!(tree.allocated_bytes() <= limit, "{} bytes of {limit}", tree.allocated_bytes());
        tree.check_refs();
        searches.push((tree.root().visits() - reused, tree.is_full()));
        let Some(edge) = best_edge(&tree) else {
            tree = graph_shared_tree_after(START, OPENING);
            continue;
        };
        if held {
            tree.hold_root(limit);
        }
        tree.move_root(&[edge]);
        tree.drop_stale_leaves(&settings, None);
        let compacted = if tree.gives_out_freed_room(limit) { tree.compact(4, None, limit) } else { None };
        match compacted {
            Some(compacted) => tree = compacted,
            None => tree.sweep_now(),
        }
        held_moves += usize::from(tree.in_use().nodes > tree.reachable().nodes);
    }
    (searches, held_moves)
}

fn one_graphs_bytes(visits: u32) -> usize {
    let tree = graph_shared_tree_after(START, OPENING);
    shared_in_flight(&tree, visits, 2, &graph_settings(true), None, &SharedNetwork::new(), usize::MAX);
    tree.allocated_bytes()
}

fn ended_full(searches: &[(u32, bool)]) -> usize {
    searches.iter().filter(|&&(_, full)| full).count()
}

//under a memory limit that the tree takes more than half of, no root is held: the game is the game without, search for search,
//whether the searches end at their visits or fill the memory
#[test]
fn a_game_near_its_memory_limit_is_the_game_that_holds_no_root() {
    let limit = one_graphs_bytes(1_500) * 3 / 2;
    for (visits, full) in [(Some(1_500), 0), (None, 30)] {
        let ((with, held_moves), (without, _)) = (game_under_a_limit(true, limit, visits, 30), game_under_a_limit(false, limit, visits, 30));
        assert_eq!(with, without, "the searches' visits and whether they ended full, to {visits:?} visits");
        assert_eq!((ended_full(&with), held_moves), (full, 0), "the searches to {visits:?} visits that ended full, and the roots held");
    }
}

//under roomier limits, from one the tree just takes half of, the roots are held and no search to a number of visits is refused
//any, as none is in the game that holds no root
#[test]
fn a_search_to_its_visits_under_a_memory_limit_is_not_refused_for_a_held_root() {
    let bytes = one_graphs_bytes(1_500);
    for limit in [bytes * 9 / 4, bytes * 5 / 2, bytes * 5] {
        let ((with, held_moves), (without, _)) = (game_under_a_limit(true, limit, Some(1_500), 40), game_under_a_limit(false, limit, Some(1_500), 40));
        assert_eq!((ended_full(&with), ended_full(&without)), (0, 0), "searches that ended full under {limit} bytes for a tree of {bytes}");
        assert!(held_moves >= 10, "{held_moves} moves left a root held under {limit} bytes for a tree of {bytes}");
    }
}
