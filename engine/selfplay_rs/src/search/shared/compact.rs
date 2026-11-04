//what the root reaches copied into new arenas on several threads, for a tree whose arenas give out freed room scattered among it
use super::arena::{Fresh, Layout};
use super::*;

//what the walk finds of a node to lay it out: its moves with a node, its moves, and whether it gets a slot a move
#[derive(Clone, Copy)]
struct Shape {
    links: u8,
    edges: u8,
    solid: bool,
}

//the least nodes a job of the walk or the copy takes, and the jobs a thread, for the threads that finish first to take others'
const JOB_NODES: usize = if cfg!(test) { 8 } else { 4096 };
const JOBS_A_THREAD: usize = 4;

//a compaction takes at most this share of its search's time, judged by its walk: the whole takes up to COMPACTION_PER_WALK times as long
const COMPACTION_SHARE: f64 = 0.1;
const COMPACTION_PER_WALK: f64 = 5.0;

//with transpositions the walk keeps of each old node its number in the copy, else the least position of a parent that claims it
//(CLAIMED set), else nothing: complemented, in an array that starts zeroed, so that fetch_max keeps the least
const CLAIMED: u32 = 1 << 31;

fn mark(slot: &AtomicU32, value: u32) {
    slot.fetch_max(!value, Ordering::Relaxed);
}

fn marked(slot: &AtomicU32) -> u32 {
    !slot.load(Ordering::Relaxed)
}

//a job's part of a level: each node's shape, and its moves' nodes in their order
struct Found {
    shapes: Vec<Shape>,
    children: Vec<u32>,
}

//what every stretch of the copy reads and writes
struct Plan<'a> {
    fresh: &'a Fresh,
    graph: Option<&'a SharedGraph>,
    order: &'a [u32],
    shapes: &'a [Shape],
    numbers: Option<&'a [AtomicU32]>,
}

impl SharedTree {
    //what the root reaches copied into new arenas in breadth first order on `threads` threads: compacted()'s tree, its nodes, edges,
    //links and slots numbered as there. None, the tree left as it was, when its walk says the whole would take more than
    //COMPACTION_SHARE of a search of `seconds`. For a tree no thread works in; the old arenas stay with it
    pub fn compact(&self, threads: usize, seconds: Option<f64>) -> Option<SharedTree> {
        assert_eq!(self.pending(), 0, "a tree to compact has nothing waiting");
        let began = std::time::Instant::now();
        let numbers = self.graph.as_ref().map(|_| zeroed(self.nodes.len()));
        let (order, shapes) = self.walk(threads, numbers.as_deref());
        if seconds.is_some_and(|seconds| began.elapsed().as_secs_f64() * COMPACTION_PER_WALK > COMPACTION_SHARE * seconds) {
            return None;
        }
        //the copy in stretches, each with where its room starts and (without transpositions) its first move's node
        let stretches = jobs(0..order.len(), threads);
        let (mut layout, mut children) = (Layout::new(), 1);
        let starts: Vec<(Layout, u32)> = stretches
            .iter()
            .map(|stretch| {
                let start = (layout, children);
                for shape in &shapes[stretch.clone()] {
                    layout.place(usize::from(shape.edges), usize::from(shape.links), shape.solid);
                    children += u32::from(shape.links);
                }
                start
            })
            .collect();
        let fresh = Fresh::new(order.len(), &layout);
        in_parallel(threads, fresh.chunks(), |chunk| fresh.make_chunk(chunk));
        let graph = self.graph.as_ref().map(|_| {
            let graph = SharedGraph::new();
            let room = order.len().div_ceil(TABLE_SHARDS) * 9 / 8;
            in_parallel(threads, TABLE_SHARDS, |shard| graph.shards[shard].lock().expect("a table shard").reserve(room));
            graph
        });
        let plan = Plan { fresh: &fresh, graph: graph.as_ref(), order: &order, shapes: &shapes, numbers: numbers.as_deref() };
        in_parallel(threads, stretches.len(), |job| self.copy_stretch(&plan, stretches[job].clone(), starts[job].0, starts[job].1));
        let root = fresh.nodes.get(0);
        root.add_reference();
        root.clear_proof();
        if let Some(graph) = &graph {
            let entries = graph.shards.iter().map(|shard| shard.lock().expect("a table shard").len()).sum();
            graph.entries.store(entries, Ordering::Relaxed);
        }
        let Fresh { nodes, edges, links } = fresh;
        Some(SharedTree {
            full: AtomicBool::new(self.is_full()),
            solid_threshold: self.solid_threshold,
            solid_batch: self.solid_batch,
            ..SharedTree::with_arenas(self.board.clone(), self.keys.clone(), 0, nodes, edges, links, graph)
        })
    }

    //the nodes the root reaches in breadth first order, a node that several moves lead to at the first, and their shapes; with
    //transpositions `numbers` gets each one's number. A level's children are claimed by all its jobs before any job looks
    fn walk(&self, threads: usize, numbers: Option<&[AtomicU32]>) -> (Vec<u32>, Vec<Shape>) {
        let (mut order, mut shapes) = (vec![self.root as u32], Vec::new());
        let mut level = 0..1;
        while !level.is_empty() {
            let parts = jobs(level.clone(), threads);
            let found = in_parallel(threads, parts.len(), |part| self.find_children(&order, parts[part].clone(), numbers));
            let kept =
                numbers.map(|numbers| in_parallel(threads, parts.len(), |part| first_claims(&found[part], parts[part].start, numbers)));
            for (part, found) in found.into_iter().enumerate() {
                shapes.extend(found.shapes);
                order.extend(kept.as_ref().map_or(&found.children, |kept| &kept[part]));
            }
            level = level.end..order.len();
        }
        (order, shapes)
    }

    //the shapes and children of the nodes at `positions` of the order; with transpositions each node takes its number, and claims its
    //children for its position
    fn find_children(&self, order: &[u32], positions: Range<usize>, numbers: Option<&[AtomicU32]>) -> Found {
        let mut found = Found { shapes: Vec::with_capacity(positions.len()), children: Vec::new() };
        for position in positions {
            let id = order[position] as usize;
            if let Some(numbers) = numbers {
                mark(&numbers[id], position as u32);
            }
            let node = self.nodes.get(id);
            let before = found.children.len();
            for (_, link) in self.links.linked(node) {
                let child = self.links.child(link);
                if let Some(numbers) = numbers {
                    mark(&numbers[child as usize], CLAIMED | position as u32);
                }
                found.children.push(child);
            }
            let edges = node.edges().len();
            let solid = edges > 0 && node.visits() >= self.solid_threshold;
            found.shapes.push(Shape { links: (found.children.len() - before) as u8, edges: edges as u8, solid });
        }
        found
    }

    //copies the nodes at `positions` of the order, their room from `layout` on and (without transpositions) their moves' nodes numbered
    //from `next_child`
    fn copy_stretch(&self, plan: &Plan, positions: Range<usize>, mut layout: Layout, mut next_child: u32) {
        for position in positions {
            let (old, shape) = (self.nodes.get(plan.order[position] as usize), plan.shapes[position]);
            let (first_edge, children) = layout.place(usize::from(shape.edges), usize::from(shape.links), shape.solid);
            plan.fresh.nodes.get(position).copy_of(old, first_edge, children);
            if shape.edges > 0 {
                plan.fresh.edges.copy_from(first_edge, &self.edges, old.edges());
            }
            for (index, (place, link)) in self.links.linked(old).enumerate() {
                let child = plan.numbers.map_or(next_child, |numbers| marked(&numbers[self.links.child(link) as usize]));
                next_child += 1;
                plan.fresh.nodes.get(child as usize).add_reference();
                let visits = self.links.visits(link).load(Ordering::Relaxed);
                if shape.solid {
                    plan.fresh.links.set_solid(children + place as u32, child, visits);
                } else {
                    let link = children + index as u32;
                    let next = if index + 1 < usize::from(shape.links) { link + 1 } else { NO_LINK };
                    plan.fresh.links.write(link, child, visits, place, next);
                }
            }
            if let Some(graph) = plan.graph {
                let entry = graph_key(old.key(), u32::from(old.halfmoves()));
                graph.shard(entry).lock().expect("a table shard").insert(entry, position as u32);
            }
        }
    }
}

//the moves' nodes of a job's nodes (from position `start`) that were first claimed there: those numbered next, in their order
fn first_claims(found: &Found, start: usize, numbers: &[AtomicU32]) -> Vec<u32> {
    let mut children = found.children.iter();
    let mut kept = Vec::new();
    for (index, shape) in found.shapes.iter().enumerate() {
        let claim = CLAIMED | (start + index) as u32;
        kept.extend(children.by_ref().take(usize::from(shape.links)).filter(|&&child| marked(&numbers[child as usize]) == claim));
    }
    kept
}

//`items` in jobs of at least JOB_NODES, JOBS_A_THREAD a thread when there are enough
fn jobs(items: Range<usize>, threads: usize) -> Vec<Range<usize>> {
    let size = items.len().div_ceil(threads.max(1) * JOBS_A_THREAD).max(JOB_NODES);
    items.clone().step_by(size).map(|start| start..(start + size).min(items.end)).collect()
}

//f of each job on up to `threads` threads, which take the jobs in turn; the results in the jobs' order
fn in_parallel<R: Send>(threads: usize, jobs: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    let workers = threads.min(jobs);
    if workers <= 1 {
        return (0..jobs).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let work = || {
        let mut done = Vec::new();
        loop {
            let job = next.fetch_add(1, Ordering::Relaxed);
            if job >= jobs {
                return done;
            }
            done.push((job, f(job)));
        }
    };
    let mut done = std::thread::scope(|scope| {
        let helpers: Vec<_> = (1..workers).map(|_| scope.spawn(&work)).collect();
        let mut done = work();
        for helper in helpers {
            done.extend(helper.join().expect("a thread of the compaction"));
        }
        done
    });
    done.sort_unstable_by_key(|&(job, _)| job);
    done.into_iter().map(|(_, result)| result).collect()
}

//zeroed atomics whose zero pages the system gives, unwritten
fn zeroed(len: usize) -> Vec<AtomicU32> {
    let mut zeros = std::mem::ManuallyDrop::new(vec![0u32; len]);
    // SAFETY: AtomicU32 has u32's size, alignment and bit validity, and the allocation passes whole to the new vector
    unsafe { Vec::from_raw_parts(zeros.as_mut_ptr().cast::<AtomicU32>(), zeros.len(), zeros.capacity()) }
}
