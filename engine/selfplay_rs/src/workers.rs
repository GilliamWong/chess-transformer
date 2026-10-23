//threads that run a job with the caller (worker 0) until all are done, so it may borrow; idle ones spin, then sleep
use crate::affinity;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

type Job = &'static (dyn Fn(usize) + Sync);

struct Shared {
    job: AtomicPtr<Job>, //the caller's, while it runs
    sleep: Mutex<()>,
    wake: Condvar,
    generation: AtomicU64, //the jobs started, changed under sleep's lock so that a worker going to sleep sees it
    running: AtomicUsize,  //workers still on the current job
    panicked: AtomicBool,
    stop: AtomicBool,
    spin: Duration,
}

pub struct Workers {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    //count workers in all, the caller one of them; cpus: none, or one for each of the others to be pinned to
    pub fn new(count: usize, spin: Duration, cpus: &[usize]) -> Workers {
        assert!(count >= 1, "at least the caller");
        assert!(cpus.is_empty() || cpus.len() == count - 1, "a cpu for each worker but the caller");
        let shared = Arc::new(Shared {
            job: AtomicPtr::new(std::ptr::null_mut()),
            sleep: Mutex::new(()),
            wake: Condvar::new(),
            generation: AtomicU64::new(0),
            running: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            spin,
        });
        let threads = (1..count)
            .map(|worker| {
                let shared = Arc::clone(&shared);
                let cpu = cpus.get(worker - 1).copied();
                std::thread::Builder::new()
                    .name(format!("gather worker {worker}"))
                    .spawn(move || {
                        //pinning only moves the thread, so a cpu the os refuses leaves it where it is
                        if let Some(cpu) = cpu {
                            affinity::pin_current(cpu);
                        }
                        work(&shared, worker)
                    })
                    .expect("a worker thread")
            })
            .collect();
        Workers { shared, threads }
    }

    pub fn count(&self) -> usize {
        self.threads.len() + 1
    }

    pub fn run(&self, job: &(dyn Fn(usize) + Sync)) {
        if self.threads.is_empty() {
            job(0);
            return;
        }
        //SAFETY: the workers call the job only until running is back to 0, which run waits for before it returns
        let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), Job>(job) };
        let shared = &*self.shared;
        shared.running.store(self.threads.len(), Ordering::Relaxed);
        shared.job.store(&job as *const Job as *mut Job, Ordering::Relaxed);
        {
            let _sleep = shared.sleep.lock().expect("the workers' sleep");
            shared.generation.fetch_add(1, Ordering::Release);
        }
        shared.wake.notify_all();
        let caller = catch_unwind(AssertUnwindSafe(|| job(0)));
        let mut spins = 0u32;
        while shared.running.load(Ordering::Acquire) > 0 {
            spins += 1;
            if spins < 1 << 16 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        shared.job.store(std::ptr::null_mut(), Ordering::Relaxed);
        if let Err(panic) = caller {
            resume_unwind(panic);
        }
        assert!(!shared.panicked.swap(false, Ordering::Relaxed), "a gather worker panicked");
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        {
            let _sleep = self.shared.sleep.lock().expect("the workers' sleep");
            self.shared.stop.store(true, Ordering::Relaxed);
        }
        self.shared.wake.notify_all();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

fn work(shared: &Shared, worker: usize) {
    let mut seen = 0;
    loop {
        let waiting = Instant::now();
        let mut spins = 0u32;
        while shared.generation.load(Ordering::Acquire) == seen && !shared.stop.load(Ordering::Relaxed) {
            spins += 1;
            if spins % 64 == 0 && waiting.elapsed() >= shared.spin {
                break;
            }
            std::hint::spin_loop();
        }
        if shared.generation.load(Ordering::Acquire) == seen {
            let mut sleep = shared.sleep.lock().expect("the workers' sleep");
            while shared.generation.load(Ordering::Acquire) == seen && !shared.stop.load(Ordering::Relaxed) {
                sleep = shared.wake.wait(sleep).expect("the workers' sleep");
            }
        }
        if shared.stop.load(Ordering::Relaxed) {
            return;
        }
        seen = shared.generation.load(Ordering::Acquire);
        //SAFETY: the caller's job stays in place until this worker is done with it (run)
        let job = unsafe { *shared.job.load(Ordering::Relaxed) };
        if catch_unwind(AssertUnwindSafe(|| job(worker))).is_err() {
            shared.panicked.store(true, Ordering::Relaxed);
        }
        shared.running.fetch_sub(1, Ordering::Release);
    }
}
