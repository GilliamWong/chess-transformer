//the logical cpus as the os numbers them (core_affinity's ids), grouped as the hardware shares them: by L3 cache (a ccd
//on zen) and by physical core (smt siblings); and a thread pinned to one of them
use core_affinity::CoreId;
use std::path::Path;
use std::sync::OnceLock;

pub struct Topology {
    pub l3s: Vec<(u64, Vec<usize>)>, //each L3's bytes and cpus
    pub cores: Vec<Vec<usize>>,      //each physical core's cpus
}

//the os's L3s (bytes, cpus) and physical cores as it lists them, cpus the process may not run on among them
type Groups = (Vec<(u64, Vec<usize>)>, Vec<Vec<usize>>);

//the cpus the process may run on, read once before any pin: on linux they are the calling thread's own
pub fn allowed() -> &'static [usize] {
    static ALLOWED: OnceLock<Vec<usize>> = OnceLock::new();
    ALLOWED.get_or_init(|| core_affinity::get_core_ids().map(|ids| ids.into_iter().map(|id| id.id).collect()).unwrap_or_default())
}

//pins the calling thread to the cpu: false if it is not one of allowed()'s or the os refuses
pub fn pin_current(cpu: usize) -> bool {
    allowed().contains(&cpu) && core_affinity::set_for_current(CoreId { id: cpu })
}

//the groups of allowed()'s cpus, each by its lowest cpu; None where the os does not say
pub fn topology() -> Option<Topology> {
    grouped(os::groups()?, allowed())
}

fn grouped((l3s, cores): Groups, allowed: &[usize]) -> Option<Topology> {
    let kept = |cpus: Vec<usize>| -> Vec<usize> {
        let mut kept: Vec<usize> = cpus.into_iter().filter(|cpu| allowed.contains(cpu)).collect();
        kept.sort_unstable();
        kept
    };
    let mut l3s: Vec<(u64, Vec<usize>)> =
        l3s.into_iter().map(|(bytes, cpus)| (bytes, kept(cpus))).filter(|(_, cpus)| !cpus.is_empty()).collect();
    let mut cores: Vec<Vec<usize>> = cores.into_iter().map(kept).filter(|cpus| !cpus.is_empty()).collect();
    l3s.sort_by_key(|(_, cpus)| cpus[0]);
    l3s.dedup_by(|one, other| one.1 == other.1);
    cores.sort_by_key(|cpus| cpus[0]);
    cores.dedup();
    (!l3s.is_empty() && !cores.is_empty()).then_some(Topology { l3s, cores })
}

//sysfs's list of cpus, as "0-3,8,10-11"
pub fn cpu_list(text: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in text.trim().split(',').filter(|part| !part.is_empty()) {
        let (first, last) = part.split_once('-').unwrap_or((part, part));
        cpus.extend(first.parse::<usize>().ok()?..=last.parse::<usize>().ok()?);
    }
    Some(cpus)
}

//sysfs's size of a cache, as "32768K"
pub fn cache_bytes(text: &str) -> Option<u64> {
    let text = text.trim();
    let (number, unit) = text.find(|c: char| !c.is_ascii_digit()).map_or((text, ""), |at| text.split_at(at));
    let scale = match unit {
        "" => 1,
        "K" => 1 << 10,
        "M" => 1 << 20,
        "G" => 1 << 30,
        _ => return None,
    };
    Some(number.parse::<u64>().ok()? * scale)
}

//the groups sysfs's cpu folder (linux's /sys/devices/system/cpu) gives these cpus: their cores' smt siblings
//(topology/thread_siblings_list) and their L3s (the cache/index* of level 3)
pub fn sysfs_groups(folder: &Path, cpus: &[usize]) -> Option<Groups> {
    let read = |path: &Path| std::fs::read_to_string(path).ok();
    let (mut l3s, mut cores) = (Vec::new(), Vec::new());
    for &cpu in cpus {
        let folder = folder.join(format!("cpu{cpu}"));
        cores.push(cpu_list(&read(&folder.join("topology/thread_siblings_list"))?)?);
        for index in 0.. {
            let cache = folder.join(format!("cache/index{index}"));
            let Some(level) = read(&cache.join("level")) else { break };
            if level.trim() == "3" {
                l3s.push((cache_bytes(&read(&cache.join("size"))?)?, cpu_list(&read(&cache.join("shared_cpu_list"))?)?));
            }
        }
    }
    Some((l3s, cores))
}

#[cfg(windows)]
mod os {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetLogicalProcessorInformationEx(relationship: u32, buffer: *mut u8, length: *mut u32) -> i32;
    }

    const RELATION_PROCESSOR_CORE: u32 = 0;
    const RELATION_CACHE: u32 = 2;

    //each SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX's body, after its relationship and size
    fn records(relationship: u32) -> Option<Vec<Vec<u8>>> {
        let mut length = 0u32;
        //SAFETY: a null buffer only asks for the length
        unsafe { GetLogicalProcessorInformationEx(relationship, std::ptr::null_mut(), &mut length) };
        let mut buffer = vec![0u8; length as usize];
        //SAFETY: the buffer has the length the call asked for
        if unsafe { GetLogicalProcessorInformationEx(relationship, buffer.as_mut_ptr(), &mut length) } == 0 {
            return None;
        }
        let (mut records, mut offset) = (Vec::new(), 0);
        while offset < length as usize {
            let size = u32::from_le_bytes(buffer.get(offset + 4..offset + 8)?.try_into().ok()?) as usize;
            if size <= 8 {
                return None;
            }
            records.push(buffer.get(offset + 8..offset + size)?.to_vec());
            offset += size;
        }
        Some(records)
    }

    //the cpus of the GROUP_AFFINITY at offset: those of processor group 0, the only one core_affinity pins in
    fn group_cpus(body: &[u8], offset: usize) -> Option<Vec<usize>> {
        let mask = u64::from_le_bytes(body.get(offset..offset + 8)?.try_into().ok()?);
        let group = u16::from_le_bytes(body.get(offset + 8..offset + 10)?.try_into().ok()?);
        Some(if group == 0 { (0..64).filter(|cpu| mask >> cpu & 1 == 1).collect() } else { Vec::new() })
    }

    pub fn groups() -> Option<super::Groups> {
        //PROCESSOR_RELATIONSHIP: Flags, EfficiencyClass, Reserved[20], GroupCount, then its GROUP_AFFINITY at 24
        let cores = records(RELATION_PROCESSOR_CORE)?.iter().map(|body| group_cpus(body, 24)).collect::<Option<_>>()?;
        //CACHE_RELATIONSHIP: Level, Associativity, LineSize, CacheSize (u32 at 4), Type, Reserved[18], GroupCount, then
        //its GROUP_AFFINITY at 32
        let l3s = records(RELATION_CACHE)?
            .iter()
            .filter(|body| body.first() == Some(&3))
            .map(|body| Some((u64::from(u32::from_le_bytes(body.get(4..8)?.try_into().ok()?)), group_cpus(body, 32)?)))
            .collect::<Option<_>>()?;
        Some((l3s, cores))
    }
}

#[cfg(target_os = "linux")]
mod os {
    pub fn groups() -> Option<super::Groups> {
        super::sysfs_groups(std::path::Path::new("/sys/devices/system/cpu"), super::allowed())
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod os {
    pub fn groups() -> Option<super::Groups> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workers::Workers;
    use std::sync::Mutex;
    use std::time::Duration;

    //the cpu the calling thread runs on now
    #[cfg(windows)]
    fn current_cpu() -> usize {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcessorNumber() -> u32;
        }
        //SAFETY: no arguments, no state
        unsafe { GetCurrentProcessorNumber() as usize }
    }

    #[cfg(target_os = "linux")]
    fn current_cpu() -> usize {
        unsafe extern "C" {
            fn sched_getcpu() -> i32;
        }
        //SAFETY: no arguments, no state
        unsafe { sched_getcpu() as usize }
    }

    #[test]
    fn sysfs_lists_and_sizes_parse() {
        assert_eq!(cpu_list("0-3,8,10-11\n"), Some(vec![0, 1, 2, 3, 8, 10, 11]));
        assert_eq!(cpu_list("5"), Some(vec![5]));
        assert_eq!(cpu_list("1-x"), None);
        assert_eq!(cache_bytes("32768K\n"), Some(32 << 20));
        assert_eq!(cache_bytes("96M"), Some(96 << 20));
        assert_eq!(cache_bytes("12Q"), None);
    }

    //a cpu's folder of a fake sysfs: its smt siblings, its L1s and L2 with them, and its L3 (size, shared_cpu_list)
    fn write_cpu(root: &Path, cpu: usize, siblings: &str, (l3_size, l3_cpus): (&str, &str)) {
        let folder = root.join(format!("cpu{cpu}"));
        std::fs::create_dir_all(folder.join("topology")).expect("the fake sysfs");
        std::fs::write(folder.join("topology/thread_siblings_list"), format!("{siblings}\n")).expect("the fake sysfs");
        let caches = [(0, 1, "48K", siblings), (1, 1, "32K", siblings), (2, 2, "1024K", siblings), (3, 3, l3_size, l3_cpus)];
        for (index, level, size, shared) in caches {
            let cache = folder.join(format!("cache/index{index}"));
            std::fs::create_dir_all(&cache).expect("the fake sysfs");
            for (name, text) in [("level", level.to_string()), ("size", size.to_string()), ("shared_cpu_list", shared.to_string())] {
                std::fs::write(cache.join(name), format!("{text}\n")).expect("the fake sysfs");
            }
        }
    }

    //sysfs as linux gives a 2 x EPYC 9175F (a core's siblings k and k + 32, one 32 MB L3 a core) to a process held to
    //socket 0 (numactl --cpunodebind=0) and to one on both sockets, and this 9950X3D (siblings k and k + 16, ccds 0-7
    //and 8-15); without the caches' folders there is no topology
    #[test]
    fn sysfs_gives_the_topology() {
        let root = std::env::temp_dir().join(format!("selfplay_rs_sysfs_{}", std::process::id()));
        let epyc = root.join("epyc");
        for cpu in 0..64 {
            let pair = format!("{},{}", cpu % 32, cpu % 32 + 32);
            write_cpu(&epyc, cpu, &pair, ("32768K", &pair));
        }
        let socket: Vec<usize> = (0..16).chain(32..48).collect();
        let topology = grouped(sysfs_groups(&epyc, &socket).expect("the fake sysfs"), &socket).expect("a topology");
        assert_eq!(topology.cores, (0..16).map(|core| vec![core, core + 32]).collect::<Vec<_>>());
        assert_eq!(topology.l3s, (0..16).map(|core| (32 << 20, vec![core, core + 32])).collect::<Vec<_>>());
        let machine: Vec<usize> = (0..64).collect();
        let topology = grouped(sysfs_groups(&epyc, &machine).expect("the fake sysfs"), &machine).expect("a topology");
        assert_eq!(topology.cores, (0..32).map(|core| vec![core, core + 32]).collect::<Vec<_>>());
        assert_eq!(topology.l3s, (0..32).map(|core| (32 << 20, vec![core, core + 32])).collect::<Vec<_>>());
        let zen = root.join("zen");
        for cpu in 0..32 {
            let ccd = if cpu % 16 < 8 { ("98304K", "0-7,16-23") } else { ("32768K", "8-15,24-31") };
            write_cpu(&zen, cpu, &format!("{},{}", cpu % 16, cpu % 16 + 16), ccd);
        }
        let all: Vec<usize> = (0..32).collect();
        let topology = grouped(sysfs_groups(&zen, &all).expect("the fake sysfs"), &all).expect("a topology");
        let ccd = |halves: [std::ops::Range<usize>; 2]| halves.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(topology.l3s, vec![(96 << 20, ccd([0..8, 16..24])), (32 << 20, ccd([8..16, 24..32]))]);
        assert_eq!(topology.cores, (0..16).map(|core| vec![core, core + 16]).collect::<Vec<_>>());
        for cpu in 0..32 {
            std::fs::remove_dir_all(zen.join(format!("cpu{cpu}/cache"))).expect("the fake sysfs");
        }
        assert!(sysfs_groups(&zen, &all).and_then(|groups| grouped(groups, &all)).is_none());
        std::fs::remove_dir_all(&root).expect("the fake sysfs");
    }

    //every cpu the process may run on is in one physical core and one L3, and a core's cpus share its L3
    #[test]
    fn the_topology_groups_every_allowed_cpu_once() {
        let topology = topology().expect("windows and linux give their topology");
        println!("L3s {:?}, cores {:?}", topology.l3s, topology.cores);
        let mut grouped: Vec<usize> = topology.cores.concat();
        grouped.sort_unstable();
        assert_eq!(grouped, allowed());
        let mut cached: Vec<usize> = topology.l3s.iter().flat_map(|(_, cpus)| cpus.clone()).collect();
        cached.sort_unstable();
        assert_eq!(cached, allowed());
        assert!(topology.l3s.iter().all(|&(bytes, _)| bytes >= 1 << 20), "{:?}", topology.l3s);
        for core in &topology.cores {
            assert!(topology.l3s.iter().any(|(_, cpus)| core.iter().all(|cpu| cpus.contains(cpu))), "core {core:?}");
        }
    }

    //a pinned thread runs only on its cpu, and one the process may not run on is refused
    #[test]
    fn a_pinned_thread_runs_on_its_cpu() {
        let cpus = allowed().to_vec();
        std::thread::spawn(move || {
            for &cpu in cpus.iter().rev().take(3) {
                assert!(pin_current(cpu));
                for _ in 0..1_000 {
                    assert_eq!(current_cpu(), cpu);
                    std::hint::spin_loop();
                }
            }
            assert!(!pin_current(usize::MAX));
        })
        .join()
        .expect("the pinned thread");
    }

    //each worker but the caller runs its jobs on the cpu it was given
    #[test]
    fn pinned_workers_run_on_their_cpus() {
        let cpus: Vec<usize> = allowed().iter().rev().take(3).copied().collect();
        if cpus.len() < 3 {
            return;
        }
        let workers = Workers::new(4, Duration::from_micros(50), &cpus);
        let seen: Vec<Mutex<Vec<usize>>> = (0..4).map(|_| Mutex::new(Vec::new())).collect();
        for _ in 0..200 {
            workers.run(&|worker| seen[worker].lock().expect("a worker's cpus").push(current_cpu()));
        }
        for (worker, &cpu) in cpus.iter().enumerate() {
            let seen = seen[worker + 1].lock().expect("a worker's cpus");
            assert!(seen.len() == 200 && seen.iter().all(|&on| on == cpu), "worker {}: {seen:?} not {cpu}", worker + 1);
        }
    }
}
