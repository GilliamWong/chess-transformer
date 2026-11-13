use core_affinity::CoreId;
use std::sync::OnceLock;

pub struct Topology {
    pub l3s: Vec<(u64, Vec<usize>)>, //(bytes, cpus) of each L3
    pub cores: Vec<Vec<usize>>,
}

//read once before any pin: on linux they are the calling thread's own cpus
pub fn allowed() -> &'static [usize] {
    static ALLOWED: OnceLock<Vec<usize>> = OnceLock::new();
    ALLOWED.get_or_init(|| core_affinity::get_core_ids().map(|ids| ids.into_iter().map(|id| id.id).collect()).unwrap_or_default())
}

pub fn pin_current(cpu: usize) -> bool {
    allowed().contains(&cpu) && core_affinity::set_for_current(CoreId { id: cpu })
}

pub fn topology() -> Option<Topology> {
    grouped(os::groups()?, allowed())
}

//each socket's cpus among those this process may run on, in the sockets' order; None where the os does not say
pub fn sockets() -> Option<Vec<Vec<usize>>> {
    os::sockets()
}

fn grouped(Topology { l3s, cores }: Topology, allowed: &[usize]) -> Option<Topology> {
    let kept = |mut cpus: Vec<usize>| {
        cpus.retain(|cpu| allowed.contains(cpu));
        cpus.sort_unstable();
        cpus
    };
    let mut l3s: Vec<(u64, Vec<usize>)> =
        l3s.into_iter().map(|(bytes, cpus)| (bytes, kept(cpus))).filter(|(_, cpus)| !cpus.is_empty()).collect();
    let mut cores: Vec<Vec<usize>> = cores.into_iter().map(kept).filter(|cpus| !cpus.is_empty()).collect();
    //sysfs repeats a group for each of its cpus
    l3s.sort_by_key(|(_, cpus)| cpus[0]);
    l3s.dedup_by(|one, other| one.1 == other.1);
    cores.sort_by_key(|cpus| cpus[0]);
    cores.dedup();
    (!l3s.is_empty() && !cores.is_empty()).then_some(Topology { l3s, cores })
}

#[cfg(any(target_os = "linux", test))]
mod sysfs {
    use super::Topology;
    use std::path::Path;

    pub fn cpu_list(text: &str) -> Option<Vec<usize>> {
        let mut cpus = Vec::new();
        for part in text.trim().split(',').filter(|part| !part.is_empty()) {
            let (first, last) = part.split_once('-').unwrap_or((part, part));
            cpus.extend(first.parse::<usize>().ok()?..=last.parse::<usize>().ok()?);
        }
        Some(cpus)
    }

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

    pub fn groups(folder: &Path, cpus: &[usize]) -> Option<Topology> {
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
        Some(Topology { l3s, cores })
    }

    pub fn sockets(folder: &Path, cpus: &[usize]) -> Option<Vec<Vec<usize>>> {
        let mut sockets: Vec<(usize, Vec<usize>)> = Vec::new();
        for &cpu in cpus {
            let package = std::fs::read_to_string(folder.join(format!("cpu{cpu}/topology/physical_package_id"))).ok()?;
            let package = package.trim().parse::<usize>().ok()?;
            match sockets.iter_mut().find(|(id, _)| *id == package) {
                Some((_, cpus)) => cpus.push(cpu),
                None => sockets.push((package, vec![cpu])),
            }
        }
        sockets.sort_by_key(|&(id, _)| id);
        Some(sockets.into_iter().map(|(_, cpus)| cpus).collect())
    }
}

#[cfg(windows)]
mod os {
    pub fn sockets() -> Option<Vec<Vec<usize>>> {
        None
    }

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

    //the GROUP_AFFINITY at offset, empty outside processor group 0, the only one core_affinity pins in
    fn group_cpus(body: &[u8], offset: usize) -> Option<Vec<usize>> {
        let mask = u64::from_le_bytes(body.get(offset..offset + 8)?.try_into().ok()?);
        let group = u16::from_le_bytes(body.get(offset + 8..offset + 10)?.try_into().ok()?);
        Some(if group == 0 { (0..64).filter(|cpu| mask >> cpu & 1 == 1).collect() } else { Vec::new() })
    }

    pub fn groups() -> Option<super::Topology> {
        //a PROCESSOR_RELATIONSHIP's GROUP_AFFINITY is at 24
        let cores = records(RELATION_PROCESSOR_CORE)?.iter().map(|body| group_cpus(body, 24)).collect::<Option<_>>()?;
        //a CACHE_RELATIONSHIP has its Level at 0, CacheSize (u32) at 4 and GROUP_AFFINITY at 32
        let l3s = records(RELATION_CACHE)?
            .iter()
            .filter(|body| body.first() == Some(&3))
            .map(|body| Some((u64::from(u32::from_le_bytes(body.get(4..8)?.try_into().ok()?)), group_cpus(body, 32)?)))
            .collect::<Option<_>>()?;
        Some(super::Topology { l3s, cores })
    }
}

#[cfg(target_os = "linux")]
mod os {
    pub fn groups() -> Option<super::Topology> {
        super::sysfs::groups(std::path::Path::new("/sys/devices/system/cpu"), super::allowed())
    }

    pub fn sockets() -> Option<Vec<Vec<usize>>> {
        super::sysfs::sockets(std::path::Path::new("/sys/devices/system/cpu"), super::allowed())
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod os {
    pub fn groups() -> Option<super::Topology> {
        None
    }

    pub fn sockets() -> Option<Vec<Vec<usize>>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::sysfs::{cache_bytes, cpu_list};
    use super::*;
    use std::path::Path;

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

    fn write_cpu(root: &Path, cpu: usize, siblings: &str, (l3_size, l3_cpus): (&str, &str)) {
        let folder = root.join(format!("cpu{cpu}"));
        std::fs::create_dir_all(folder.join("topology")).expect("the fake sysfs");
        std::fs::write(folder.join("topology/thread_siblings_list"), format!("{siblings}\n")).expect("the fake sysfs");
        let caches = [("1", "48K", siblings), ("1", "32K", siblings), ("2", "1024K", siblings), ("3", l3_size, l3_cpus)];
        for (index, (level, size, shared)) in caches.into_iter().enumerate() {
            let cache = folder.join(format!("cache/index{index}"));
            std::fs::create_dir_all(&cache).expect("the fake sysfs");
            for (name, text) in [("level", level), ("size", size), ("shared_cpu_list", shared)] {
                std::fs::write(cache.join(name), format!("{text}\n")).expect("the fake sysfs");
            }
        }
    }

    #[test]
    fn sysfs_gives_the_sockets() {
        let root = std::env::temp_dir().join(format!("selfplay_rs_sockets_{}", std::process::id()));
        for cpu in 0..8 {
            let folder = root.join(format!("cpu{cpu}/topology"));
            std::fs::create_dir_all(&folder).expect("the fake sysfs");
            //numbered as some machines do: the sockets' cpus in turn
            std::fs::write(folder.join("physical_package_id"), format!("{}\n", cpu % 2)).expect("the fake sysfs");
        }
        let all: Vec<usize> = (0..8).collect();
        assert_eq!(sysfs::sockets(&root, &all), Some(vec![vec![0, 2, 4, 6], vec![1, 3, 5, 7]]));
        assert_eq!(sysfs::sockets(&root, &[3, 4, 5]), Some(vec![vec![4], vec![3, 5]]));
        assert_eq!(sysfs::sockets(&root, &[7, 8]), None);
        std::fs::remove_dir_all(&root).expect("the fake sysfs");
    }

    #[test]
    fn sysfs_gives_the_topology() {
        let root = std::env::temp_dir().join(format!("selfplay_rs_sysfs_{}", std::process::id()));
        let epyc = root.join("epyc");
        for cpu in 0..64 {
            let pair = format!("{},{}", cpu % 32, cpu % 32 + 32);
            write_cpu(&epyc, cpu, &pair, ("32768K", &pair));
        }
        let socket: Vec<usize> = (0..16).chain(32..48).collect();
        let machine: Vec<usize> = (0..64).collect();
        for (cpus, cores) in [(socket, 16), (machine, 32)] {
            let topology = grouped(sysfs::groups(&epyc, &cpus).expect("the fake sysfs"), &cpus).expect("a topology");
            let pairs: Vec<Vec<usize>> = (0..cores).map(|core| vec![core, core + 32]).collect();
            assert_eq!(topology.cores, pairs);
            assert_eq!(topology.l3s, pairs.into_iter().map(|pair| (32 << 20, pair)).collect::<Vec<_>>());
        }
        let zen = root.join("zen");
        for cpu in 0..32 {
            let ccd = if cpu % 16 < 8 { ("98304K", "0-7,16-23") } else { ("32768K", "8-15,24-31") };
            write_cpu(&zen, cpu, &format!("{},{}", cpu % 16, cpu % 16 + 16), ccd);
        }
        let all: Vec<usize> = (0..32).collect();
        let topology = grouped(sysfs::groups(&zen, &all).expect("the fake sysfs"), &all).expect("a topology");
        let ccd = |halves: [std::ops::Range<usize>; 2]| halves.into_iter().flatten().collect::<Vec<_>>();
        assert_eq!(topology.l3s, vec![(96 << 20, ccd([0..8, 16..24])), (32 << 20, ccd([8..16, 24..32]))]);
        assert_eq!(topology.cores, (0..16).map(|core| vec![core, core + 16]).collect::<Vec<_>>());
        for cpu in 0..32 {
            std::fs::remove_dir_all(zen.join(format!("cpu{cpu}/cache"))).expect("the fake sysfs");
        }
        assert!(sysfs::groups(&zen, &all).and_then(|groups| grouped(groups, &all)).is_none());
        std::fs::remove_dir_all(&root).expect("the fake sysfs");
    }

    #[test]
    fn the_topology_groups_every_allowed_cpu_once() {
        let topology = topology().expect("windows and linux give their topology");
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
}
