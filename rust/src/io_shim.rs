// Absolute-path I/O with a test-only root override.
//
// Production `resolve` uses paths verbatim: the HAL only reads `/proc` and
// `/sys`, in parallel across the binder thread pool, so the I/O path must not
// take a global lock. Tests compile a second `resolve` that prepends a fixture
// root so the real code paths run unmodified against a fake `/proc` tree.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Mutex;

/// A reply parcel is capped at ~1 MiB by libbinder. Several of the nodes this
/// HAL stands in for are far larger than the OPlus ones they replace
/// (/proc/vmallocinfo alone is 1.15 MB on this kernel), and a reply that size
/// fails the transaction outright with EX_TRANSACTION_FAILED (-129). Cap the
/// read so the call still succeeds and returns a truncated but usable dump.
pub const MAX_READ: usize = 256 * 1024;

#[cfg(test)]
static TEST_ROOT: Mutex<Option<String>> = Mutex::new(None);

#[cfg(test)]
static TESTS: Mutex<()> = Mutex::new(());

/// A panicking test poisons its guard, and the fixture root is process-global,
/// so without recovery one failure would cascade into every later test. The
/// lock only serialises tests and protects no invariant, so recovering the
/// guard is correct.
#[cfg(test)]
fn lock<T>(m: &'static Mutex<T>) -> std::sync::MutexGuard<'static, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
pub fn set_test_root(root: &str) {
    *lock(&TEST_ROOT) = Some(root.to_string());
}

#[cfg(test)]
pub fn clear_test_root() {
    *lock(&TEST_ROOT) = None;
}

#[cfg(test)]
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    lock(&TESTS)
}

/// Production: absolute paths are used verbatim. No lock, no allocation
/// beyond the `PathBuf` the caller needs for the syscall.
#[cfg(not(test))]
pub fn resolve(path: &Path) -> PathBuf {
    path.to_path_buf()
}

/// Test only: prepends the fixture root when set. Idempotent: a path already
/// under the root is returned unchanged, because callers legitimately hand a
/// resolved path (from a directory listing or a symlink target) back to the
/// read helpers.
#[cfg(test)]
pub fn resolve(path: &Path) -> PathBuf {
    if path.is_absolute() {
        let root = lock(&TEST_ROOT).clone();
        if let Some(root) = root {
            if path.starts_with(&root) {
                return path.to_path_buf();
            }
            let stripped = path
                .strip_prefix("/")
                .expect("absolute path strips cleanly");
            return Path::new(&root).join(stripped);
        }
    }
    path.to_path_buf()
}

/// Read a node, capped at `MAX_READ`. A missing or unreadable node is an
/// empty string, never an error: every caller in this HAL treats "no data"
/// and "no node" the same way.
pub fn read(path: &str) -> String {
    let full = resolve(Path::new(path));
    match fs::File::open(&full) {
        Ok(file) => {
            use std::io::Read;
            let mut buf = Vec::with_capacity(4096);
            let _ = file.take(MAX_READ as u64).read_to_end(&mut buf);
            match String::from_utf8(buf) {
                Ok(text) => text,
                Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
            }
        }
        Err(_) => String::new(),
    }
}

/// Write a node. Returns whether the write was accepted.
pub fn write(path: &str, value: &str) -> bool {
    let full = resolve(Path::new(path));
    match fs::OpenOptions::new().write(true).open(&full) {
        Ok(mut file) => file.write_all(value.as_bytes()).is_ok(),
        Err(_) => false,
    }
}

pub fn exists(path: &str) -> bool {
    resolve(Path::new(path)).exists()
}

pub fn is_dir(path: &str) -> bool {
    resolve(Path::new(path)).is_dir()
}

/// Directory entries, dotfiles skipped, newline separated. Matches the
/// `readdir` loop in the C++ backend, which dropped any name starting with '.'.
pub fn list_dir(path: &str) -> String {
    let full = resolve(Path::new(path));
    let entries = match fs::read_dir(&full) {
        Ok(e) => e,
        Err(_) => return String::new(),
    };
    let mut names: Vec<String> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        names.push(name);
    }
    names.join("\n")
}

/// Numeric subdirectory names, sorted. Used to walk `/proc/<pid>`.
pub fn numeric_dirs(path: &str) -> Vec<String> {
    let full = resolve(Path::new(path));
    let mut out: Vec<String> = Vec::new();
    if let Ok(entries) = fs::read_dir(&full) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.chars().next().map_or(true, |c| !c.is_ascii_digit()) {
                continue;
            }
            out.push(name);
        }
    }
    out.sort();
    out
}

/// Page size in KiB, for turning statm's page counts into kB.
pub fn page_kib() -> i64 {
    // 4 KiB on every arm64 kernel Android ships; asked of the OS anyway so a
    // 16 KiB-page kernel does not silently report a quarter of the memory.
    match sysconf_page_size() {
        Some(bytes) if bytes > 0 => (bytes / 1024) as i64,
        _ => 4,
    }
}

fn sysconf_page_size() -> Option<u64> {
    extern "C" {
        fn sysconf(name: i32) -> i64;
    }
    const SC_PAGESIZE: i32 = 39;
    // SAFETY: sysconf with a constant, side-effect-free query.
    let v = unsafe { sysconf(SC_PAGESIZE) };
    if v > 0 {
        Some(v as u64)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_is_idempotent() {
        let _g = test_lock();
        let root = "/tmp/perf_shim_idem";
        let _ = fs::remove_dir_all(root);
        set_test_root(root);
        let once = resolve(Path::new("/proc/version"));
        let twice = resolve(&once);
        assert_eq!(once, twice);
        clear_test_root();
    }

    #[test]
    fn read_returns_empty_for_a_missing_node() {
        let _g = test_lock();
        let root = "/tmp/perf_shim_missing";
        let _ = fs::remove_dir_all(root);
        fs::create_dir_all(root).unwrap();
        set_test_root(root);
        assert_eq!(read("/proc/does_not_exist"), "");
        clear_test_root();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn read_caps_oversized_nodes() {
        let _g = test_lock();
        let root = "/tmp/perf_shim_cap";
        let _ = fs::remove_dir_all(root);
        fs::create_dir_all(format!("{}/proc", root)).unwrap();
        set_test_root(root);
        let big = "x".repeat(MAX_READ + 4096);
        fs::write(format!("{}/proc/big", root), &big).unwrap();
        assert_eq!(read("/proc/big").len(), MAX_READ);
        clear_test_root();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn list_dir_skips_dotfiles() {
        let _g = test_lock();
        let root = "/tmp/perf_shim_listing";
        let _ = fs::remove_dir_all(root);
        fs::create_dir_all(format!("{}/sys/fs", root)).unwrap();
        set_test_root(root);
        fs::write(format!("{}/sys/fs/visible", root), "1").unwrap();
        fs::write(format!("{}/sys/fs/.hidden", root), "1").unwrap();
        let listing = list_dir("/sys/fs");
        assert!(listing.contains("visible"));
        assert!(!listing.contains("hidden"));
        clear_test_root();
        let _ = fs::remove_dir_all(root);
    }
}