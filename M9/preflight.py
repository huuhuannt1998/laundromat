"""Resource preflight. Run before any heavy experiment; exits non-zero if unsafe.

The laptop crashed once with five projects running at the same time, so the queue
refuses to start rather than competing for RAM. Peak RSS for the 6.9B arms is one
model in fp16, about 14 GB, so this asks for headroom above that.
"""
import sys, shutil, subprocess, pathlib

NEED_RAM_GB  = 16.0     # default: one fp16 6.9B model (MPK's subprocess) + headroom.
                        # Override with --need N for smaller arms; BERT-scale work
                        # needs ~4 GB, not 16, and a blanket gate wrongly blocks it.
NEED_DISK_GB = 20.0

def free_ram_gb():
    """Available RAM in GB.

    An earlier version summed free+inactive+speculative pages from vm_stat and
    read 4.9 GB while macOS itself reported 39% free with zero swap in use --
    too conservative, because compressed and reclaimable pages were not counted.
    Prefer macOS's own figure and fall back to the vm_stat sum.
    """
    total = 0
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                   capture_output=True, text=True).stdout.strip())
    except Exception:
        pass
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
        for l in out.splitlines():
            if "free percentage" in l:
                pct = float(l.split(":")[1].strip().rstrip("%"))
                if total:
                    return total * pct / 100 / 1e9
    except Exception:
        pass
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        size = 16384
        for l in out.splitlines():
            if "page size of" in l:
                size = int(l.split("page size of")[1].split()[0])
        def pg(name):
            for l in out.splitlines():
                if l.startswith(name):
                    return int(l.split(":")[1].strip().rstrip("."))
            return 0
        return (pg("Pages free") + pg("Pages inactive") + pg("Pages speculative")) * size / 1e9
    except Exception:
        return float("nan")

def heavy_others():
    """Other python/torch jobs that would compete for RAM."""
    out = subprocess.run(["ps", "-Ao", "pid=,rss=,comm="], capture_output=True, text=True).stdout
    rows = []
    for l in out.splitlines():
        parts = l.split(None, 2)
        if len(parts) < 3: continue
        pid, rss, comm = parts[0], int(parts[1]), parts[2]
        if rss > 2_000_000 and ("python" in comm or "uv" in comm):   # >2 GB RSS
            rows.append((pid, rss/1e6, comm.strip()))
    return rows

def main():
    global NEED_RAM_GB
    if "--need" in sys.argv:
        NEED_RAM_GB = float(sys.argv[sys.argv.index("--need")+1])
    ram  = free_ram_gb()
    disk = shutil.disk_usage(pathlib.Path.home()).free / 1e9
    others = heavy_others()
    ok = True
    print(f"  free RAM   {ram:6.1f} GB   (need >= {NEED_RAM_GB})")
    print(f"  free disk  {disk:6.1f} GB   (need >= {NEED_DISK_GB})")
    if ram < NEED_RAM_GB:  print("  BLOCK: not enough free RAM -- close other projects first"); ok = False
    if disk < NEED_DISK_GB: print("  BLOCK: not enough free disk"); ok = False
    if others:
        print("  BLOCK: other heavy python/uv processes are running:")
        for pid, gb, comm in others: print(f"     pid {pid}  {gb:.1f} GB  {comm}")
        ok = False
    print("  PREFLIGHT OK" if ok else "  PREFLIGHT FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
