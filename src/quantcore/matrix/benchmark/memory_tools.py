import time
import tracemalloc
from threading import Thread


class MemoryPoller:
    """
    Example usage.

    with MemoryPoller() as mp:
        do some stuff here
        print('initial memory usage', mp.initial_memory)
        print('max memory usage', mp.max_memory)
        excess_memory_used = mp.max_memory - mp.initial_memory
    """

    def _poll_max_memory_usage(self):
        while not self.stop_polling:
            self.snapshots.append(tracemalloc.take_snapshot())
            time.sleep(1e-3)

    def __enter__(self):
        tracemalloc.start()
        self.stop_polling = False
        self.snapshots = [tracemalloc.take_snapshot()]
        self.t = Thread(target=self._poll_max_memory_usage)
        self.t.start()
        return self

    def __exit__(self, *excargs):
        self.stop_polling = True
        self.t.join()
        self.final_usage, self.peak_usage = tracemalloc.get_traced_memory()
        tracemalloc.stop()


def track_peak_mem(f, *args, **kwargs):
    """Track peak memory. Used in benchmarks to track memory used during matrix operations."""
    with MemoryPoller() as mp:
        f(*args, **kwargs)
    for s in mp.snapshots:
        top_stats = s.statistics("lineno")
        print("[ Top 4 ]")
        for stat in top_stats[:4]:
            print(stat)
    return mp.peak_usage
