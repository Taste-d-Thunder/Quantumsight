import time
import psutil
import torch


class ResourceTracker:
    """Monitors runtime elapsed duration, CPU percentage, RAM consumption, and GPU metrics."""

    def __init__(self):
        self.start_time = None
        self.start_cpu = None
        self._has_gputil = False
        self._gpus = []

        if torch.cuda.is_available():
            try:
                import GPUtil
                self._gpus = GPUtil.getGPUs()
                self._has_gputil = True
            except ImportError:
                self._has_gputil = False

    def start(self):
        """Mark the start timestamp and CPU counter."""
        self.start_time = time.time()
        self.start_cpu = psutil.cpu_percent(interval=None)

    def end(self):
        """Compute resource delta and return consumption statistics."""
        elapsed = time.time() - (self.start_time or time.time())
        cpu_usage = psutil.cpu_percent(interval=None) - (self.start_cpu or 0.0)
        memory_usage = psutil.virtual_memory().used / (1024 ** 3)

        metrics = {
            'time_sec': elapsed,
            'cpu_usage': max(cpu_usage, 0.0),
            'memory_gb': memory_usage,
        }

        if self._has_gputil and self._gpus:
            for gpu in self._gpus:
                metrics[f'GPU_{gpu.id}_load'] = gpu.load
                metrics[f'GPU_{gpu.id}_mem'] = gpu.memoryUsed

        return metrics
