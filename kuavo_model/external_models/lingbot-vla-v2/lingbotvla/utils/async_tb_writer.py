"""Non-blocking TensorBoard writer backed by a daemon thread.

Wraps ``SummaryWriter`` so that actual file I/O runs on a background
thread.  The training loop only pays for ``.item()`` (memcpy, GPU
already synced) and ``queue.put()`` (microseconds).
"""

import logging
import queue
import threading

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class AsyncTBWriter:

    _QUEUE_WARN_THRESHOLD = 500

    def __init__(self, log_dir: str):
        self._writer = SummaryWriter(log_dir=log_dir)
        self._queue: queue.Queue = queue.Queue()
        self._warn_logged = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ---- public API (called from main thread) ----

    def add_scalar(self, tag, scalar_value, global_step):
        self._maybe_warn_queue_size()
        self._queue.put(("add_scalar", (tag, scalar_value, global_step)))

    def add_histogram(self, tag, values, global_step):
        self._queue.put(("add_histogram", (tag, values, global_step)))

    def add_histogram_from_counts(self, tag, counts_cpu, global_step):
        """Queue histogram write from a CPU counts tensor [num_experts].

        The ``repeat_interleave`` prep runs on the background thread so
        the main thread is not blocked.
        """
        self._queue.put(("_hist_from_counts", (tag, counts_cpu, global_step)))

    def add_expert_bar(self, tag, counts_cpu, global_step, title=None):
        """Queue an exact per-expert bar chart (matplotlib figure) from a CPU
        counts tensor [num_experts].

        Unlike ``add_histogram`` (which bins expert IDs as if they were a
        continuous variable and visually distorts the edges), a bar chart shows
        one bar per expert = the exact token count it received. Matches
        VideoPretrain's MoE_Expert_Load_Bar. The (slow) matplotlib rendering
        runs on the background thread so training is not blocked.
        """
        self._queue.put(("_expert_bar", (tag, counts_cpu, global_step, title)))

    def flush(self):
        """Block until all queued writes are done, then flush the event file."""
        self._queue.join()
        self._writer.flush()

    def close(self):
        self.flush()
        self._queue.put(None)  # sentinel to stop worker
        self._thread.join(timeout=10)
        self._writer.close()

    # ---- internals ----

    def _maybe_warn_queue_size(self):
        qsize = self._queue.qsize()
        if qsize > self._QUEUE_WARN_THRESHOLD and not self._warn_logged:
            logger.warning(
                "AsyncTBWriter: queue backlog = %d, NFS may be slow", qsize
            )
            self._warn_logged = True
        elif qsize <= self._QUEUE_WARN_THRESHOLD // 2:
            self._warn_logged = False

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            method, args = item
            try:
                if method == "_hist_from_counts":
                    self._do_histogram_from_counts(*args)
                elif method == "_expert_bar":
                    self._do_expert_bar(*args)
                else:
                    getattr(self._writer, method)(*args)
            except Exception as e:
                logger.warning("AsyncTBWriter: %s failed: %s", method, e)
            self._queue.task_done()

    def _do_expert_bar(self, tag, counts_cpu, global_step, title=None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        counts = counts_cpu.detach().float().numpy()
        n = counts.shape[0]
        fig = plt.figure(figsize=(10, 4))
        plt.bar(range(n), counts)
        avg = counts.mean()
        plt.axhline(avg, color="r", linestyle="--", linewidth=1, label=f"avg={avg:.0f}")
        plt.title(title or f"Expert load (step {global_step})")
        plt.xlabel("Expert ID")
        plt.ylabel("Token count")
        plt.legend(loc="upper right", fontsize=8)
        self._writer.add_figure(tag, fig, global_step)
        plt.close(fig)

    def _do_histogram_from_counts(self, tag, counts_cpu, global_step):
        total = counts_cpu.sum().item()
        if total <= 0:
            return
        max_hist_events = 100000
        if total > max_hist_events:
            counts_cpu = counts_cpu * (max_hist_events / total)
        counts_long = counts_cpu.round().long().clamp(min=0)
        num_experts = counts_long.numel()
        indices = torch.repeat_interleave(
            torch.arange(num_experts), counts_long
        )
        if indices.numel() > 0:
            # Integer-aligned bin edges [-0.5, 0.5, ..., num_experts-0.5] so each
            # expert id maps to exactly one bucket. The default 'tensorflow' bins
            # are exponentially spaced and merge high-index experts (2-3 per bucket
            # past id ~11), producing fake peaks/holes in the per-expert view.
            edges = np.arange(num_experts + 1, dtype=np.float64) - 0.5
            self._writer.add_histogram(tag, indices, global_step, bins=edges)
