import json
import pathlib

import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile
    q02: numpydantic.NDArray | None = None  # 2nd quantile
    q98: numpydantic.NDArray | None = None  # 98th quantile
    min: numpydantic.NDArray | None = None  # 1st quantile
    max: numpydantic.NDArray | None = None  # 99th quantile


class RunningStatsState(pydantic.BaseModel):
    """Model for persisting the internal state of RunningStats"""
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    
    count: int
    mean: numpydantic.NDArray
    mean_of_squares: numpydantic.NDArray
    min_val: numpydantic.NDArray
    max_val: numpydantic.NDArray
    histograms: numpydantic.NDArray  # Shape: (vector_length, num_bins)
    bin_edges: numpydantic.NDArray   # Shape: (vector_length, num_bins + 1)
    num_quantile_bins: int

class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): A 2D array where each row is a new vector.
        """
        if batch.ndim == 1:
            batch = batch.reshape(-1, 1)

        num_elements, vector_length = batch.shape

        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self, chunk_size=None) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        q02, q98 = self._compute_quantiles([0.02, 0.98])

        if chunk_size is not None:
            assert isinstance(chunk_size, int)
            self._mean = self._mean.reshape(chunk_size, -1)
            self._min = self._min.reshape(chunk_size, -1)
            self._max = self._max.reshape(chunk_size, -1)
            stddev = stddev.reshape(chunk_size, -1)
            q01 = q01.reshape(chunk_size, -1)
            q99 = q99.reshape(chunk_size, -1)
            q02 = q02.reshape(chunk_size, -1)
            q98 = q98.reshape(chunk_size, -1)

        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99, q02=q02, q98=q98, min=self._min, max=self._max)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results

    def get_state(self) -> RunningStatsState:
        """Get all current internal states"""
        if self._count == 0:
            raise ValueError("No data processed yet.")
        return RunningStatsState(
            count=self._count,
            mean=self._mean,
            mean_of_squares=self._mean_of_squares,
            min_val=self._min,
            max_val=self._max,
            histograms=np.stack(self._histograms, axis=0),
            bin_edges=np.stack(self._bin_edges, axis=0),
            num_quantile_bins=self._num_quantile_bins
        )

    @classmethod
    def from_state(cls, state: RunningStatsState):
        """Restore a RunningStats object from its state"""
        instance = cls()
        instance._num_quantile_bins = state.num_quantile_bins
        instance._count = state.count
        instance._mean = np.asarray(state.mean)
        instance._mean_of_squares = np.asarray(state.mean_of_squares)
        instance._min = np.asarray(state.min_val)
        instance._max = np.asarray(state.max_val)
        # After numpydantic serialization, histograms/bin_edges become a single 2D array.
        # Internally we split it back into a list[1D-array] per dim, so that
        # _update_histograms / _adjust_histograms can be reused.
        hist = np.asarray(state.histograms)
        edges = np.asarray(state.bin_edges)
        instance._histograms = [hist[i] for i in range(hist.shape[0])]
        instance._bin_edges = [edges[i] for i in range(edges.shape[0])]
        return instance

    @classmethod
    def merge(cls, others: list["RunningStats"]) -> "RunningStats":
        """Merge multiple RunningStats (typical use: aggregating across ranks).

        Merge formula (per-dim):
            count = Σ cᵢ
            mean = Σ cᵢ·meanᵢ / count
            mean_of_squares = Σ cᵢ·msᵢ / count
            min/max = elementwise min/max
            histograms = rebin each shard's histogram onto unified new_edges, then sum
        """
        valid = [o for o in others if o is not None and o._count > 0]
        if not valid:
            raise ValueError("merge() requires at least one non-empty RunningStats.")
        if len(valid) == 1:
            return valid[0]

        num_bins = valid[0]._num_quantile_bins
        assert all(o._num_quantile_bins == num_bins for o in valid), (
            "All RunningStats must share the same num_quantile_bins to merge."
        )
        vector_length = valid[0]._mean.size
        assert all(o._mean.size == vector_length for o in valid), (
            "All RunningStats must share the same vector length to merge."
        )

        counts = np.array([o._count for o in valid], dtype=np.float64)
        total_count = counts.sum()
        weights = counts / total_count

        merged_mean = sum(w * o._mean for w, o in zip(weights, valid))
        merged_ms = sum(w * o._mean_of_squares for w, o in zip(weights, valid))
        merged_min = np.minimum.reduce([o._min for o in valid])
        merged_max = np.maximum.reduce([o._max for o in valid])

        # Leave a little padding for linspace, consistent with update() init logic (see line 66)
        merged_histograms = []
        merged_bin_edges = []
        for dim in range(vector_length):
            new_edges = np.linspace(
                merged_min[dim] - 1e-10, merged_max[dim] + 1e-10, num_bins + 1
            )
            acc = np.zeros(num_bins)
            for o in valid:
                old_edges = o._bin_edges[dim]
                old_hist = o._histograms[dim]
                # Same rebin approach as _adjust_histograms
                rebinned, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=old_hist)
                acc += rebinned
            merged_histograms.append(acc)
            merged_bin_edges.append(new_edges)

        instance = cls()
        instance._num_quantile_bins = num_bins
        instance._count = int(total_count)
        instance._mean = merged_mean
        instance._mean_of_squares = merged_ms
        instance._min = merged_min
        instance._max = merged_max
        instance._histograms = merged_histograms
        instance._bin_edges = merged_bin_edges
        return instance


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]
    count: int


def serialize_json(norm_stats: dict[str, NormStats], count: int) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats, count=count).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats], count: int) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats, count))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())


class RunningStatsState(pydantic.BaseModel):
    """Model for persisting the internal state of RunningStats"""
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    
    count: int
    mean: numpydantic.NDArray
    mean_of_squares: numpydantic.NDArray
    min_val: numpydantic.NDArray
    max_val: numpydantic.NDArray
    histograms: numpydantic.NDArray  # Shape: (vector_length, num_bins)
    bin_edges: numpydantic.NDArray   # Shape: (vector_length, num_bins + 1)
    num_quantile_bins: int

def save_running_state(path: pathlib.Path | str, stats: dict):
    """Save the full computed intermediate state to JSON"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = {key: state.get_state().model_dump_json() for key, state in stats.items()}
    json.dumps(stats)
    path.write_text(json.dumps(stats))

def load_running_state(path: pathlib.Path | str) -> RunningStats:
    """Load intermediate state from JSON and restore a RunningStats object"""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"State file not found: {path}")
    data = json.loads(path.read_text())
    stats = {key: RunningStats.from_state(RunningStatsState(**json.loads(state))) for key, state in data.items()}
    return stats