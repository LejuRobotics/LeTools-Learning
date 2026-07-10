from __future__ import annotations

from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any


class ModelServerAdapter(ABC):
    """Contract implemented by each model-specific server adapter."""

    name = ""

    @classmethod
    def add_cli_args(cls, parser: ArgumentParser) -> None:
        """Register adapter-specific CLI args."""

    @classmethod
    @abstractmethod
    def from_args(cls, args: Namespace) -> "ModelServerAdapter":
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return {"status": "ok", "adapter": self.name}

    def reset(self) -> dict[str, Any]:
        return {"status": "ok", "message": "adapter reset noop"}

    @abstractmethod
    def select_action(self, obs: dict[str, Any]) -> Any:
        raise NotImplementedError

    def select_action_chunk(self, obs: dict[str, Any]) -> Any:
        """Return a chunk of robot actions.

        Adapters that natively predict chunks should override this. The default
        keeps compatibility for single-step policies by returning a one-step
        chunk.
        """
        return [self.select_action(obs)]


def kuavo_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


DEFAULT_MODEL_REPOS: dict[str, Path] = {
    "openpi": kuavo_repo_root() / "kuavo_model" / "external_models" / "openpi",
    "isaac_gr00t_n17": kuavo_repo_root() / "kuavo_model" / "external_models" / "gr00tn1d7",
    "lingbot_vla": kuavo_repo_root() / "kuavo_model" / "external_models" / "lingbot-vla",
    "lingbot_vla_v2": kuavo_repo_root() / "kuavo_model" / "external_models" / "lingbot-vla-v2",
}


def resolve_model_repo_root(adapter_name: str, model_repo_root: str | None = None) -> Path:
    if model_repo_root:
        repo_path = Path(model_repo_root).expanduser().resolve()
    else:
        try:
            repo_path = DEFAULT_MODEL_REPOS[adapter_name].resolve()
        except KeyError as exc:
            raise ValueError(
                f"Adapter `{adapter_name}` does not have a built-in default repo path. "
                "Please provide `--model_repo_root`."
            ) from exc

    if not repo_path.is_dir():
        if model_repo_root:
            raise FileNotFoundError(f"repo not found at: {repo_path}")
        raise FileNotFoundError(
            f"Default repo for adapter `{adapter_name}` not found at: {repo_path}. "
            "Please vendor the model repo into `kuavo_model/external_models` or pass `--model_repo_root`."
        )

    return repo_path
