"""ONNX Runtime provider selection and model inference helpers.

The QNN provider is intentionally selected first. Provider availability is
runtime-dependent on Windows on ARM64, so the loader keeps a deterministic
DirectML and CPU fallback path for development and unsupported machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class ModelConfig:
    """Configuration needed to create an ONNX Runtime session."""

    model_path: Path
    prefer_qnn: bool = True
    intra_op_num_threads: int = 1


class ModelLoader:
    """Load an ONNX model with QNN -> DirectML -> CPU fallback."""

    def __init__(self, config: ModelConfig) -> None:
        model_path = Path(config.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.model_path = model_path
        self.session, self.execution_backend = self._create_session(config)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type
        self.output_names = [output.name for output in self.session.get_outputs()]

    @staticmethod
    def _create_session(config: ModelConfig) -> tuple[ort.InferenceSession, str]:
        """Create a session, only selecting providers installed locally."""
        available = set(ort.get_available_providers())
        provider_chain: list[Any] = []

        if config.prefer_qnn and "QNNExecutionProvider" in available:
            # QNN defaults to the Hexagon backend when the QNN runtime is
            # correctly installed and configured on the target PC.
            provider_chain.append(("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}))

        if "DmlExecutionProvider" in available:
            provider_chain.append("DmlExecutionProvider")

        if "CPUExecutionProvider" in available:
            provider_chain.append("CPUExecutionProvider")

        if not provider_chain:
            raise RuntimeError("ONNX Runtime has no usable execution provider installed")

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = config.intra_op_num_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # A provider can be listed as available while its native DLL is
        # missing. Retry with the next provider so local development remains
        # usable without weakening the preferred QNN path on the target.
        session = None
        for provider_index in range(len(provider_chain)):
            try:
                session = ort.InferenceSession(
                    str(config.model_path),
                    sess_options=session_options,
                    providers=provider_chain[provider_index:],
                )
                break
            except Exception:
                if provider_index == len(provider_chain) - 1:
                    raise

        assert session is not None

        active_provider = session.get_providers()[0]
        backend = {
            "QNNExecutionProvider": "Hexagon NPU (QNN)",
            "DmlExecutionProvider": "DirectML",
            "CPUExecutionProvider": "CPU",
        }.get(active_provider, active_provider)
        return session, backend

    def infer(self, input_tensor: np.ndarray) -> list[np.ndarray]:
        """Run one inference and return outputs in model declaration order."""
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return [np.asarray(output) for output in outputs]
