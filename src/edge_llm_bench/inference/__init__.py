"""Inference backend abstraction (Ollama, llama.cpp, ...)."""

from .base import BackendRegistry, GenerateResult, InferenceBackend
from .ollama_backend import OllamaBackend

# Importing the backend modules is what registers them in BackendRegistry.
# llama.cpp is registered inside its own module (only when installed).
__all__ = ["BackendRegistry", "GenerateResult", "InferenceBackend", "OllamaBackend"]
