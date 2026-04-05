#!/usr/bin/env python3
"""Entry point for LiveCodeBench benchmark-time RLM+RAG provider."""

from livecodebench_method_driver import run_provider


if __name__ == "__main__":
    run_provider("rlm_rag")
