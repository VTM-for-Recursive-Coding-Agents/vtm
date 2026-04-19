from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from vtm.benchmarks import longcot_pilot


class FakeChatClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def create_chat_completion(
        self,
        *,
        model: str,
        prompt: str,
        max_completion_tokens: int,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "prompt": prompt,
                "max_completion_tokens": max_completion_tokens,
            }
        )
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_run_longcot_pilot_writes_outputs_and_metrics(tmp_path) -> None:
    questions = [
        SimpleNamespace(
            question_id="cs_easy_1",
            domain="cs",
            difficulty="easy",
            prompt="Question 1",
        ),
        SimpleNamespace(
            question_id="cs_easy_2",
            domain="cs",
            difficulty="easy",
            prompt="Question 2",
        ),
        SimpleNamespace(
            question_id="cs_easy_3",
            domain="cs",
            difficulty="easy",
            prompt="Question 3",
        ),
    ]
    client = FakeChatClient(
        [
            {
                "choices": [{"message": {"content": "solution = 42"}}],
                "usage": {"completion_tokens": 10},
            },
            {
                "choices": [{"message": {"content": "final answer: 7"}}],
                "usage": {"output_tokens": 30},
            },
            RuntimeError("rate limited"),
        ]
    )

    def verify_fn(question, response_text: str) -> bool:
        return question.question_id == "cs_easy_1" and response_text == "solution = 42"

    summary = longcot_pilot.run_longcot_pilot(
        questions,
        verify_fn=verify_fn,
        client=client,
        model="openrouter/test-model",
        output_dir=tmp_path / "longcot-pilot",
        domain="cs",
        difficulty="easy",
        base_url="https://openrouter.example/api/v1",
        max_completion_tokens=2048,
    )

    assert summary["total"] == 3
    assert summary["correct"] == 1
    assert summary["incorrect"] == 1
    assert summary["failed"] == 1
    assert summary["wrong_formatting"] == 1
    assert summary["accuracy"] == 0.5
    assert summary["overall_accuracy"] == pytest.approx(1 / 3)
    assert summary["median_output_tokens"] == 20

    responses_path = tmp_path / "longcot-pilot" / "responses.jsonl"
    summary_path = tmp_path / "longcot-pilot" / "summary.json"
    paper_table_path = tmp_path / "longcot-pilot" / "paper_table.md"
    assert responses_path.exists()
    assert summary_path.exists()
    assert paper_table_path.exists()

    response_rows = [
        json.loads(line)
        for line in responses_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert response_rows[0]["status"] == "correct"
    assert response_rows[1]["status"] == "incorrect"
    assert response_rows[1]["wrong_formatting"] is True
    assert response_rows[2]["status"] == "failed"
    assert response_rows[2]["successful"] is False

    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary["paper_table_md"] == str(paper_table_path)
    assert "External reasoning pilot only." in paper_table_path.read_text(encoding="utf-8")
    assert client.calls[0]["model"] == "openrouter/test-model"
    assert client.calls[0]["max_completion_tokens"] == 2048


def test_main_reports_missing_longcot_install_instructions(monkeypatch, capsys) -> None:
    import builtins

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001
        if name == "longcot":
            raise ModuleNotFoundError("No module named 'longcot'")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    exit_code = longcot_pilot.main([])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "git clone https://github.com/LongHorizonReasoning/longcot.git" in captured.err
    assert "uv pip install -e ./.vendor/longcot" in captured.err
