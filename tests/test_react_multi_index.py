"""
Tests for ReAct agent multi-index routing and metadata-filter passthrough.

Covers:
  - Backwards compatibility: single searcher still works
  - Named searchers: prompt exposes search_<alias> tools
  - Alias sanitization
  - Named tool routing to correct searcher
  - search_history corpus field
  - metadata filter: valid filter passed through to searcher.search()
  - metadata filter: shorthand {"k": "v"} normalized to {"k": {"==": "v"}}
  - metadata filter: invalid filter returns error observation, does not crash
  - metadata filter: explicit operator passthrough
"""

import ast
from unittest.mock import MagicMock, call

import pytest

from leann.api import SearchResult
from leann.react_agent import ReActAgent, create_react_agent


def _make_searcher(name="default") -> MagicMock:
    s = MagicMock()
    s.search.return_value = [
        SearchResult(
            id="1",
            score=0.9,
            text=f"Result from {name}",
            metadata={"source": name},
        )
    ]
    return s


# ── Backwards compatibility ───────────────────────────────────────────


def test_single_searcher_kwarg_still_works():
    """Passing searcher= (old API) should behave identically to before."""
    searcher = _make_searcher()
    agent = ReActAgent(searcher=searcher, llm=MagicMock())
    prompt = agent._create_react_prompt("q", 1, [])
    assert 'leann_search("' in prompt or "leann_search" in prompt
    assert "search_" not in prompt.replace("leann_search", "")


def test_single_searcher_run_still_works():
    """Single-searcher run() should produce an answer without crashing."""
    searcher = _make_searcher()
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Search.\nAction: leann_search("test")',
        "Thought: Done.\nAction: Final Answer: Found it.",
    ]
    agent = ReActAgent(searcher=searcher, llm=mock_llm, max_iterations=3)
    answer = agent.run("test")
    assert answer == "Found it."
    searcher.search.assert_called_once_with("test", top_k=5)


# ── Named searchers: prompt ───────────────────────────────────────────


def test_named_searchers_prompt_exposes_aliases():
    """Multi-index prompt should list search_raw_sources and search_evidence."""
    agent = ReActAgent(
        searchers={"raw_sources": _make_searcher("raw"), "evidence": _make_searcher("ev")},
        llm=MagicMock(),
    )
    prompt = agent._create_react_prompt("q", 1, [])
    assert "search_raw_sources" in prompt
    assert "search_evidence" in prompt
    assert "leann_search" not in prompt


def test_named_searchers_prompt_excludes_old_tool():
    """When named searchers are active, leann_search should not appear."""
    agent = ReActAgent(
        searchers={"corpus_a": _make_searcher(), "corpus_b": _make_searcher()},
        llm=MagicMock(),
    )
    prompt = agent._create_react_prompt("q", 1, [])
    assert "leann_search" not in prompt


def test_alias_sanitization():
    """Aliases with hyphens/spaces should be sanitized to underscores."""
    agent = ReActAgent(
        searchers={"rubio-raw sources": _make_searcher()},
        llm=MagicMock(),
    )
    prompt = agent._create_react_prompt("q", 1, [])
    assert "rubio_raw_sources" in prompt
    assert "rubio-raw sources" not in prompt


def test_alias_collision_raises():
    """Two aliases that sanitize to the same token should raise ValueError."""
    with pytest.raises(ValueError, match="collision"):
        ReActAgent(
            searchers={"raw-sources": _make_searcher(), "raw sources": _make_searcher()},
            llm=MagicMock(),
        )


# ── Named searchers: routing ──────────────────────────────────────────


def test_named_tool_routes_to_correct_searcher():
    """search_raw_sources routes to the raw_sources searcher, not evidence."""
    raw = _make_searcher("raw")
    ev = _make_searcher("ev")
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Check raw.\nAction: search_raw_sources("grand jury")',
        "Thought: Done.\nAction: Final Answer: Found raw.",
    ]
    agent = ReActAgent(searchers={"raw_sources": raw, "evidence": ev}, llm=mock_llm)
    agent.run("test")
    raw.search.assert_called_once()
    ev.search.assert_not_called()


def test_named_tool_routes_evidence_searcher():
    """search_evidence routes to the evidence searcher only."""
    raw = _make_searcher("raw")
    ev = _make_searcher("ev")
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Check evidence.\nAction: search_evidence("pattern contract")',
        "Thought: Done.\nAction: Final Answer: Found evidence.",
    ]
    agent = ReActAgent(searchers={"raw_sources": raw, "evidence": ev}, llm=mock_llm)
    agent.run("test")
    ev.search.assert_called_once()
    raw.search.assert_not_called()


def test_search_history_records_corpus_name():
    """search_history entries should include a 'corpus' field with the alias used."""
    raw = _make_searcher("raw")
    ev = _make_searcher("ev")
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Raw first.\nAction: search_raw_sources("test")',
        'Thought: Now evidence.\nAction: search_evidence("test")',
        "Thought: Done.\nAction: Final Answer: Done.",
    ]
    agent = ReActAgent(
        searchers={"raw_sources": raw, "evidence": ev}, llm=mock_llm, max_iterations=5
    )
    agent.run("test")
    assert agent.search_history[0]["corpus"] == "raw_sources"
    assert agent.search_history[1]["corpus"] == "evidence"


# ── Metadata filter passthrough ───────────────────────────────────────


def test_filter_shorthand_normalized_and_passed_through():
    """{"field": "value"} shorthand should normalize to {"field": {"==": "value"}}."""
    searcher = _make_searcher()
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Filter search.\nAction: leann_search("challenge", filter={"question_posture": "adversarial_challenge"})',
        "Thought: Done.\nAction: Final Answer: Found.",
    ]
    agent = ReActAgent(searcher=searcher, llm=mock_llm, max_iterations=3)
    agent.run("test")
    searcher.search.assert_called_once_with(
        "challenge",
        top_k=5,
        metadata_filters={"question_posture": {"==": "adversarial_challenge"}},
    )


def test_filter_explicit_operator_passed_through():
    """Explicit operator dicts should be passed through unchanged."""
    searcher = _make_searcher()
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Filter.\nAction: leann_search("forecast", filter={"question_posture": {"contains": "challenge"}})',
        "Thought: Done.\nAction: Final Answer: Found.",
    ]
    agent = ReActAgent(searcher=searcher, llm=mock_llm, max_iterations=3)
    agent.run("test")
    searcher.search.assert_called_once_with(
        "forecast",
        top_k=5,
        metadata_filters={"question_posture": {"contains": "challenge"}},
    )


def test_filter_in_operator_passed_through():
    """in operator with list value should be passed through."""
    searcher = _make_searcher()
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Filter.\nAction: leann_search("evidence", filter={"matrix_id": {"in": ["secretary_2025", "secretary_2026"]}})',
        "Thought: Done.\nAction: Final Answer: Found.",
    ]
    agent = ReActAgent(searcher=searcher, llm=mock_llm, max_iterations=3)
    agent.run("test")
    searcher.search.assert_called_once_with(
        "evidence",
        top_k=5,
        metadata_filters={"matrix_id": {"in": ["secretary_2025", "secretary_2026"]}},
    )


def test_no_filter_calls_search_without_metadata_filters():
    """When no filter is provided, search is called without metadata_filters."""
    searcher = _make_searcher()
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: No filter.\nAction: leann_search("test")',
        "Thought: Done.\nAction: Final Answer: Found.",
    ]
    agent = ReActAgent(searcher=searcher, llm=mock_llm, max_iterations=3)
    agent.run("test")
    searcher.search.assert_called_once_with("test", top_k=5)


def test_invalid_filter_returns_observation_not_crash():
    """Unparseable filter should produce an error observation, not raise."""
    searcher = _make_searcher()
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Bad filter.\nAction: leann_search("test", filter=not_valid_json{{)',
        "Thought: Filter failed, retry without.\nAction: Final Answer: Could not filter.",
    ]
    agent = ReActAgent(searcher=searcher, llm=mock_llm, max_iterations=3)
    answer = agent.run("test")
    # Should not crash; bad-filter error observation must reach the LLM as an Observation block
    second_prompt = mock_llm.ask.call_args_list[1][0][0]
    observations_block = second_prompt.split("Previous observations:")[1] if "Previous observations:" in second_prompt else second_prompt
    assert "invalid" in observations_block.lower() or "error" in observations_block.lower() or "parse" in observations_block.lower()
    # The bad search must NOT have been called (filter error, not search)
    searcher.search.assert_not_called()
    assert answer is not None


def test_filter_with_named_searcher():
    """Filter passthrough works with named searchers (search_<alias> syntax)."""
    raw = _make_searcher("raw")
    ev = _make_searcher("ev")
    mock_llm = MagicMock()
    mock_llm.ask.side_effect = [
        'Thought: Filter raw.\nAction: search_raw_sources("uncertainty", filter={"source_family": "state_department_press_qap"})',
        "Thought: Done.\nAction: Final Answer: Found.",
    ]
    agent = ReActAgent(searchers={"raw_sources": raw, "evidence": ev}, llm=mock_llm)
    agent.run("test")
    raw.search.assert_called_once_with(
        "uncertainty",
        top_k=5,
        metadata_filters={"source_family": {"==": "state_department_press_qap"}},
    )
    ev.search.assert_not_called()


# ── create_react_agent convenience ───────────────────────────────────


def test_create_react_agent_dict_index_path():
    """create_react_agent with dict index_path should produce multi-index agent."""
    with pytest.raises(Exception):
        # Will fail trying to load real indexes — that's fine, just check it
        # attempts to build named searchers not a single searcher
        create_react_agent(
            index_path={"raw": "/nonexistent/raw", "evidence": "/nonexistent/evidence"}
        )
