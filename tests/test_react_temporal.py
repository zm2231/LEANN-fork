from leann.api import SearchResult
from leann.react_agent import ReActAgent


class FakeSearcher:
    def __init__(self):
        self.calls = []

    def search(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [
            SearchResult(
                id="doc1",
                score=1.0,
                text="Slack update from last week",
                metadata={"source_type": "slack"},
            )
        ]


class FakeLLM:
    def __init__(self, responses=None):
        self.responses = list(responses or [])

    def ask(self, prompt, **kwargs):
        if self.responses:
            return self.responses.pop(0)
        return "Thought: done\nAction: Final Answer: done"


def test_react_search_forwards_metadata_filters_and_enable_temporal():
    searcher = FakeSearcher()
    agent = ReActAgent(searcher=searcher, llm=FakeLLM())

    results = agent.search(
        "Slack discussions last week",
        top_k=3,
        metadata_filters={"source_type": {"==": "slack"}},
        enable_temporal=True,
    )

    assert len(results) == 1
    assert searcher.calls == [
        (
            ("Slack discussions last week",),
            {
                "top_k": 3,
                "metadata_filters": {"source_type": {"==": "slack"}},
                "enable_temporal": True,
            },
        )
    ]


def test_react_run_forwards_temporal_search_options():
    searcher = FakeSearcher()
    llm = FakeLLM(
        [
            'Thought: search local history\nAction: leann_search("Slack discussions last week")',
            "Thought: done\nAction: Final Answer: found it",
        ]
    )
    agent = ReActAgent(searcher=searcher, llm=llm, max_iterations=2)

    answer = agent.run(
        "What happened in Slack last week?",
        top_k=4,
        metadata_filters={"source_type": {"==": "slack"}},
        enable_temporal=True,
    )

    assert answer == "found it"
    assert searcher.calls[-1] == (
        ("Slack discussions last week",),
        {
            "top_k": 4,
            "metadata_filters": {"source_type": {"==": "slack"}},
            "enable_temporal": True,
        },
    )


def test_react_prompt_mentions_temporal_search_capability():
    agent = ReActAgent(searcher=FakeSearcher(), llm=FakeLLM())

    prompt = agent._create_react_prompt("What happened last week?", 1, [])

    assert "natural-language time expressions" in prompt
    assert "last week" in prompt
