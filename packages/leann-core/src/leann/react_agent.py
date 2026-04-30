"""
ReAct agent for multiturn retrieval with LEANN.

Implements the ReAct (Reasoning + Acting) pattern:
  Thought → Action → Observation → repeat until Final Answer.

Supports:
  - Single index:  ReActAgent(searcher=LeannSearcher("my-index"))
  - Multi-index:   ReActAgent(searchers={"raw": LeannSearcher("raw"), "ev": LeannSearcher("ev")})
    Exposes named tools: search_raw("q"), search_ev("q")
  - Metadata filter passthrough from tool syntax to LeannSearcher.search():
    leann_search("q", filter={"field": "value"})        # normalized to {"field": {"==": "value"}}
    search_raw("q", filter={"field": {"contains": "x"}}) # explicit operator, passed through
  - Web search via Serper API (when configured)
  - Page fetching via Jina Reader (when configured)
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

from .api import LeannSearcher, SearchResult
from .chat import LLMInterface, get_llm
from .web_search import WebSearcher

logger = logging.getLogger(__name__)

# Characters allowed in a search alias after sanitization.
_ALIAS_RE = re.compile(r"[^a-zA-Z0-9_]")


def _sanitize_alias(name: str) -> str:
    """Convert an index name to a safe tool-name component."""
    return _ALIAS_RE.sub("_", name)


def _normalize_filter(raw: dict) -> dict:
    """
    Normalize a filter dict to LEANN's operator format.

    Shorthand  {"field": "value"}           → {"field": {"==": "value"}}
    Explicit   {"field": {"op": value}}     → unchanged
    """
    normalized: dict = {}
    for field, spec in raw.items():
        if isinstance(spec, dict):
            normalized[field] = spec
        else:
            normalized[field] = {"==": spec}
    return normalized


_VALID_OPERATORS = frozenset({
    "==", "!=", "<", "<=", ">", ">=",
    "in", "not_in", "contains", "starts_with", "ends_with",
    "is_true", "is_false",
})

_SCALAR_TYPES = (str, int, float, bool, type(None))


def _parse_filter(filter_str: str) -> tuple[dict | None, str | None]:
    """
    Parse a filter string produced by the LLM into a normalized filter dict.

    Uses ast.literal_eval (safe — evaluates only Python literals) then validates
    shape and operator names before normalizing.

    Returns (filter_dict, None) on success or (None, error_message) on failure.
    """
    try:
        value = ast.literal_eval(filter_str.strip())
    except (ValueError, SyntaxError) as exc:
        return None, (
            f"Invalid filter — could not parse: {exc}. "
            'Expected a dict like {"field": "value"} or {"field": {"==": "value"}}.'
        )

    if not isinstance(value, dict):
        return None, f"filter must be a dict, got {type(value).__name__}"

    for k, v in value.items():
        if not isinstance(k, str):
            return None, f"filter keys must be strings, got {type(k).__name__!r}"
        if isinstance(v, dict):
            for op, operand in v.items():
                if op not in _VALID_OPERATORS:
                    return None, (
                        f"Unknown filter operator {op!r} for field {k!r}. "
                        f"Valid operators: {sorted(_VALID_OPERATORS)}"
                    )
                if op in ("in", "not_in") and not isinstance(operand, list):
                    return None, f"Operator {op!r} requires a list value, got {type(operand).__name__}"
        elif not isinstance(v, _SCALAR_TYPES):
            return None, (
                f"filter value for {k!r} must be a scalar or operator dict, "
                f"got {type(v).__name__}"
            )

    return _normalize_filter(value), None


class ReActAgent:
    """
    ReAct agent for multiturn retrieval with local and web search.

    Accepts either:
      searcher  — single LeannSearcher (backwards-compatible)
      searchers — dict[alias, LeannSearcher] for named multi-index routing

    Tools exposed in the prompt depend on configuration:
      Single index:  leann_search("query")
      Multi-index:   search_<alias>("query") per named index
      Web (optional): web_search("query"), visit_page("url")

    All local tools accept an optional filter kwarg:
      leann_search("query", filter={"field": "value"})
      search_raw_sources("query", filter={"question_posture": {"contains": "challenge"}})
    """

    def __init__(
        self,
        searcher: LeannSearcher | None = None,
        searchers: dict[str, LeannSearcher] | None = None,
        llm: LLMInterface | None = None,
        llm_config: dict[str, Any] | None = None,
        max_iterations: int = 5,
        exa_api_key: str | None = None,
        searxng_url: str | None = None,
        jina_api_key: str | None = None,
        # Legacy — ignored, kept for call-site compat
        serper_api_key: str | None = None,
    ):
        if searchers is not None and searcher is not None:
            raise ValueError("Pass either searcher= or searchers=, not both.")
        if searchers is None and searcher is None:
            raise ValueError("Pass either searcher= or searchers=.")

        # Internal representation: always a dict.  Single-searcher mode uses
        # the sentinel key None to indicate legacy behaviour.
        if searchers is not None:
            self._searchers: dict[str | None, LeannSearcher] = {
                k: v for k, v in searchers.items()
            }
            self._multi = True
        else:
            self._searchers = {None: searcher}  # type: ignore[dict-item]
            self._multi = False

        # Sanitized alias → original key mapping (multi-index only)
        self._alias_map: dict[str, str] = {}
        if self._multi:
            for key in self._searchers:
                sanitized = _sanitize_alias(key)  # type: ignore[arg-type]
                if sanitized in self._alias_map:
                    raise ValueError(
                        f"Alias collision: {key!r} and {self._alias_map[sanitized]!r} "
                        f"both sanitize to {sanitized!r}. Use distinct aliases."
                    )
                self._alias_map[sanitized] = key  # type: ignore[arg-type]

        if llm is None:
            self.llm = get_llm(llm_config)
        else:
            self.llm = llm

        self.max_iterations = max_iterations
        self.search_history: list[dict[str, Any]] = []
        self.web_searcher = WebSearcher(
            exa_api_key=exa_api_key,
            searxng_url=searxng_url,
            jina_api_key=jina_api_key,
        )
        self.web_search_available = self.web_searcher.available

        # Expose .searcher for single-index backwards compatibility
        if not self._multi:
            self.searcher = searcher

    # ── Prompt construction ──────────────────────────────────────────

    def _local_tool_lines(self) -> list[str]:
        """Return tool description lines for all local search tools."""
        if not self._multi:
            return ['1. leann_search("query"): Search the local private knowledge base.']
        lines = []
        for i, alias in enumerate(sorted(self._alias_map), start=1):
            lines.append(f'{i}. search_{alias}("query"): Search the {alias!r} corpus.')
        return lines

    def _local_tool_syntax_examples(self) -> str:
        """Return Action syntax examples for local tools."""
        if not self._multi:
            examples = [
                'Action: leann_search("your query")',
                'Action: leann_search("your query", filter={"field": "value"})',
            ]
        else:
            aliases = sorted(self._alias_map)
            first = aliases[0]
            second = aliases[1] if len(aliases) > 1 else first
            examples = [
                f'Action: search_{first}("your query")',
                f'Action: search_{second}("your query", filter={{"field": "value"}})',
            ]
        return "\n\nOR\n\n".join(f"Thought: [your reasoning]\n{ex}" for ex in examples)

    def _create_react_prompt(
        self, question: str, iteration: int, previous_observations: list[str]
    ) -> str:
        local_tool_desc = "\n".join(self._local_tool_lines())
        _filter_tool = "search_X" if self._multi else "leann_search"
        filter_note = (
            f'\nAll local tools accept an optional filter: {_filter_tool}("q", filter={{"field": "value"}}) '
            f'or {_filter_tool}("q", filter={{"field": {{"contains": "x"}}}}).'
        )

        if self.web_search_available:
            n = len(self._alias_map) if self._multi else 1
            web_num = n + 1
            jina_num = n + 2
            tools_block = (
                "You have access to these tools:\n"
                f"{local_tool_desc}\n"
                f'{web_num}. web_search("query"): Search the public internet.\n'
                f'{jina_num}. visit_page("url"): Read full content of a URL.\n'
                f"{filter_note}\n"
                "\nStrategies:\n"
                "- Use local tools for internal corpora, evidence, or private history.\n"
                "- Use web_search for public documentation, current events, or general concepts.\n"
                "- Use visit_page if you need the full content of a specific URL.\n"
                "- You can combine tools across iterations."
            )
        else:
            tools_block = (
                "You have access to these tools:\n"
                f"{local_tool_desc}\n"
                f"{filter_note}\n"
                "\nNote: Web search is not available. Answer using local corpora only."
            )

        local_examples = self._local_tool_syntax_examples()
        if self.web_search_available:
            action_examples = (
                f"{local_examples}\n\nOR\n\n"
                'Thought: [your reasoning]\nAction: web_search("your query")\n\nOR\n\n'
                "Thought: [your reasoning]\nAction: Final Answer: [your answer]"
            )
        else:
            action_examples = (
                f"{local_examples}\n\nOR\n\n"
                "Thought: [your reasoning]\nAction: Final Answer: [your answer]"
            )

        prompt = (
            "You are a helpful assistant that answers questions by searching through corpora"
        )
        if self.web_search_available:
            prompt += " and the internet"
        prompt += f".\n\nQuestion: {question}\n\n{tools_block}\n\nPrevious observations:\n"

        if previous_observations:
            for i, obs in enumerate(previous_observations, 1):
                prompt += f"\nObservation {i}:\n{obs}\n"
        else:
            prompt += "None yet.\n"

        prompt += (
            f"\nCurrent iteration: {iteration}/{self.max_iterations}\n\n"
            "Think step by step.\n"
            "Format your response EXACTLY like this:\n\n"
            f"Thought: [your reasoning]\n{action_examples}\n\n"
            'IMPORTANT: You MUST start a new line with "Action:" to trigger a tool.\n'
        )
        return prompt

    # ── Response parsing ─────────────────────────────────────────────

    def _parse_llm_response(
        self, response: str
    ) -> tuple[str, str | None]:
        """
        Parse LLM response into (thought, action).

        action is one of:
          "leann_search:query"
          "leann_search:query\x00<filter_json>"   (with filter, \x00 separator)
          "search_<alias>:query"
          "search_<alias>:query\x00<filter_json>"
          "web_search:query"
          "visit_page:url"
          None  (Final Answer)
        """
        thought = ""
        action = None

        if "Thought:" in response:
            thought_part = response.split("Thought:")[1]
            if "Action:" in thought_part:
                thought = thought_part.split("Action:")[0].strip()
            elif "Final Answer:" in thought_part:
                thought = thought_part.split("Final Answer:")[0].strip()
            else:
                thought = thought_part.strip()
        else:
            thought = response.split("Action:")[0].split("Final Answer:")[0].strip()

        if "Final Answer:" in response:
            return thought, None

        if "Action:" not in response:
            # Try to detect bare search() call
            match = re.search(r'search\(["\']([^"\']+)["\']\)', response, re.IGNORECASE)
            if match:
                return thought, f"leann_search:{match.group(1)}"
            return thought, None

        action_part = response.split("Action:")[1].strip()

        # web/visit_page — handle first, simple patterns
        if action_part.startswith("web_search") or action_part.startswith("visit_page"):
            simple = re.search(r'(?:web_search|visit_page)\(["\']([^"\']+)["\']\)', action_part)
            if simple:
                verb = "web_search" if action_part.startswith("web_search") else "visit_page"
                return thought, f"{verb}:{simple.group(1)}"
            return thought, None

        # Local tool — Stage 1: match tool + query (bare, no filter requirement)
        if self._multi:
            aliases = "|".join(re.escape(a) for a in self._alias_map)
            bare_re = re.compile(rf'search_({aliases})\(\s*["\']([^"\']+)["\']', re.DOTALL)
        else:
            bare_re = re.compile(r'(?:leann_search|search)\(\s*["\']([^"\']+)["\']', re.DOTALL)

        m_bare = bare_re.search(action_part)
        if not m_bare:
            return thought, None

        if self._multi:
            alias, query = m_bare.group(1), m_bare.group(2)
            tool_key = f"search_{alias}"
        else:
            query = m_bare.group(1)
            tool_key = "leann_search"

        # Stage 2: try to extract filter= text from what follows the query.
        # Capture anything after filter= (not just {…}) so unparseable filters
        # reach _parse_filter and produce an error observation rather than being silently dropped.
        after_query = action_part[m_bare.end():]
        filter_match = re.search(r'filter\s*=\s*(.+)', after_query, re.DOTALL)
        if filter_match:
            raw_filter = filter_match.group(1).strip()
            # Strip trailing ) that closes the tool call
            raw_filter = re.sub(r'\)\s*$', '', raw_filter).strip()
            return thought, f"{tool_key}:{query}\x00{raw_filter}"

        return thought, f"{tool_key}:{query}"

    # ── Search dispatch ───────────────────────────────────────────────

    def _format_search_results(
        self, results: list[SearchResult], corpus_label: str | None = None
    ) -> str:
        if not results:
            return "No results found."
        formatted = []
        for i, result in enumerate(results, 1):
            entry = f"[Result {i}] (Score: {result.score:.3f})\n{result.text[:500]}..."
            if result.metadata.get("source"):
                entry += f"\nSource: {result.metadata['source']}"
            if corpus_label:
                entry += f"\nCorpus: {corpus_label}"
            formatted.append(entry)
        return "\n\n".join(formatted)

    def search(
        self,
        query: str,
        top_k: int = 5,
        corpus: str | None = None,
        metadata_filters: dict | None = None,
        enable_temporal: bool = False,
    ) -> list[SearchResult]:
        """Search the appropriate corpus, routing by alias when multi-index.

        When `enable_temporal=True`, the underlying searcher parses natural-language
        time expressions from the query into event_time metadata filters. Wave 1
        passthrough; caller filters take precedence on key conflict.
        """
        if self._multi:
            if corpus is None:
                # Default to first registered searcher
                key = next(iter(self._searchers))
            else:
                key = self._alias_map.get(corpus, corpus)
            searcher = self._searchers[key]
        else:
            searcher = self._searchers[None]

        kwargs: dict[str, Any] = {"top_k": top_k}
        if metadata_filters:
            kwargs["metadata_filters"] = metadata_filters
        if enable_temporal:
            kwargs["enable_temporal"] = True

        logger.info(
            f"Searching corpus={corpus!r} query={query!r} "
            f"filters={metadata_filters} enable_temporal={enable_temporal}"
        )
        return searcher.search(query, **kwargs)

    # ── Main loop ────────────────────────────────────────────────────

    def run(self, question: str, top_k: int = 5) -> str:
        logger.info(f"Starting ReAct agent: {question!r}")
        self.search_history = []
        previous_observations: list[str] = []
        all_context: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            prompt = self._create_react_prompt(question, iteration, previous_observations)
            response = self.llm.ask(prompt)
            thought, action = self._parse_llm_response(response)

            logger.info(f"[{iteration}] Thought: {thought!r}  Action: {action!r}")

            if action is None:
                if "Final Answer:" in response:
                    return response.split("Final Answer:")[1].strip()
                return response.strip()

            results_count = 0
            source = "local"
            corpus_name: str | None = None
            observation = ""

            if action.startswith("web_search:"):
                source = "web"
                query_str = action.split(":", 1)[1]
                if not self.web_search_available:
                    observation = (
                        "Web search is not available (no SERPER_API_KEY). "
                        "Use a local search tool instead."
                    )
                else:
                    web_results = self.web_searcher.search(query_str, top_k=top_k)
                    is_error = len(web_results) == 1 and web_results[0].get("title") == "Error"
                    if is_error:
                        observation = (
                            f"Web search failed: {web_results[0].get('snippet', 'Unknown error')}. "
                            "Try a local search tool instead."
                        )
                    elif not web_results:
                        observation = "No web results found."
                    else:
                        results_count = len(web_results)
                        formatted = []
                        for i, res in enumerate(web_results, 1):
                            formatted.append(
                                f"[Web Result {i}]\nTitle: {res['title']}\n"
                                f"Link: {res['link']}\nSnippet: {res['snippet']}"
                            )
                        observation = "\n\n".join(formatted)

            elif action.startswith("visit_page:"):
                source = "web"
                url = action.split(":", 1)[1]
                try:
                    content = self.web_searcher.get_page_content(url)
                except Exception as e:
                    content = f"Error fetching page: {e}"
                results_count = 0 if content.startswith("Error") else 1
                observation = f"Content of {url}:\n{content[:15000]}"

            else:
                # Local search — may have \x00filter_str appended
                parts = action.split(":", 1)
                tool_key = parts[0]  # e.g. "leann_search" or "search_raw_sources"
                rest = parts[1] if len(parts) > 1 else ""

                metadata_filters: dict | None = None
                if "\x00" in rest:
                    query_str, filter_str = rest.split("\x00", 1)
                    parsed, err = _parse_filter(filter_str)
                    if err:
                        observation = (
                            f"Invalid filter — {err}. "
                            'Accepted format: filter={"field": "value"} or '
                            'filter={"field": {"contains": "x"}}. '
                            "Retrying without filter."
                        )
                        previous_observations.append(observation)
                        all_context.append(f"Action: {action}\n{observation}")
                        continue
                    metadata_filters = parsed
                else:
                    query_str = rest

                # Resolve corpus alias
                if self._multi and tool_key.startswith("search_"):
                    corpus_name = tool_key[len("search_"):]  # sanitized alias
                else:
                    corpus_name = None

                results = self.search(
                    query_str,
                    top_k=top_k,
                    corpus=corpus_name,
                    metadata_filters=metadata_filters,
                    enable_temporal=True,
                )
                results_count = len(results)
                label = corpus_name if corpus_name else None
                observation = self._format_search_results(results, corpus_label=label)

            previous_observations.append(observation)
            all_context.append(f"Action: {action}\n{observation}")
            self.search_history.append(
                {
                    "iteration": iteration,
                    "thought": thought,
                    "action": action,
                    "results_count": results_count,
                    "source": source,
                    "corpus": corpus_name,
                }
            )

            if results_count == 0 and iteration >= 2:
                logger.warning("No results, asking LLM for final answer.")
                final_prompt = (
                    f"Based on previous searches, answer the question.\n\n"
                    f"Question: {question}\n\nSearches:\n"
                    + "\n".join(all_context)
                    + "\n\nProvide your final answer."
                )
                return self.llm.ask(final_prompt).strip()

        logger.warning(f"Max iterations ({self.max_iterations}) reached.")
        final_prompt = (
            f"Based on all searches, answer the question.\n\nQuestion: {question}\n\n"
            f"All results:\n" + "\n".join(all_context) + "\n\nProvide your final answer."
        )
        return self.llm.ask(final_prompt).strip()


def create_react_agent(
    index_path: str | dict[str, str],
    llm_config: dict[str, Any] | None = None,
    max_iterations: int = 5,
    exa_api_key: str | None = None,
    searxng_url: str | None = None,
    jina_api_key: str | None = None,
    **searcher_kwargs,
) -> ReActAgent:
    """
    Convenience constructor.

    Single index:
        create_react_agent("my-index")

    Multi-index:
        create_react_agent({"raw": "rubio-raw-sources", "evidence": "rubio-cf-evidence"})
    """
    web_kwargs = dict(exa_api_key=exa_api_key, searxng_url=searxng_url, jina_api_key=jina_api_key)
    if isinstance(index_path, dict):
        searchers = {alias: LeannSearcher(path, **searcher_kwargs) for alias, path in index_path.items()}
        return ReActAgent(searchers=searchers, llm_config=llm_config, max_iterations=max_iterations, **web_kwargs)
    else:
        searcher = LeannSearcher(index_path, **searcher_kwargs)
        return ReActAgent(searcher=searcher, llm_config=llm_config, max_iterations=max_iterations, **web_kwargs)
