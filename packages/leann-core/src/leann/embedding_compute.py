"""
Unified embedding computation module
Consolidates all embedding computation logic using SentenceTransformer
Preserves all optimization parameters to ensure performance
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Optional, Protocol, cast

import numpy as np

from .settings import resolve_ollama_host, resolve_openai_api_key, resolve_openai_base_url

# torch and tiktoken are imported lazily inside the functions that use them, so
# `import leann` (e.g. for MCP search over an existing index, BM25-only flows,
# or non-embedding utilities) doesn't pull torch's ~1 GB of state into memory.

# Set up logger with proper level
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LEANN_LOG_LEVEL", "WARNING").upper()
log_level = getattr(logging, LOG_LEVEL, logging.WARNING)
logger.setLevel(log_level)


class _SentenceTransformerLike(Protocol):
    def eval(self) -> Any: ...
    def parameters(self) -> Any: ...
    def encode(self, *args: Any, **kwargs: Any) -> Any: ...
    def half(self) -> Any: ...


# Token limit registry for embedding models
# Used as fallback when dynamic discovery fails (e.g., LM Studio, OpenAI)
# Ollama models use dynamic discovery via /api/show
EMBEDDING_MODEL_LIMITS = {
    # Nomic models (common across servers)
    "nomic-embed-text": 2048,  # Corrected from 512 - verified via /api/show
    "nomic-embed-text-v1.5": 2048,
    "nomic-embed-text-v2": 512,
    # Other embedding models
    "mxbai-embed-large": 512,
    "all-minilm": 512,
    "bge-m3": 8192,
    "snowflake-arctic-embed": 512,
    # OpenAI models
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-ada-002": 8192,
}

# Runtime cache for dynamically discovered token limits
# Key: (model_name, base_url), Value: token_limit
# Prevents repeated SDK/API calls for the same model
_token_limit_cache: dict[tuple[str, str], int] = {}


def get_model_token_limit(
    model_name: str,
    base_url: Optional[str] = None,
    default: int = 2048,
) -> int:
    """
    Get token limit for a given embedding model.
    Uses hybrid approach: dynamic discovery for Ollama, registry fallback for others.
    Caches discovered limits to prevent repeated API/SDK calls.

    Args:
        model_name: Name of the embedding model
        base_url: Base URL of the embedding server (for dynamic discovery)
        default: Default token limit if model not found

    Returns:
        Token limit for the model in tokens
    """
    # Check cache first to avoid repeated SDK/API calls
    cache_key = (model_name, base_url or "")
    if cache_key in _token_limit_cache:
        cached_limit = _token_limit_cache[cache_key]
        logger.debug(f"Using cached token limit for {model_name}: {cached_limit}")
        return cached_limit

    # Try Ollama dynamic discovery if base_url provided
    if base_url:
        # Detect Ollama servers by port or "ollama" in URL
        if "11434" in base_url or "ollama" in base_url.lower():
            limit = _query_ollama_context_limit(model_name, base_url)
            if limit:
                _token_limit_cache[cache_key] = limit
                return limit

        # Try LM Studio SDK discovery
        if "1234" in base_url or "lmstudio" in base_url.lower() or "lm.studio" in base_url.lower():
            # Convert HTTP to WebSocket URL
            ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://")
            # Remove /v1 suffix if present
            if ws_url.endswith("/v1"):
                ws_url = ws_url[:-3]

            limit = _query_lmstudio_context_limit(model_name, ws_url)
            if limit:
                _token_limit_cache[cache_key] = limit
                return limit

    # Fallback to known model registry with version handling (from PR #154)
    # Handle versioned model names (e.g., "nomic-embed-text:latest" -> "nomic-embed-text")
    base_model_name = model_name.split(":")[0]

    # Check exact match first
    if model_name in EMBEDDING_MODEL_LIMITS:
        limit = EMBEDDING_MODEL_LIMITS[model_name]
        _token_limit_cache[cache_key] = limit
        return limit

    # Check base name match
    if base_model_name in EMBEDDING_MODEL_LIMITS:
        limit = EMBEDDING_MODEL_LIMITS[base_model_name]
        _token_limit_cache[cache_key] = limit
        return limit

    # Check partial matches for common patterns
    for known_model, registry_limit in EMBEDDING_MODEL_LIMITS.items():
        if known_model in base_model_name or base_model_name in known_model:
            _token_limit_cache[cache_key] = registry_limit
            return registry_limit

    # Default fallback
    logger.warning(f"Unknown model '{model_name}', using default {default} token limit")
    _token_limit_cache[cache_key] = default
    return default


# Embedding models whose true tokenizer IS OpenAI's cl100k_base. For these,
# counting with tiktoken is exact, so no HuggingFace tokenizer is needed.
_OPENAI_TIKTOKEN_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
}

# Short model aliases → HuggingFace repo id, for loading the model's own
# tokenizer. bge-m3 tokenizes with an XLM-RoBERTa SentencePiece tokenizer that
# produces ~1.16x more tokens than cl100k for the same text; counting with
# cl100k under-counts and lets text silently exceed the model's real 8192-token
# context window (observed: 8192 cl100k tokens = 9534 bge-m3 tokens).
_HF_TOKENIZER_ALIASES = {
    "bge-m3": "BAAI/bge-m3",
}

# Cache loaded tokenizers by key so we don't reload on every batch.
_truncation_tokenizer_cache: dict[str, Any] = {}


def _resolve_truncation_tokenizer(model_name: Optional[str]):
    """Return ("hf", PreTrainedTokenizer) or ("tiktoken", Encoding).

    Picks the model's OWN tokenizer when it can be identified/loaded, so the
    token count matches what the embedding server actually tokenizes. Falls back
    to cl100k_base when the model is an OpenAI model (where cl100k is correct) or
    when the real tokenizer can't be loaded.
    """

    def _tiktoken():
        cached = _truncation_tokenizer_cache.get("__tiktoken__")
        if cached is None:
            import tiktoken

            cached = tiktoken.get_encoding("cl100k_base")
            _truncation_tokenizer_cache["__tiktoken__"] = cached
        return ("tiktoken", cached)

    if not model_name:
        return _tiktoken()

    base = model_name.split(":")[0]  # strip ":latest" etc.
    if base in _OPENAI_TIKTOKEN_MODELS:
        return _tiktoken()

    # Resolve to an HF repo id: explicit id (contains "/") or a known alias.
    hf_id = model_name if "/" in model_name else _HF_TOKENIZER_ALIASES.get(base)
    if not hf_id:
        return _tiktoken()

    cached = _truncation_tokenizer_cache.get(hf_id)
    if cached is None:
        try:
            from transformers import AutoTokenizer

            cached = AutoTokenizer.from_pretrained(hf_id)
            _truncation_tokenizer_cache[hf_id] = cached
        except Exception as e:
            logger.warning(
                f"Could not load real tokenizer for '{hf_id}' ({e}); "
                "falling back to cl100k_base token counting (may under-count)"
            )
            return _tiktoken()
    return ("hf", cached)


def truncate_to_token_limit(
    texts: list[str], token_limit: int, model_name: Optional[str] = None
) -> list[str]:
    """
    Truncate texts to fit within a model's token limit.

    Counts tokens with the model's OWN tokenizer when available (e.g. bge-m3's
    SentencePiece tokenizer) so the count matches what the embedding server sees;
    otherwise falls back to cl100k_base. For HuggingFace tokenizers the special
    tokens the model adds at encode time ([CLS]/[SEP]) are reserved out of the
    budget so the re-tokenized text on the server stays within `token_limit`.

    Args:
        texts: List of text strings to truncate
        token_limit: Maximum number of tokens allowed (the model's real context)
        model_name: Embedding model name, used to pick the correct tokenizer

    Returns:
        List of truncated texts (same length as input)
    """
    if not texts:
        return []

    kind, tok = _resolve_truncation_tokenizer(model_name)

    if kind == "hf":
        # Reserve room for the special tokens the model adds on encode, so the
        # server's re-tokenization (which re-adds them) stays within the limit.
        reserve = tok.num_special_tokens_to_add(pair=False)
        budget = max(1, token_limit - reserve)

        def _encode(text):
            return tok.encode(text, add_special_tokens=False)

        def _decode(ids):
            return tok.decode(ids, skip_special_tokens=True)
    else:

        def _encode(text):
            # disallowed_special=() so literal "<|endoftext|>" / "<|im_start|>" /
            # similar strings that appear in real content (agent session logs,
            # tokenization tutorials, model output examples) are tokenized as
            # ordinary text instead of raising.
            return tok.encode(text, disallowed_special=())

        def _decode(ids):
            return tok.decode(ids)

        budget = token_limit

    truncated_texts = []
    truncation_count = 0
    total_tokens_removed = 0
    max_original_length = 0

    for i, text in enumerate(texts):
        tokens = _encode(text)
        original_length = len(tokens)

        if original_length <= budget:
            # Text is within limit, keep as is
            truncated_texts.append(text)
        else:
            # Truncate to budget
            truncated_tokens = tokens[:budget]
            truncated_text = _decode(truncated_tokens)
            truncated_texts.append(truncated_text)

            # Track truncation statistics
            truncation_count += 1
            tokens_removed = original_length - budget
            total_tokens_removed += tokens_removed
            max_original_length = max(max_original_length, original_length)

            # Log individual truncation at WARNING level (first few only)
            if truncation_count <= 3:
                logger.warning(
                    f"Text {i + 1} truncated: {original_length} → {budget} tokens "
                    f"({tokens_removed} tokens removed, {kind} tokenizer)"
                )
            elif truncation_count == 4:
                logger.warning("Further truncation warnings suppressed...")

    # Log summary at INFO level
    if truncation_count > 0:
        logger.warning(
            f"Truncation summary: {truncation_count}/{len(texts)} texts truncated "
            f"(removed {total_tokens_removed} tokens total, longest was {max_original_length} tokens, "
            f"{kind} tokenizer)"
        )
    else:
        logger.debug(f"No truncation needed - all {len(texts)} texts within {budget} token budget")

    return truncated_texts


def truncate_for_model(
    texts: list[str],
    model_name: str,
    base_url: Optional[str] = None,
) -> list[str]:
    """Clip texts to a model's real token limit — the client-side embedding guard.

    One call that resolves the model's context window (registry + dynamic
    discovery, via get_model_token_limit) and truncates each text with the
    model's OWN tokenizer (truncate_to_token_limit). Any embedding client should
    call this before sending text to an embedding server — including non-LEANN
    scripts that POST to a self-hosted shim directly — so no request can exceed
    the model's context window. Over-limit input otherwise 413s on a remote shim,
    or on a local/MLX model OOMs by padding the whole batch up to the oversized
    row (metal::malloc).

    Example (client-side, no LEANN index involved):
        from leann import truncate_for_model
        safe = truncate_for_model(rows, "BAAI/bge-m3")   # clips to 8192 tokens
        # ... POST `safe` to the embedding endpoint ...
    """
    token_limit = get_model_token_limit(model_name, base_url=base_url)
    return truncate_to_token_limit(texts, token_limit, model_name=model_name)


def _query_ollama_context_limit(model_name: str, base_url: str) -> Optional[int]:
    """
    Query Ollama /api/show for model context limit.

    Args:
        model_name: Name of the Ollama model
        base_url: Base URL of the Ollama server

    Returns:
        Context limit in tokens if found, None otherwise
    """
    try:
        import requests

        response = requests.post(
            f"{base_url}/api/show",
            json={"name": model_name},
            timeout=5,
        )
        if response.status_code == 200:
            data = response.json()
            if "model_info" in data:
                # Look for *.context_length in model_info
                for key, value in data["model_info"].items():
                    if "context_length" in key and isinstance(value, int):
                        logger.info(f"Detected {model_name} context limit: {value} tokens")
                        return value
    except Exception as e:
        logger.debug(f"Failed to query Ollama context limit: {e}")

    return None


def _query_lmstudio_context_limit(model_name: str, base_url: str) -> Optional[int]:
    """
    Query LM Studio SDK for model context length via Node.js subprocess.

    Args:
        model_name: Name of the LM Studio model
        base_url: Base URL of the LM Studio server (WebSocket format, e.g., "ws://localhost:1234")

    Returns:
        Context limit in tokens if found, None otherwise
    """
    # Inline JavaScript using @lmstudio/sdk
    # Note: Load model temporarily for metadata, then unload to respect JIT auto-evict
    js_code = f"""
    const {{ LMStudioClient }} = require('@lmstudio/sdk');
    (async () => {{
        try {{
            const client = new LMStudioClient({{ baseUrl: '{base_url}' }});
            const model = await client.embedding.load('{model_name}', {{ verbose: false }});
            const contextLength = await model.getContextLength();
            await model.unload();  // Unload immediately to respect JIT auto-evict settings
            console.log(JSON.stringify({{ contextLength, identifier: '{model_name}' }}));
        }} catch (error) {{
            console.error(JSON.stringify({{ error: error.message }}));
            process.exit(1);
        }}
    }})();
    """

    try:
        # Set NODE_PATH to include global modules for @lmstudio/sdk resolution
        env = os.environ.copy()

        # Try to get npm global root (works with nvm, brew node, etc.)
        try:
            npm_root = subprocess.run(
                ["npm", "root", "-g"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if npm_root.returncode == 0:
                global_modules = npm_root.stdout.strip()
                # Append to existing NODE_PATH if present
                existing_node_path = env.get("NODE_PATH", "")
                env["NODE_PATH"] = (
                    f"{global_modules}:{existing_node_path}"
                    if existing_node_path
                    else global_modules
                )
        except Exception:
            # If npm not available, continue with existing NODE_PATH
            pass

        result = subprocess.run(
            ["node", "-e", js_code],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )

        if result.returncode != 0:
            logger.debug(f"LM Studio SDK error: {result.stderr}")
            return None

        data = json.loads(result.stdout)
        context_length = data.get("contextLength")

        if context_length and context_length > 0:
            logger.info(f"LM Studio SDK detected {model_name} context length: {context_length}")
            return context_length

    except FileNotFoundError:
        logger.debug("Node.js not found - install Node.js for LM Studio SDK features")
    except subprocess.TimeoutExpired:
        logger.debug("LM Studio SDK query timeout")
    except json.JSONDecodeError:
        logger.debug("LM Studio SDK returned invalid JSON")
    except Exception as e:
        logger.debug(f"LM Studio SDK query failed: {e}")

    return None


# Global model cache to avoid repeated loading
_model_cache: dict[str, Any] = {}


def compute_embeddings(
    texts: list[str],
    model_name: str,
    mode: str = "sentence-transformers",
    is_build: bool = False,
    batch_size: int = 32,
    adaptive_optimization: bool = True,
    manual_tokenize: bool = False,
    max_length: int = 512,
    provider_options: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """
    Unified embedding computation entry point

    Args:
        texts: List of texts to compute embeddings for
        model_name: Model name
        mode: Computation mode ('sentence-transformers', 'openai', 'mlx', 'ollama')
        is_build: Whether this is a build operation (shows progress bar)
        batch_size: Batch size for processing
        adaptive_optimization: Whether to use adaptive optimization based on batch size

    Returns:
        Normalized embeddings array, shape: (len(texts), embedding_dim)
    """
    provider_options = provider_options or {}
    wrapper_start_time = time.time()
    logger.debug(
        f"[compute_embeddings] entry: mode={mode}, model='{model_name}', text_count={len(texts)}"
    )

    # Allow batch_size override from provider_options (disables adaptive_optimization)
    if "batch_size" in provider_options:
        batch_size = provider_options["batch_size"]
        adaptive_optimization = False  # User-specified batch_size takes precedence

    if mode == "sentence-transformers":
        inner_start_time = time.time()
        result = compute_embeddings_sentence_transformers(
            texts,
            model_name,
            is_build=is_build,
            batch_size=batch_size,
            adaptive_optimization=adaptive_optimization,
            manual_tokenize=manual_tokenize,
            max_length=max_length,
        )
        inner_end_time = time.time()
        wrapper_end_time = time.time()
        logger.debug(
            "[compute_embeddings] sentence-transformers timings: "
            f"inner={inner_end_time - inner_start_time:.6f}s, "
            f"wrapper_total={wrapper_end_time - wrapper_start_time:.6f}s"
        )
        return result
    elif mode == "openai":
        return compute_embeddings_openai(
            texts,
            model_name,
            base_url=provider_options.get("base_url"),
            api_key=provider_options.get("api_key"),
            provider_options=provider_options,
        )
    elif mode == "mlx":
        return compute_embeddings_mlx(texts, model_name)
    elif mode == "ollama":
        return compute_embeddings_ollama(
            texts,
            model_name,
            is_build=is_build,
            host=provider_options.get("host"),
            provider_options=provider_options,
        )
    elif mode == "gemini":
        return compute_embeddings_gemini(texts, model_name, is_build=is_build)
    else:
        raise ValueError(f"Unsupported embedding mode: {mode}")


def compute_embeddings_sentence_transformers(
    texts: list[str],
    model_name: str,
    use_fp16: bool = True,
    device: str = "auto",
    batch_size: int = 32,
    is_build: bool = False,
    adaptive_optimization: bool = True,
    manual_tokenize: bool = False,
    max_length: int = 512,
) -> np.ndarray:
    """
    Compute embeddings using SentenceTransformer with model caching and adaptive optimization

    Args:
        texts: List of texts to compute embeddings for
        model_name: Model name
        use_fp16: Whether to use FP16 precision
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        batch_size: Batch size for processing
        is_build: Whether this is a build operation (shows progress bar)
        adaptive_optimization: Whether to use adaptive optimization based on batch size
    """
    import torch

    outer_start_time = time.time()
    # Handle empty input
    if not texts:
        raise ValueError("Cannot compute embeddings for empty text list")
    logger.info(
        f"Computing embeddings for {len(texts)} texts using SentenceTransformer, model: '{model_name}'"
    )

    # Auto-detect device
    if device == "auto":
        # Check environment variable first
        env_device = os.getenv("LEANN_EMBEDDING_DEVICE")
        if env_device:
            device = env_device
            logger.info(f"Using device from LEANN_EMBEDDING_DEVICE: {device}")
        elif torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Apply optimizations based on benchmark results
    if adaptive_optimization:
        # Use optimal batch_size constants for different devices based on benchmark results
        if device == "mps":
            batch_size = 128  # MPS optimal batch size from benchmark
            if model_name == "Qwen/Qwen3-Embedding-0.6B":
                batch_size = 32
        elif device == "cuda":
            batch_size = 256  # CUDA optimal batch size
        # Keep original batch_size for CPU

    # Create cache key
    cache_key = f"sentence_transformers_{model_name}_{device}_{use_fp16}_optimized"

    pre_model_init_end_time = time.time()
    logger.debug(
        "compute_embeddings_sentence_transformers pre-model-init time "
        f"(device/batch selection etc.): {pre_model_init_end_time - outer_start_time:.6f}s"
    )

    # Check if model is already cached
    start_time = time.time()
    if cache_key in _model_cache:
        logger.info(f"Using cached optimized model: {model_name}")
        model = cast(_SentenceTransformerLike, _model_cache[cache_key])
    else:
        logger.info(f"Loading and caching optimized SentenceTransformer model: {model_name}")
        from sentence_transformers import SentenceTransformer

        logger.info(f"Using device: {device}")

        # Apply hardware optimizations
        if device == "cuda":
            # TODO: Haven't tested this yet
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.set_per_process_memory_fraction(0.9)
        elif device == "mps":
            # No device-level init for MPS. set_per_process_memory_fraction causes
            # greedy allocation; torch.compile causes graph buffer bloat. Cache
            # clearing is handled per-batch in the compute loop below.
            pass
        elif device == "cpu":
            # TODO: Haven't tested this yet
            torch.set_num_threads(min(8, os.cpu_count() or 4))
            try:
                torch.backends.mkldnn.enabled = True
            except AttributeError:
                pass

        # Prepare optimized model and tokenizer parameters
        model_kwargs = {
            "torch_dtype": torch.float16 if use_fp16 else torch.float32,
            "low_cpu_mem_usage": True,
            "_fast_init": True,
            "attn_implementation": "eager",  # Use eager attention for speed
        }

        tokenizer_kwargs = {
            "use_fast": True,
            "padding": True,
            "truncation": True,
        }

        try:
            # Try loading with advanced parameters first (newer versions)
            local_model_kwargs = model_kwargs.copy()
            local_tokenizer_kwargs = tokenizer_kwargs.copy()
            local_model_kwargs["local_files_only"] = True
            local_tokenizer_kwargs["local_files_only"] = True

            model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs=local_model_kwargs,
                tokenizer_kwargs=local_tokenizer_kwargs,
                local_files_only=True,
            )
            logger.info("Model loaded successfully! (local + optimized)")
        except TypeError as e:
            if "model_kwargs" in str(e) or "tokenizer_kwargs" in str(e):
                logger.warning(
                    f"Advanced parameters not supported ({e}), using basic initialization..."
                )
                # Fallback to basic initialization for older versions
                try:
                    model = SentenceTransformer(
                        model_name,
                        device=device,
                        local_files_only=True,
                    )
                    logger.info("Model loaded successfully! (local + basic)")
                except Exception as e2:
                    logger.warning(f"Local loading failed ({e2}), trying network download...")
                    model = SentenceTransformer(
                        model_name,
                        device=device,
                        local_files_only=False,
                    )
                    logger.info("Model loaded successfully! (network + basic)")
            else:
                raise
        except Exception as e:
            logger.warning(f"Local loading failed ({e}), trying network download...")
            # Fallback to network loading with advanced parameters
            try:
                network_model_kwargs = model_kwargs.copy()
                network_tokenizer_kwargs = tokenizer_kwargs.copy()
                network_model_kwargs["local_files_only"] = False
                network_tokenizer_kwargs["local_files_only"] = False

                model = SentenceTransformer(
                    model_name,
                    device=device,
                    model_kwargs=network_model_kwargs,
                    tokenizer_kwargs=network_tokenizer_kwargs,
                    local_files_only=False,
                )
                logger.info("Model loaded successfully! (network + optimized)")
            except TypeError as e2:
                if "model_kwargs" in str(e2) or "tokenizer_kwargs" in str(e2):
                    logger.warning(
                        f"Advanced parameters not supported ({e2}), using basic network loading..."
                    )
                    model = SentenceTransformer(
                        model_name,
                        device=device,
                        local_files_only=False,
                    )
                    logger.info("Model loaded successfully! (network + basic)")
                else:
                    raise

        # Apply additional optimizations based on mode
        if use_fp16 and device in ["cuda", "mps"]:
            try:
                model = model.half()
                logger.info(f"Applied FP16 precision: {model_name}")
            except Exception as e:
                logger.warning(f"FP16 optimization failed: {e}")

        # Apply torch.compile optimization
        if device == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead", dynamic=True)
                logger.info(f"Applied torch.compile optimization: {model_name}")
            except Exception as e:
                logger.warning(f"torch.compile optimization failed: {e}")

        model = cast(_SentenceTransformerLike, model)

        # Set model to eval mode and disable gradients for inference
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)

        # Cache the model
        _model_cache[cache_key] = model
        logger.info(f"Model cached: {cache_key}")

        end_time = time.time()

        # Compute embeddings with optimized inference mode
        logger.info(
            f"Starting embedding computation... (batch_size: {batch_size}, manual_tokenize={manual_tokenize})"
        )
        logger.info(f"start sentence transformers {model} takes {end_time - start_time}")

    start_time = time.time()
    if not manual_tokenize:
        # Use SentenceTransformer's optimized encode path (default)
        with torch.inference_mode():
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=is_build,  # Don't show progress bar in server environment
                convert_to_numpy=True,
                normalize_embeddings=False,
                device=device,
            )
        # Synchronize if CUDA to measure accurate wall time
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
    else:
        # Manual tokenization + forward pass using HF AutoTokenizer/AutoModel.
        # This path is reserved for an aggressively optimized FP pipeline
        # (no quantization), mainly for experimentation.
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:
            raise ImportError(f"transformers is required for manual_tokenize=True: {e}")

        tok_cache_key = f"hf_tokenizer_{model_name}"
        mdl_cache_key = f"hf_model_{model_name}_{device}_{use_fp16}_fp"

        if tok_cache_key in _model_cache and mdl_cache_key in _model_cache:
            hf_tokenizer = _model_cache[tok_cache_key]
            hf_model = _model_cache[mdl_cache_key]
            logger.info("Using cached HF tokenizer/model for manual FP path")
        else:
            logger.info("Loading HF tokenizer/model for manual FP path")
            hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            torch_dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
            hf_model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
            )
            hf_model.to(device)

            hf_model.eval()
            # Optional compile on supported devices
            if device == "cuda":
                try:
                    hf_model = torch.compile(hf_model, mode="reduce-overhead", dynamic=True)
                    logger.info(
                        f"Applied torch.compile to HF model for {model_name} "
                        f"(device={device}, dtype={torch_dtype})"
                    )
                except Exception as exc:
                    logger.warning(f"torch.compile optimization failed: {exc}")

            _model_cache[tok_cache_key] = hf_tokenizer
            _model_cache[mdl_cache_key] = hf_model

        all_embeddings: list[np.ndarray] = []
        # Progress bar when building or for large inputs
        show_progress = is_build or len(texts) > 32
        try:
            if show_progress:
                from tqdm import tqdm

                batch_iter = tqdm(
                    range(0, len(texts), batch_size),
                    desc="Embedding (manual)",
                    unit="batch",
                )
            else:
                batch_iter = range(0, len(texts), batch_size)
        except Exception:
            batch_iter = range(0, len(texts), batch_size)

        start_time_manual = time.time()
        with torch.inference_mode():
            for start_index in batch_iter:
                end_index = min(start_index + batch_size, len(texts))
                batch_texts = texts[start_index:end_index]
                inputs = hf_tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = hf_model(**inputs)
                last_hidden_state = outputs.last_hidden_state  # (B, L, H)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    pooled = last_hidden_state.mean(dim=1)
                else:
                    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
                    masked = last_hidden_state * mask
                    lengths = mask.sum(dim=1).clamp(min=1)
                    pooled = masked.sum(dim=1) / lengths
                batch_embeddings = pooled.detach().to("cpu").float().numpy()
                all_embeddings.append(batch_embeddings)
                if device == "mps":
                    try:
                        torch.mps.empty_cache()
                    except (RuntimeError, AttributeError):
                        pass

        embeddings = np.vstack(all_embeddings).astype(np.float32, copy=False)
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        end_time = time.time()
        logger.info(f"Manual tokenize time taken: {end_time - start_time_manual} seconds")
    end_time = time.time()
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")
    logger.info(f"Time taken: {end_time - start_time} seconds")

    # Validate results
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise RuntimeError(f"Detected NaN or Inf values in embeddings, model: {model_name}")

    outer_end_time = time.time()
    logger.debug(
        "compute_embeddings_sentence_transformers total time "
        f"(function entry -> return): {outer_end_time - outer_start_time:.6f}s"
    )

    return embeddings


def compute_embeddings_openai(
    texts: list[str],
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    provider_options: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    # TODO: @yichuan-w add progress bar only in build mode
    """Compute embeddings using OpenAI API"""
    try:
        import openai
    except ImportError as e:
        raise ImportError(f"OpenAI package not installed: {e}")

    # Validate input list
    if not texts:
        raise ValueError("Cannot compute embeddings for empty text list")
    # Extra validation: abort early if any item is empty/whitespace
    invalid_count = sum(1 for t in texts if not isinstance(t, str) or not t.strip())
    if invalid_count > 0:
        raise ValueError(
            f"Found {invalid_count} empty/invalid text(s) in input. Upstream should filter before calling OpenAI."
        )

    # Extract base_url and api_key from provider_options if not provided directly
    provider_options = provider_options or {}
    effective_base_url = base_url or provider_options.get("base_url")
    effective_api_key = api_key or provider_options.get("api_key")

    resolved_base_url = resolve_openai_base_url(effective_base_url)
    resolved_api_key = resolve_openai_api_key(effective_api_key)

    if not resolved_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")

    openai_timeout = float(os.environ.get("LEANN_OPENAI_TIMEOUT", "600"))

    # Create OpenAI client
    client = openai.OpenAI(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        timeout=openai_timeout,
    )

    logger.info(
        f"Computing embeddings for {len(texts)} texts using OpenAI API, model: '{model_name}'"
    )

    # Apply prompt template if provided
    # Priority: build_prompt_template (new format) > prompt_template (old format)
    prompt_template = provider_options.get("build_prompt_template") or provider_options.get(
        "prompt_template"
    )

    if prompt_template:
        logger.warning(f"Applying prompt template: '{prompt_template}'")
        texts = [f"{prompt_template}{text}" for text in texts]

    # Clip to the model's real token limit before sending (client-side guard).
    texts = truncate_for_model(texts, model_name, base_url=effective_base_url)

    # OpenAI has limits on batch size and input length
    max_batch_size = 800  # Conservative batch size because the token limit is 300K
    all_embeddings = []
    text_lengths = [len(text) for text in texts]
    # get the avg len of texts
    avg_len = sum(text_lengths) / len(texts)
    # if avg len is less than 1000, use the max batch size
    if avg_len > 300:
        max_batch_size = 500

    # Gemini's OpenAI-compatible endpoint hard-limits embedding batches to 100 inputs per request.
    # If we exceed this, the API returns:
    #   "BatchEmbedContentsRequest.requests: at most 100 requests can be in one batch"
    if "generativelanguage.googleapis.com" in (resolved_base_url or ""):
        max_batch_size = min(max_batch_size, 100)
        logger.info(
            "Detected Gemini OpenAI-compatible base_url; capping embedding batch_size to %d.",
            max_batch_size,
        )

    # Alibaba Cloud DashScope's OpenAI-compatible endpoint hard-limits embedding batches
    # to 10 inputs per request (e.g. text-embedding-v4). Exceeding this returns HTTP 400:
    #   "InternalError.Algo.InvalidParameter: Value error, batch size is invalid,
    #    it should not be larger than 10.: input.contents"
    if "dashscope" in (resolved_base_url or ""):
        max_batch_size = min(max_batch_size, 10)
        logger.info(
            "Detected DashScope OpenAI-compatible base_url; capping embedding batch_size to %d.",
            max_batch_size,
        )

    # Allow LEANN_EMBED_BATCH_SIZE env override (any endpoint). Useful when a self-hosted
    # MLX/HF shim is memory-bound and the default 500-input batch causes metal::malloc
    # OOMs — e.g. bge-m3 on a 38 GB machine routinely blew past with batch=500.
    env_cap = os.environ.get("LEANN_EMBED_BATCH_SIZE")
    if env_cap:
        try:
            cap = int(env_cap)
            if cap > 0:
                max_batch_size = min(max_batch_size, cap)
                logger.info(
                    "LEANN_EMBED_BATCH_SIZE=%d → capping embedding batch_size to %d.",
                    cap,
                    max_batch_size,
                )
        except ValueError:
            logger.warning("Ignoring non-integer LEANN_EMBED_BATCH_SIZE=%r", env_cap)

    # if avg len is less than 1000, use the max batch size

    total_batches = (len(texts) + max_batch_size - 1) // max_batch_size
    logger.warning(
        "OpenAI embedding config: model=%s base_url=%s texts=%d batch_size=%d "
        "timeout_s=%.1f total_chars=%d avg_chars=%.1f max_chars=%d",
        model_name,
        resolved_base_url,
        len(texts),
        max_batch_size,
        openai_timeout,
        sum(text_lengths),
        avg_len,
        max(text_lengths) if text_lengths else 0,
    )

    try:
        from tqdm import tqdm

        batch_range = range(0, len(texts), max_batch_size)
        batch_iterator = tqdm(
            batch_range, desc="Computing embeddings", unit="batch", total=total_batches
        )
    except ImportError:
        # Fallback when tqdm is not available
        batch_iterator = range(0, len(texts), max_batch_size)

    for i in batch_iterator:
        batch_texts = texts[i : i + max_batch_size]
        batch_lengths = text_lengths[i : i + max_batch_size]
        batch_number = (i // max_batch_size) + 1
        batch_started = time.time()

        try:
            response = client.embeddings.create(model=model_name, input=batch_texts)
            batch_embeddings = [embedding.embedding for embedding in response.data]
            elapsed = time.time() - batch_started

            # Verify we got the expected number of embeddings
            if len(batch_embeddings) != len(batch_texts):
                logger.warning(
                    f"Expected {len(batch_texts)} embeddings but got {len(batch_embeddings)}"
                )

            # Only take the number of embeddings that match the batch size
            all_embeddings.extend(batch_embeddings[: len(batch_texts)])
            logger.warning(
                "OpenAI embedding batch %d/%d ok: offset=%d size=%d elapsed_s=%.2f "
                "total_chars=%d avg_chars=%.1f max_chars=%d",
                batch_number,
                total_batches,
                i,
                len(batch_texts),
                elapsed,
                sum(batch_lengths),
                (sum(batch_lengths) / len(batch_lengths)) if batch_lengths else 0.0,
                max(batch_lengths) if batch_lengths else 0,
            )
        except Exception as e:
            elapsed = time.time() - batch_started
            logger.error(
                "OpenAI embedding batch %d/%d failed: offset=%d size=%d elapsed_s=%.2f "
                "total_chars=%d max_chars=%d error=%r",
                batch_number,
                total_batches,
                i,
                len(batch_texts),
                elapsed,
                sum(batch_lengths),
                max(batch_lengths) if batch_lengths else 0,
                e,
            )
            raise

    embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")
    return embeddings


def compute_embeddings_mlx(chunks: list[str], model_name: str, batch_size: int = 16) -> np.ndarray:
    # TODO: @yichuan-w add progress bar only in build mode
    """Computes embeddings using an MLX model."""
    try:
        import mlx.core as mx
        from mlx_lm.utils import load
    except ImportError as e:
        raise RuntimeError(
            "MLX or related libraries not available. Install with: uv pip install mlx mlx-lm"
        ) from e

    logger.info(
        f"Computing embeddings for {len(chunks)} chunks using MLX model '{model_name}' with batch_size={batch_size}..."
    )

    # Cache MLX model and tokenizer
    cache_key = f"mlx_{model_name}"
    if cache_key in _model_cache:
        logger.info(f"Using cached MLX model: {model_name}")
        model, tokenizer = _model_cache[cache_key]
    else:
        logger.info(f"Loading and caching MLX model: {model_name}")
        model, tokenizer = load(model_name)
        _model_cache[cache_key] = (model, tokenizer)
        logger.info(f"MLX model cached: {cache_key}")

    # Clip to the model's real token limit BEFORE batching. MLX pads every row in
    # a batch up to the longest row (see max_length below), so one oversized chunk
    # would blow up Metal allocation (metal::malloc OOM). Truncation bounds the
    # per-row worst case; batch-size still bounds the row count.
    chunks = truncate_for_model(chunks, model_name)

    # Process chunks in batches with progress bar
    all_embeddings = []

    try:
        from tqdm import tqdm

        batch_iterator = tqdm(
            range(0, len(chunks), batch_size), desc="Computing embeddings", unit="batch"
        )
    except ImportError:
        batch_iterator = range(0, len(chunks), batch_size)

    for i in batch_iterator:
        batch_chunks = chunks[i : i + batch_size]

        # Tokenize all chunks in the batch
        batch_token_ids = []
        for chunk in batch_chunks:
            token_ids = tokenizer.encode(chunk)
            batch_token_ids.append(token_ids)

        # Pad sequences to the same length for batch processing
        max_length = max(len(ids) for ids in batch_token_ids)
        padded_token_ids = []
        for token_ids in batch_token_ids:
            # Pad with tokenizer.pad_token_id or 0
            padded = token_ids + [0] * (max_length - len(token_ids))
            padded_token_ids.append(padded)

        # Convert to MLX array with batch dimension
        input_ids = mx.array(padded_token_ids)

        # Get embeddings for the batch
        embeddings = model(input_ids)

        # Mean pooling for each sequence in the batch
        pooled = embeddings.mean(axis=1)  # Shape: (batch_size, hidden_size)

        # Convert batch embeddings to numpy
        for j in range(len(batch_chunks)):
            pooled_list = pooled[j].tolist()  # Convert to list
            pooled_numpy = np.array(pooled_list, dtype=np.float32)
            all_embeddings.append(pooled_numpy)

    # Stack numpy arrays
    return np.stack(all_embeddings)


def compute_embeddings_ollama(
    texts: list[str],
    model_name: str,
    is_build: bool = False,
    host: Optional[str] = None,
    provider_options: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    """
    Compute embeddings using Ollama API with true batch processing.

    Uses the /api/embed endpoint which supports batch inputs.
    Batch size: 32 for MPS/CPU, 128 for CUDA to optimize performance.

    Args:
        texts: List of texts to compute embeddings for
        model_name: Ollama model name (e.g., "nomic-embed-text", "mxbai-embed-large")
        is_build: Whether this is a build operation (shows progress bar)
        host: Ollama host URL (defaults to environment or http://localhost:11434)
        provider_options: Optional provider-specific options (e.g., prompt_template)

    Returns:
        Normalized embeddings array, shape: (len(texts), embedding_dim)
    """
    try:
        import requests
    except ImportError:
        raise ImportError(
            "The 'requests' library is required for Ollama embeddings. Install with: uv pip install requests"
        )

    if not texts:
        raise ValueError("Cannot compute embeddings for empty text list")

    resolved_host = resolve_ollama_host(host)

    logger.info(
        f"Computing embeddings for {len(texts)} texts using Ollama API, model: '{model_name}', host: '{resolved_host}'"
    )

    # Check if Ollama is running
    try:
        response = requests.get(f"{resolved_host}/api/version", timeout=5)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        error_msg = (
            f"❌ Could not connect to Ollama at {resolved_host}.\n\n"
            "Please ensure Ollama is running:\n"
            "  • macOS/Linux: ollama serve\n"
            "  • Windows: Make sure Ollama is running in the system tray\n\n"
            "Installation: https://ollama.com/download"
        )
        raise RuntimeError(error_msg)
    except Exception as e:
        raise RuntimeError(f"Unexpected error connecting to Ollama: {e}")

    # Check if model exists and provide helpful suggestions
    try:
        response = requests.get(f"{resolved_host}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json()
        model_names = [model["name"] for model in models.get("models", [])]

        # Filter for embedding models (models that support embeddings)
        embedding_models = []
        suggested_embedding_models = [
            "nomic-embed-text",
            "mxbai-embed-large",
            "bge-m3",
            "all-minilm",
            "snowflake-arctic-embed",
        ]

        for model in model_names:
            # Check if it's an embedding model (by name patterns or known models)
            base_name = model.split(":")[0]
            if any(emb in base_name for emb in ["embed", "bge", "minilm", "e5"]):
                embedding_models.append(model)

        # Check if model exists (handle versioned names) and resolve to full name
        resolved_model_name = None
        for name in model_names:
            # Exact match
            if model_name == name:
                resolved_model_name = name
                break
            # Match without version tag (use the versioned name)
            elif model_name == name.split(":")[0]:
                resolved_model_name = name
                break

        if not resolved_model_name:
            error_msg = f"❌ Model '{model_name}' not found in local Ollama.\n\n"

            # Suggest pulling the model
            error_msg += "📦 To install this embedding model:\n"
            error_msg += f"   ollama pull {model_name}\n\n"

            # Show available embedding models
            if embedding_models:
                error_msg += "✅ Available embedding models:\n"
                for model in embedding_models[:5]:
                    error_msg += f"   • {model}\n"
                if len(embedding_models) > 5:
                    error_msg += f"   ... and {len(embedding_models) - 5} more\n"
            else:
                error_msg += "💡 Popular embedding models to install:\n"
                for model in suggested_embedding_models[:3]:
                    error_msg += f"   • ollama pull {model}\n"

            error_msg += "\n📚 Browse more: https://ollama.com/library"
            raise ValueError(error_msg)

        # Use the resolved model name for all subsequent operations
        if resolved_model_name != model_name:
            logger.info(f"Resolved model name '{model_name}' to '{resolved_model_name}'")
        model_name = resolved_model_name

        # Verify the model supports embeddings by testing it with /api/embed
        try:
            test_response = requests.post(
                f"{resolved_host}/api/embed",
                json={"model": model_name, "input": "test"},
                timeout=10,
            )
            if test_response.status_code != 200:
                error_msg = (
                    f"⚠️ Model '{model_name}' exists but may not support embeddings.\n\n"
                    f"Please use an embedding model like:\n"
                )
                for model in suggested_embedding_models[:3]:
                    error_msg += f"   • {model}\n"
                raise ValueError(error_msg)
        except requests.exceptions.RequestException:
            # If test fails, continue anyway - model might still work
            pass

    except requests.exceptions.RequestException as e:
        logger.warning(f"Could not verify model existence: {e}")

    # Determine batch size based on device availability
    # Check for CUDA/MPS availability using torch if available
    batch_size = 32  # Default for MPS/CPU
    try:
        import torch

        if torch.cuda.is_available():
            batch_size = 128  # CUDA gets larger batch size
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            batch_size = 32  # MPS gets smaller batch size
    except ImportError:
        # If torch is not available, use conservative batch size
        batch_size = 32

    logger.info(f"Using batch size: {batch_size} for true batch processing")

    # Apply prompt template if provided
    provider_options = provider_options or {}
    # Priority: build_prompt_template (new format) > prompt_template (old format)
    prompt_template = provider_options.get("build_prompt_template") or provider_options.get(
        "prompt_template"
    )

    if prompt_template:
        logger.warning(f"Applying prompt template: '{prompt_template}'")
        texts = [f"{prompt_template}{text}" for text in texts]

    # Clip to the model's real token limit before batching (client-side guard).
    texts = truncate_for_model(texts, model_name, base_url=resolved_host)

    def get_batch_embeddings(batch_texts):
        """Get embeddings for a batch of texts using /api/embed endpoint."""
        max_retries = 3
        retry_count = 0

        # Texts are already truncated to token limit by the outer function
        while retry_count < max_retries:
            try:
                # Use /api/embed endpoint with "input" parameter for batch processing
                response = requests.post(
                    f"{resolved_host}/api/embed",
                    json={"model": model_name, "input": batch_texts},
                    timeout=60,  # Increased timeout for batch processing
                )
                response.raise_for_status()

                result = response.json()
                batch_embeddings = result.get("embeddings")

                if batch_embeddings is None:
                    raise ValueError("No embeddings returned from API")

                if not isinstance(batch_embeddings, list):
                    raise ValueError(f"Invalid embeddings format: {type(batch_embeddings)}")

                if len(batch_embeddings) != len(batch_texts):
                    raise ValueError(
                        f"Mismatch: requested {len(batch_texts)} embeddings, got {len(batch_embeddings)}"
                    )

                return batch_embeddings, []

            except requests.exceptions.Timeout:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.warning(f"Timeout for batch after {max_retries} retries")
                    return None, list(range(len(batch_texts)))

            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    # Enhanced error detection for token limit violations
                    error_msg = str(e).lower()
                    if "token" in error_msg and (
                        "limit" in error_msg or "exceed" in error_msg or "length" in error_msg
                    ):
                        logger.error(
                            f"Token limit exceeded for batch. Error: {e}. "
                            f"Consider reducing chunk sizes or check token truncation."
                        )
                    else:
                        logger.error(f"Failed to get embeddings for batch: {e}")
                    return None, list(range(len(batch_texts)))

        return None, list(range(len(batch_texts)))

    # Process texts in batches
    all_embeddings = []
    all_failed_indices = []

    # Setup progress bar if needed
    show_progress = is_build or len(texts) > 10
    try:
        if show_progress:
            from tqdm import tqdm
    except ImportError:
        show_progress = False

    # Process batches
    num_batches = (len(texts) + batch_size - 1) // batch_size

    if show_progress:
        batch_iterator = tqdm(range(num_batches), desc="Computing Ollama embeddings (batched)")
    else:
        batch_iterator = range(num_batches)

    for batch_idx in batch_iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        batch_embeddings, batch_failed = get_batch_embeddings(batch_texts)

        if batch_embeddings is not None:
            all_embeddings.extend(batch_embeddings)
        else:
            # Entire batch failed, add None placeholders
            all_embeddings.extend([None] * len(batch_texts))
            # Adjust failed indices to global indices
            global_failed = [start_idx + idx for idx in batch_failed]
            all_failed_indices.extend(global_failed)

    # Handle failed embeddings
    if all_failed_indices:
        if len(all_failed_indices) == len(texts):
            raise RuntimeError("Failed to compute any embeddings")

        logger.warning(
            f"Failed to compute embeddings for {len(all_failed_indices)}/{len(texts)} texts"
        )

        # Use zero embeddings as fallback for failed ones
        valid_embedding = next((e for e in all_embeddings if e is not None), None)
        if valid_embedding:
            embedding_dim = len(valid_embedding)
            for i, embedding in enumerate(all_embeddings):
                if embedding is None:
                    all_embeddings[i] = [0.0] * embedding_dim

    # Remove None values
    all_embeddings = [e for e in all_embeddings if e is not None]

    if not all_embeddings:
        raise RuntimeError("No valid embeddings were computed")

    # Validate embedding dimensions
    expected_dim = len(all_embeddings[0])
    inconsistent_dims = []
    for i, embedding in enumerate(all_embeddings):
        if len(embedding) != expected_dim:
            inconsistent_dims.append((i, len(embedding)))

    if inconsistent_dims:
        error_msg = f"Ollama returned inconsistent embedding dimensions. Expected {expected_dim}, but got:\n"
        for idx, dim in inconsistent_dims[:10]:  # Show first 10 inconsistent ones
            error_msg += f"  - Text {idx}: {dim} dimensions\n"
        if len(inconsistent_dims) > 10:
            error_msg += f"  ... and {len(inconsistent_dims) - 10} more\n"
        error_msg += f"\nThis is likely an Ollama API bug with model '{model_name}'. Please try:\n"
        error_msg += "1. Restart Ollama service: 'ollama serve'\n"
        error_msg += f"2. Re-pull the model: 'ollama pull {model_name}'\n"
        error_msg += (
            "3. Use sentence-transformers instead: --embedding-mode sentence-transformers\n"
        )
        error_msg += "4. Report this issue to Ollama: https://github.com/ollama/ollama/issues"
        raise ValueError(error_msg)

    # Convert to numpy array and normalize
    embeddings = np.array(all_embeddings, dtype=np.float32)

    # Normalize embeddings (L2 normalization)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)  # Add small epsilon to avoid division by zero

    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")

    return embeddings


def compute_embeddings_gemini(
    texts: list[str], model_name: str = "text-embedding-004", is_build: bool = False
) -> np.ndarray:
    """
    Compute embeddings using Google Gemini API.

    Args:
        texts: List of texts to compute embeddings for
        model_name: Gemini model name (default: "text-embedding-004")
        is_build: Whether this is a build operation (shows progress bar)

    Returns:
        Embeddings array, shape: (len(texts), embedding_dim)
    """
    try:
        import os

        import google.genai as genai
    except ImportError as e:
        raise ImportError(f"Google GenAI package not installed: {e}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    # Cache Gemini client
    cache_key = "gemini_client"
    if cache_key in _model_cache:
        client = _model_cache[cache_key]
    else:
        client = genai.Client(api_key=api_key)
        _model_cache[cache_key] = client
        logger.info("Gemini client cached")

    logger.info(
        f"Computing embeddings for {len(texts)} texts using Gemini API, model: '{model_name}'"
    )

    # Gemini supports batch embedding
    max_batch_size = 100  # Conservative batch size for Gemini
    all_embeddings = []

    try:
        from tqdm import tqdm

        total_batches = (len(texts) + max_batch_size - 1) // max_batch_size
        batch_range = range(0, len(texts), max_batch_size)
        batch_iterator = tqdm(
            batch_range, desc="Computing embeddings", unit="batch", total=total_batches
        )
    except ImportError:
        # Fallback when tqdm is not available
        batch_iterator = range(0, len(texts), max_batch_size)

    for i in batch_iterator:
        batch_texts = texts[i : i + max_batch_size]

        try:
            # Use the embed_content method from the new Google GenAI SDK
            response = client.models.embed_content(
                model=model_name,
                contents=batch_texts,
                config=genai.types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"  # For document embedding
                ),
            )

            # Extract embeddings from response
            for embedding_data in response.embeddings:
                all_embeddings.append(embedding_data.values)
        except Exception as e:
            logger.error(f"Batch {i} failed: {e}")
            raise

    embeddings = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")

    return embeddings
