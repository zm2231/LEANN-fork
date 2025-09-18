"""
Enhanced chunking utilities with AST-aware code chunking support.
Packaged within leann-core so installed wheels can import it reliably.
"""

import logging
from pathlib import Path
from typing import Optional

from llama_index.core.node_parser import SentenceSplitter

logger = logging.getLogger(__name__)

# Code file extensions supported by astchunk
CODE_EXTENSIONS = {
    ".py": "python",
    ".java": "java",
    ".cs": "csharp",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "typescript",
    ".jsx": "typescript",
}


def detect_code_files(documents, code_extensions=None) -> tuple[list, list]:
    """Separate documents into code files and regular text files."""
    if code_extensions is None:
        code_extensions = CODE_EXTENSIONS

    code_docs = []
    text_docs = []

    for doc in documents:
        file_path = doc.metadata.get("file_path", "") or doc.metadata.get("file_name", "")
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            if file_ext in code_extensions:
                doc.metadata["language"] = code_extensions[file_ext]
                doc.metadata["is_code"] = True
                code_docs.append(doc)
            else:
                doc.metadata["is_code"] = False
                text_docs.append(doc)
        else:
            doc.metadata["is_code"] = False
            text_docs.append(doc)

    logger.info(f"Detected {len(code_docs)} code files and {len(text_docs)} text files")
    return code_docs, text_docs


def get_language_from_extension(file_path: str) -> Optional[str]:
    """Return language string from a filename/extension using CODE_EXTENSIONS."""
    ext = Path(file_path).suffix.lower()
    return CODE_EXTENSIONS.get(ext)


def create_ast_chunks(
    documents,
    max_chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadata_template: str = "default",
) -> list[str]:
    """Create AST-aware chunks from code documents using astchunk.

    Falls back to traditional chunking if astchunk is unavailable.
    """
    try:
        from astchunk import ASTChunkBuilder  # optional dependency
    except ImportError as e:
        logger.error(f"astchunk not available: {e}")
        logger.info("Falling back to traditional chunking for code files")
        return create_traditional_chunks(documents, max_chunk_size, chunk_overlap)

    all_chunks = []
    for doc in documents:
        language = doc.metadata.get("language")
        if not language:
            logger.warning("No language detected; falling back to traditional chunking")
            all_chunks.extend(create_traditional_chunks([doc], max_chunk_size, chunk_overlap))
            continue

        try:
            configs = {
                "max_chunk_size": max_chunk_size,
                "language": language,
                "metadata_template": metadata_template,
                "chunk_overlap": chunk_overlap if chunk_overlap > 0 else 0,
            }

            repo_metadata = {
                "file_path": doc.metadata.get("file_path", ""),
                "file_name": doc.metadata.get("file_name", ""),
                "creation_date": doc.metadata.get("creation_date", ""),
                "last_modified_date": doc.metadata.get("last_modified_date", ""),
            }
            configs["repo_level_metadata"] = repo_metadata

            chunk_builder = ASTChunkBuilder(**configs)
            code_content = doc.get_content()
            if not code_content or not code_content.strip():
                logger.warning("Empty code content, skipping")
                continue

            chunks = chunk_builder.chunkify(code_content)
            for chunk in chunks:
                if hasattr(chunk, "text"):
                    chunk_text = chunk.text
                elif isinstance(chunk, dict) and "text" in chunk:
                    chunk_text = chunk["text"]
                elif isinstance(chunk, str):
                    chunk_text = chunk
                else:
                    chunk_text = str(chunk)

                if chunk_text and chunk_text.strip():
                    all_chunks.append(chunk_text.strip())

            logger.info(
                f"Created {len(chunks)} AST chunks from {language} file: {doc.metadata.get('file_name', 'unknown')}"
            )
        except Exception as e:
            logger.warning(f"AST chunking failed for {language} file: {e}")
            logger.info("Falling back to traditional chunking")
            all_chunks.extend(create_traditional_chunks([doc], max_chunk_size, chunk_overlap))

    return all_chunks


def create_traditional_chunks(
    documents, chunk_size: int = 256, chunk_overlap: int = 128
) -> list[str]:
    """Create traditional text chunks using LlamaIndex SentenceSplitter."""
    if chunk_size <= 0:
        logger.warning(f"Invalid chunk_size={chunk_size}, using default value of 256")
        chunk_size = 256
    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 2

    node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator=" ",
        paragraph_separator="\n\n",
    )

    all_texts = []
    for doc in documents:
        try:
            nodes = node_parser.get_nodes_from_documents([doc])
            if nodes:
                all_texts.extend(node.get_content() for node in nodes)
        except Exception as e:
            logger.error(f"Traditional chunking failed for document: {e}")
            content = doc.get_content()
            if content and content.strip():
                all_texts.append(content.strip())

    return all_texts


def create_text_chunks(
    documents,
    chunk_size: int = 256,
    chunk_overlap: int = 128,
    use_ast_chunking: bool = False,
    ast_chunk_size: int = 512,
    ast_chunk_overlap: int = 64,
    code_file_extensions: Optional[list[str]] = None,
    ast_fallback_traditional: bool = True,
) -> list[str]:
    """Create text chunks from documents with optional AST support for code files."""
    if not documents:
        logger.warning("No documents provided for chunking")
        return []

    local_code_extensions = CODE_EXTENSIONS.copy()
    if code_file_extensions:
        ext_mapping = {
            ".py": "python",
            ".java": "java",
            ".cs": "c_sharp",
            ".ts": "typescript",
            ".tsx": "typescript",
        }
        for ext in code_file_extensions:
            if ext.lower() not in local_code_extensions:
                if ext.lower() in ext_mapping:
                    local_code_extensions[ext.lower()] = ext_mapping[ext.lower()]
                else:
                    logger.warning(f"Unsupported extension {ext}, will use traditional chunking")

    all_chunks = []
    if use_ast_chunking:
        code_docs, text_docs = detect_code_files(documents, local_code_extensions)
        if code_docs:
            try:
                all_chunks.extend(
                    create_ast_chunks(
                        code_docs, max_chunk_size=ast_chunk_size, chunk_overlap=ast_chunk_overlap
                    )
                )
            except Exception as e:
                logger.error(f"AST chunking failed: {e}")
                if ast_fallback_traditional:
                    all_chunks.extend(
                        create_traditional_chunks(code_docs, chunk_size, chunk_overlap)
                    )
                else:
                    raise
        if text_docs:
            all_chunks.extend(create_traditional_chunks(text_docs, chunk_size, chunk_overlap))
    else:
        all_chunks = create_traditional_chunks(documents, chunk_size, chunk_overlap)

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
