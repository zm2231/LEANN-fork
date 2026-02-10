# Architecture & Concepts

Understanding the technology that makes LEANN the smallest vector index in the world.

## Selective Recomputation

Traditional vector databases store every embedding (heavy floating-point vectors) in RAM or on disk. For 60 million chunks, this can take over 200GB.

LEANN achieves **97% storage savings** by storing only the raw text and a lightweight relational graph. During a search, it recomputes only the necessary embeddings on-the-fly. This shifts the burden from storage to compute, which is ideal for modern laptops.

## AST-Aware Chunking

Standard RAG systems break code into chunks based on fixed line counts. This often splits a function in the middle, destroying the context for the LLM.

LEANN's **Abstract Syntax Tree (AST)** parser understands code structure. It identifies function and class boundaries, ensuring that the retriever always returns a complete, logical block of code.

## Backend Selection: HNSW vs. DiskANN

- **HNSW**: Best for small to medium indexes (< 500k chunks). It is extremely fast but keeps the index structure in RAM.
- **DiskANN**: Optimized for V-Scale data. It uses memory-mapped files to handle millions of chunks without crashing your laptop's memory.

## The Docling Pipeline

For PDF processing, LEANN uses a multi-layer fallback system to ensure 100% reliability:
1. **PyMuPDF**: Fast, high-quality layout extraction.
2. **pypdf/pdfplumber**: Secondary fallbacks for specialized structures.
3. **Docling (IBM Research)**: The ultimate fallback using OCR and multimodal models for scanned or highly complex documents.
