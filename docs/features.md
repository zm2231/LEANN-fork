# ✨ Detailed Features

## 🔥 Core Features

- **🔄 Real-time Embeddings** - Eliminate heavy embedding storage with dynamic computation using optimized ZMQ servers and highly optimized search paradigm (overlapping and batching) with highly optimized embedding engine
- **🧠 AST-Aware Code Chunking** - Intelligent code chunking that preserves semantic boundaries (functions, classes, methods) for Python, Java, C#, and TypeScript files
- **📈 Scalable Architecture** - Handles millions of documents on consumer hardware; the larger your dataset, the more LEANN can save
- **🎯 Graph Pruning** - Advanced techniques to minimize the storage overhead of vector search to a limited footprint
- **🏗️ Pluggable Backends** - HNSW/FAISS (default), with optional DiskANN for large-scale deployments

## 📱 Universal Ingestion & Formats

- **📑 Multi-Format Support**: Native handling of `.pdf`, `.txt`, `.md`, `.docx`, `.pptx`, `.xlsx`, and `.mm` (Mindmaps).
- **🚀 Advanced PDF Pipeline**: Intelligent fallback chain featuring [PyMuPDF](https://github.com/pymupdf/PyMuPDF), [pypdf](https://github.com/py-pdf/pypdf), [pdfplumber](https://github.com/jsvine/pdfplumber), and [**Docling OCR**](https://github.com/docling-project/docling) (IBM Research) for high-fidelity document parsing.
- **💼 Office Document Extractors**: Structural awareness for Word ([python-docx](https://github.com/python-openxml/python-docx)), Excel ([openpyxl](https://github.com/ericgazoni/openpyxl)), and PowerPoint ([python-pptx](https://github.com/python-openxml/python-pptx)) preserving tables, slides, and sheets.
- **🧠 Mindmap Parsing**: Hierarchical node extraction for [FreeMind](http://freemind.sourceforge.net/) and [Freeplane](https://www.freeplane.org/) (`.mm`) preserving semantic relationships in zettelkasten-like structures.
- **📱 Integrated Source Connectors**: Dedicated CLI commands for Apple Mail, Calendar, iMessage, WeChat, and more.

## ⚡ Performance & Scalability

- **🚀 Instant CLI Discovery**: Optimized `leann list` and `leann remove` commands use intelligent scanning and size caching to respond in milliseconds, regardless of the number of registered projects.
- **📈 Huge Vault Support**: Native **DiskANN** backend integration allows indexing terabytes of data (millions of chunks) without exceeding laptop RAM limits.
- **🧠 Advanced Code Analysis**: AST-aware chunking supports complex logic retrieval. For large functions (e.g., algorithmic trading strategies), using `--ast-chunk-size 1000` ensures that entire logical blocks stay in the same context for the LLM.

## 🛠️ Technical Highlights
- **🔄 Recompute Mode** - Highest accuracy scenarios while eliminating vector storage overhead
- **⚡ Zero-copy Operations** - Minimize IPC overhead by transferring distances instead of embeddings
- **🚀 High-throughput Embedding Pipeline** - Optimized batched processing for maximum efficiency
- **🎯 Two-level Search** - Novel coarse-to-fine search overlap for accelerated query processing (optional)
- **💾 Memory-mapped Indices** - Fast startup with raw text mapping to reduce memory overhead
- **🚀 MLX Support** - Ultra-fast recompute/build with quantized embedding models, accelerating building and search ([minimal example](../examples/mlx_demo.py))

## 🎨 Developer Experience

- **Simple Python API** - Get started in minutes
- **Extensible backend system** - Easy to add new algorithms
- **Comprehensive examples** - From basic usage to production deployment
