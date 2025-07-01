Topic Modelling on AI Act (AI-ACT)

A Python-based framework for exploring topic modeling techniques—LDA, BERT embeddings, and language detection—tailored to analyzing documents related to the EU Artificial Intelligence Act (AI-ACT).
🔍 Overview

This project provides tools to:

    Perform topic modeling using classical LDA and contextual embeddings (BERT).

    Perform language detection.

    Integrate these methods in a unified workflow, orchestrated via main.py.

Key Features

    LDA (LDA.py): Unsupervised modeling using Latent Dirichlet Allocation.

    BERT-based embeddings (bertEx.py): Extract contextual embeddings followed by clustering, dimensionality reduction, or topic labeling.

    Language detection (language.py): Identify the language of each document.

    CLI workflow (main.py): End-to-end script combining preprocessing, modeling, and output generation.

🛠 Installation

    Clone the repo:

git clone https://github.com/Coalessence/Topic-Modelling-on-AI-ACT-main.git
cd Topic-Modelling-on-AI-ACT-main

Install dependencies:

    pip install -r requirements.txt

🚀 Quick Start

    Prepare your corpus
    Place text files in a source folder (e.g., data/) – each file is treated as one document.

    Run the full pipeline:

    python main.py --input_dir data/ --output_dir results/ --num_topics 10

    This will:

        Detect document language.

        Generate LDA topics.

        Compute BERT embeddings and cluster them.

        Save outputs (topics, clusters, embeddings) to results/.

    Explore Outputs

        LDA results: topic word distributions, document-topic assignments.

        BERT results: embeddings saved as NumPy/CSV, plus cluster labels for downstream analysis.

📚 Module Reference
Script	Purpose
LDA.py	Implements LDA topic modeling using gensim; customizable topic count.
bertEx.py	Extracts BERT embeddings and enables clustering/clustering analysis.
language.py	Detects and labels document languages using langdetect.
main.py	Orchestrator: loads docs, runs analysis modules, and saves results.
requirements.txt	Python package versions required for reproducibility.
⚙️ Configuration Options

Pass parameters via CLI or modify defaults in main.py:

    --num_topics: Number of LDA topics (default: 10).

    --input_dir, --output_dir: I/O directories.

    Other options can include embedding size, clustering type (e.g., KMeans), and language filtering. Customize as needed.

🚧 Extend & Contribute

    Add preprocessing support (e.g., stopword removal, tokenization).

    Integrate visualization modules (e.g., pyLDAvis, t-SNE/UMAP scatterplots).

    Support new models: Non-negative Matrix Factorization (NMF), BERTopic, LSA.

    Refactor into a modular package with clear CLI/API separation.

Contributions welcome! Feel free to open issues or submit pull requests.
📝 License

This project is MIT-licensed. Please see LICENSE for full details.
📧 Contact

For questions or collaboration, feel free to open an issue or contact the maintainer via GitHub.
