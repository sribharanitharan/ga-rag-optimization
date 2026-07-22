# 🧬 Genetic Algorithm for RAG Pipeline Hyperparameter Optimization

An interactive meta-optimization framework that uses a **Genetic Algorithm (GA)** to search and optimize hyperparameters for a local **Retrieval-Augmented Generation (RAG)** pipeline. Built for *Assignment III — Meta-Heuristic Optimization Techniques*, this system aims to maximize answer quality (using semantic similarity evaluation) under strict computational budget constraints.

---

## 🚀 Key Features

* **Local RAG Integration**: Powered by **LangChain**, **FAISS** (vector database), **SentenceTransformers** (`all-MiniLM-L6-v2` for embeddings and similarity scoring), and local LLMs via **Ollama** (supports `phi3`, `tinyllama`, `llama3.2`).
* **Robust Metaheuristic Logic**:
  * **Chromosomal Encoding**: Optimizes 4 dimensions: `chunk_size` (G1), `chunk_overlap` (G2), `temperature` (G3), and `top_k` (G4).
  * **Operators**: Selection (Tournament Selection), Crossover (Uniform Crossover), and Mutation (Gaussian / random walk with clipping boundary checks).
  * **Elitism**: Always carries the best-performing individual to the next generation.
* **Streamlit Interactive Dashboard**:
  * **Live Tracking**: Displays real-time logging, metrics (Best Fitness, LLM Call budget, generations done), and process state.
  * **Data Visualization**: Real-time plotting of the GA convergence curve and the normalized parameter evolution over generations.
* **Optimization Analytics**: Generates comparison dataframes, parameter evolution analysis, landscape analysis, efficiency evaluations, and domain impact studies.
* **Smart Budgeting & Caching**: Employs a local dictionary cache to prevent redundant evaluations and enforce the maximum LLM call budget.

---

## 📂 Directory Structure

```text
assignment_3/
├── Assignment_III_Documentation.pdf   # Technical documentation/research report
├── README.md                          # Repository documentation
└── ga_rag_project/                    # Application source code
    ├── ga_rag_optimization.py         # Main Streamlit dashboard & GA logic
    ├── requirements.txt               # Project dependencies
    ├── gold_qa.json                   # Gold standard Q&A validation set
    ├── document.pdf                   # Target document to feed into the RAG system
    └── results/                       # Outputs directory
        ├── convergence.png            # Saved convergence plot
        └── ga_results.csv             # Optimization metrics across generations
```

---

## ⚙️ Hyperparameters Search Space

The GA searches across the following dimensions:
1. **`chunk_size` (G1)**: range: `[100, 1000]` characters (determines context block size).
2. **`chunk_overlap` (G2)**: range: `[0, 200]` characters (helps preserve context continuity across splits).
3. **`temperature` (G3)**: range: `[0.0, 1.0]` (controls LLM generation creativity/determinism).
4. **`top_k` (G4)**: range: `[1, 10]` (number of retrieved source documents).

**Fitness Scoring**: Evaluates the RAG system output against the `gold_qa.json` reference set using semantic cosine similarity of sentence embeddings. An invalid parameter penalty (e.g. `chunk_overlap >= chunk_size`) is applied to guide the optimizer away from poor configurations.

---

## 🛠️ Setup & Running

### 1. Prerequisites

* **Python 3.10+**
* **Ollama** installed and running on your system.
* Make sure you pull one of the supported local models (e.g. `phi3` or `tinyllama`):
  ```bash
  ollama pull phi3
  ```
  *(or `tinyllama`, or `llama3.2` depending on your model preference)*

### 2. Installation

Clone this repository, navigate to the `ga_rag_project` folder, and install the required Python packages:

```bash
cd ga_rag_project
pip install -r requirements.txt
```

### 3. Run the Dashboard

To launch the interactive dashboard, run:

```bash
streamlit run ga_rag_optimization.py
```

Open your browser at the local URL provided (usually `http://localhost:8501`).

---

## 📈 Running the Optimization

1. **Configure Parameters**: Use the sidebar to set the Ollama model, population size, generations count, mutation/crossover rates, and search bounds.
2. **Launch**: Click **"🚀 Start GA Optimization"**.
3. **View Real-Time Data**:
   * Watch the **Live Log** box to track evaluations and cache hits.
   * View the **Convergence Chart** and **Parameter Evolution** plots update dynamically.
4. **Export Results**: Once the run completes, download the optimization run history (`ga_results.csv`) and execution logs directly from the dashboard.
