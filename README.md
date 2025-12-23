# Korean Medicine RAG & Embedding Optimization (한의학 특화 RAG 및 임베딩 최적화)
This project explores methodologies to improve embedding performance in Oriental Medicine, addressing issues like **token overlap bias** and **data scarcity**. It validates these methods through an optimized RAG system evaluated on the **Korean National Licensing Exam for Korean Medicine**.

> **Reference**: Inspired by research from Gachon University (Prof. Kim Chang-eop) on the limitations of LLMs in Oriental Medicine.

## 🚀 Key Achievements (Performance)
The optimized model (**Contextual FT2+FT1 Hybrid Reranked**) significantly outperformed baselines.

### 1. Retrieval Performance (Sasang Medicine)
* **Hit Rate (@k=3)**: **0.90**
* **MRR**: 0.8417
* **NDCG**: 0.8881

### 2. Generation Performance (Accuracy)
| Model Setup | Sasang Exam (In-Domain) | Shanghan Lun Exam (Out-of-Domain) |
| :--- | :---: | :---: |
| GPT-4o-mini (No RAG) | 28% | - |
| Base Dense RAG | 50% | 9 / 32 |
| **Optimized RAG** | **56%** (w/ Evidence) | **14 / 32** (FT2-FT1 Sequential) |

---

## 🛠️ Methodology

### 1. Fine-tuning Strategy (Embedding)
We employed `e5-small-ko` (planning to upgrade to `bge-m3`) with two distinct datasets to overcome semantic limitations.
* **FT1 (Synthetic Q&A)**: Generated synthetic questions from chunks to learn retrieval tasks (Address Data Scarcity).
* **FT2 (Contrastive Learning)**: Aligned original Hanja text with Korean translations to capture deep semantic meaning beyond token overlap.
* **Training Note**: Used **`CachedMultipleNegativesRankingLoss`** to maximize effective batch size and in-batch negatives under GPU memory constraints (Colab Pro).

### 2. RAG Pipeline
* **Chunking**: Section-based (500-700 chars) with **Summary & Constitutional Metadata** injection.
* **Retrieval**: **Hybrid Search** (BM25 0.5 + Dense 0.5).
* **Reranking**: `BAAI/bge-reranker-v2-m3` (Top-N: 5).

### 3. Generalization Test
* **In-Domain**: Trained on *Dongyi Suse Bowon*, tested on *Sasang* exams (Strictly excluding answer sections from training to prevent leakage).
* **Out-of-Domain**: Tested on *Shanghan Lun* (completely unseen text) to verify the model's ability to generalize to other medical texts.

---

## 📂 File Structure & Datasets

* **`untitled53_ft1_ft2.ipynb`**:
    * Main training notebook implementing the Sequential Fine-tuning strategy.
    * Includes the implementation of `CachedMultipleNegativesRankingLoss`.

* **`동의수세보원 ft2 데이터.pdf`** (Contrastive Dataset):
    * **Content**: Pairs of **Original Hanja** and **Korean Translations**.
    * **Purpose**: For **FT2**, teaching the model to map diverse terminologies (Hanja/Korean) to the same semantic space.

* **`동의수세보원 해석텍스트만.pdf`** (Corpus):
    * **Content**: Pure Korean interpretation text.
    * **Purpose**: Source text for Knowledge Injection and RAG chunking.

---

## ⚠️ Limitations & Future Work
* **Test Set Size**: Currently evaluated on ~20 high-quality manually curated Q&A pairs. Expanding to 100+ pairs is planned for statistical robustness.
* **Sparse Encoder (Splade)**: Investigating **Splade** (recently updated in `sentence-transformers`) to better handle "vocabulary mismatch" where terms differ but meanings align, as an alternative to dense embedding alignment.
