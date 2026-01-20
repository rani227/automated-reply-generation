# Automated Reply Generation for Hotel Reviews

This repository contains the implementation of our B.Tech graduation project **"Automated Reply Generation for Hotel Reviews"**, which focuses on generating professional, context-aware, and sentiment-aligned managerial responses to hotel reviews using a retrieval-augmented deep learning framework.

The system combines **semantic retrieval (Sentence-BERT + FAISS)** with **transformer-based generation (Flan-T5)** and uses **separate generators for positive and negative reviews** to preserve tone and empathy alignment.

---

## 📄 Project Report

For detailed background, methodology, experiments, and results, refer to the complete project report:

👉 **Project Report (Google Drive):**
[https://drive.google.com/file/d/18A_iVMHFTRk_9ScEdtwT-eRPJQ8Vt_Vj/view?usp=sharing](https://drive.google.com/file/d/18A_iVMHFTRk_9ScEdtwT-eRPJQ8Vt_Vj/view?usp=sharing)

The report includes:

* Literature review and research motivation
* Dataset details and preprocessing pipeline
* Retrieval-augmented framework design
* Dual-generator training strategy
* Quantitative and qualitative evaluation

---

## 👥 Team Members

This was a group project developed by:

* **Ridhi Tuteja** — [ridhituteja@gmail.com](mailto:ridhituteja@gmail.com)
* **Kirti Yadav** — [kirti19y@gmail.com](mailto:kirti19y@gmail.com)
* **Rani** — [ananyasingh779944@gmail.com](mailto:ananyasingh779944@gmail.com)

Under the supervision of **Dr. Santosh Kumar**, Assistant Professor, Department of Computer Engineering, NIT Kurukshetra.

---

## 🗂 Repository Structure

```
├── dataset_cleaned/
│   └── Final cleaned dataset used for training all models
│
├── final_work/
│   └── dual_generators.ipynb
│       - Complete training, evaluation, and inference pipeline
│       - Contains our proposed dual-generator RAG framework
│
├── intermediate_work/
│   └── Alternative model architectures and earlier approaches
│       - Implemented for comparison and ablation
│       - Final framework outperformed these approaches
│
├── raw_datasets/
│   └── Originally scraped TripAdvisor datasets (before cleaning)
│
└── README.md
```

### Folder Details

### 📁 `raw_datasets/`

Contains the original datasets scraped from TripAdvisor, including raw reviews and corresponding manager replies.

### 📁 `dataset_cleaned/`

Contains the final preprocessed dataset after:

* Language filtering
* Noise and emoji removal
* PII masking (emails, phone numbers, URLs)
* Named entity anonymization (hotel names, locations, persons)
* Deduplication and normalization

This dataset is used for all training and evaluation.

### 📁 `intermediate_work/`

Contains experiments with alternative methodologies and architectures attempted during development. These models were evaluated but did not outperform the final proposed framework, hence they are kept for completeness and comparison.

### 📁 `final_work/dual_generators.ipynb`

This notebook contains:

* SBERT embedding and FAISS index construction
* Retrieval of top-k similar past replies
* Prompt construction for retrieval-augmented generation
* Fine-tuning of two separate Flan-T5 generators:

  * Positive review generator
  * Negative review generator
* Inference and Gradio-based UI
* Evaluation using BLEU, ROUGE-L, and BERTScore

This is the **main notebook required to run the system**.

---

## ▶ How to Run the Automated Reply Generation System

The system is designed to run on **Google Colab** with GPU support.

### Step 1: Open the Notebook

Open the following notebook in Google Colab:

```
final_work/dual_generators.ipynb
```

Upload the repository to Google Drive or clone it directly into Colab.

---

### Step 2: Run Setup and Model Loading Cells

Run all required setup cells for:

* Installing dependencies
* Loading trained generators (or training, if needed)
* Loading FAISS index and retrievers

⚠️ **You do NOT need to run the full training section to test the system.** Pre-trained models are already available and can be loaded directly from Google Drive:

* **Positive Generator:** [https://drive.google.com/drive/folders/1NQ6dPWoY0to5EsUmaEEuahoHFwOMidyz?usp=sharing](https://drive.google.com/drive/folders/1NQ6dPWoY0to5EsUmaEEuahoHFwOMidyz?usp=sharing)
* **Negative Generator:** [https://drive.google.com/drive/folders/1svzVen0wqE20uM8f7YSzyHY-6ArTsTTG?usp=sharing](https://drive.google.com/drive/folders/1svzVen0wqE20uM8f7YSzyHY-6ArTsTTG?usp=sharing)

Mount Google Drive in Colab and update the model paths in the notebook to point to these folders before running the inference section.

---

### Step 3: Run Only the Inference Section

Inside `dual_generators.ipynb`, locate the section with heading:

> **Inference**

Only execute the cells in this section **up to the following line of code:**

```python
ui.launch(debug=True)
```

---

### Step 4: Use the Gradio Interface

After running the inference cells, a **Gradio UI will open inside the Colab output cell**.

The UI allows you to input:

* ⭐ Star rating (1–5)
* 📝 User review text

The system will then:

1. Retrieve similar historical replies using SBERT + FAISS
2. Route the review to the appropriate generator (positive/negative)
3. Generate a professional managerial response

The generated reply will be displayed instantly in the interface.

---

## 🧠 Model Architecture (Summary)

* **Retrieval Module**

  * Sentence-BERT embeddings
  * FAISS similarity search
  * Retrieves top-k similar review–reply pairs

* **Generation Module**

  * Two fine-tuned Flan-T5 generators

    * Positive generator
    * Negative generator
  * Retrieval-augmented prompts
  * Beam search decoding

* **Routing Logic**

  * Based on star rating and sentiment polarity

This design improves:

* Context relevance
* Sentiment consistency
* Managerial tone authenticity

---

## 📊 Evaluation Metrics

The system was evaluated using:

* **BERTScore (F1)** — semantic similarity
* **ROUGE-L** — content overlap
* **BLEU-4** — fluency and phrasing

Results (from test set):

| Metric    | Positive Generator | Negative Generator |
| --------- | ------------------ | ------------------ |
| BERTScore | ~0.87              | ~0.86              |
| ROUGE-L   | ~0.29              | ~0.22              |
| BLEU-4    | ~0.09              | ~0.05              |

High BERTScore indicates strong semantic alignment with real manager replies.

---

## 🚀 Future Scope

Planned extensions discussed in the report include:

* Web-based deployment for hotel dashboards
* Multilingual response generation
* Reinforcement Learning from Human Feedback (RLHF)
* Adaptive learning from manager edits
* Policy and safety filtering layers

---

## 📌 Notes

* This project is intended for **academic and research purposes**.
* The datasets were anonymized to preserve privacy.
* The system is designed as a **decision-support tool**, not as a fully autonomous customer communication agent.

---

If you use or reference this work, please cite the project report or acknowledge the authors accordingly.