# Review Intelligence Platform
### An AI-Powered Platform for Customer Review Analytics and Aspect-Based Sentiment Intelligence

## Overview

The **Review Intelligence Platform** is an end-to-end data analytics and NLP project designed to transform large volumes of customer reviews into meaningful business insights.

Online customer reviews contain valuable information about product quality, customer satisfaction, and purchasing experiences. However, manually analyzing thousands of reviews to identify customer opinions and recurring product issues is time-consuming.

This project addresses that challenge by combining machine learning, transformer-based NLP models, aspect-based sentiment analysis, and interactive data visualization.

The platform analyzes customer reviews at two levels:

- **Review-Level Sentiment Analysis:** Classifies an entire review as Positive, Neutral, or Negative.
- **Aspect-Based Sentiment Analysis (ABSA):** Identifies specific product aspects mentioned in reviews and determines the sentiment associated with each aspect.

The results are presented through an interactive Streamlit dashboard, allowing users to explore sentiment patterns, compare product categories, investigate customer concerns, and analyze individual reviews.

## Key Features

- **Review Sentiment Classification:** Classifies customer reviews into positive, neutral, and negative sentiment categories.
- **Transformer-Based NLP:** Uses a fine-tuned DistilBERT model for contextual sentiment classification.
- **Traditional ML Benchmark:** Implements TF-IDF with Logistic Regression as a baseline model.
- **Aspect Extraction:** Identifies product-related aspects using a combination of curated vocabulary, spaCy PhraseMatcher, and noun-chunk extraction.
- **Aspect-Level Sentiment Analysis:** Applies a pretrained DeBERTa-based ABSA model to identify sentiment associated with individual aspects.
- **Interactive Analytics Dashboard:** Explores sentiment trends, model performance, aspect distributions, review engagement, and individual customer feedback.
- **Category-Level Comparison:** Compares customer sentiment across Electronics, Beauty and Personal Care, and Home and Kitchen.
- **AI-Assisted Insights:** Generates rule-based analytical observations from review and aspect-level data.
- **Live Review Analysis:** Allows users to submit review text and explore model predictions.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Storage | Parquet, CSV, JSON |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Sentiment Classification | DistilBERT |
| Aspect-Based Sentiment Analysis | DeBERTa-based ABSA model |
| Text Processing | spaCy |
| Data Visualization | Plotly |
| Dashboard | Streamlit |
| Development Environment | VS Code, Google Colab |

## Dataset

This project uses the **Amazon Reviews 2023 dataset** released by the McAuley Lab.

The analysis focuses on three product categories:

1. Electronics
2. Beauty and Personal Care
3. Home and Kitchen

The selected dataset contains approximately **135.2 million reviews** across these three categories.

To make the initial experimentation and model development computationally manageable, representative samples were extracted from each category using reservoir sampling.

### Dataset Preparation

- Extracted 50,000 reviews from each product category.
- Combined the samples into a dataset containing 149,998 reviews.
- Applied sentiment labels based on the numerical ratings.
- Prepared stratified training, validation, and testing splits.

### Sentiment Labeling Strategy

| Rating | Sentiment |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

**Important:** These labels are derived from customer ratings rather than manually annotated review text. Therefore, they represent weak supervision and may not always reflect the actual sentiment expressed in a review.

The raw dataset is not included in this repository because of its size. Users should obtain the dataset independently from the original source.

## Project Architecture

The project follows a modular structure that separates data processing, model development, aspect analysis, and dashboard implementation.

```text
review-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── representative/
│       ├── splits/
│       └── absa/
│
├── notebooks/
│
├── src/
│   ├── aspect_extraction.py
│   ├── absa_sentiment.py
│   ├── run_absa_pilot.py
│   ├── enrich_absa_results.py
│   ├── analyze_absa_pilot.py
│   └── test_aspect_extraction_regression.py
│
├── app/
│   └── dashboard.py
│
├── configs/
│
├── models/
│
├── reports/
│   ├── figures/
│   ├── baseline_test_results.json
│   ├── distilbert_test_results.json
│   └── distilbert_training_history.json
│
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: The structure above highlights the main project components. Some directories contain additional intermediate files and artifacts.*

## Methodology

### 1. Data Preprocessing

The raw Amazon review data was processed to create a structured dataset suitable for machine learning experiments.

The preprocessing workflow included:

- Extracting review records from compressed JSONL files.
- Selecting representative samples from the three product categories.
- Handling missing and unsuitable review records.
- Converting numerical ratings into sentiment labels.
- Creating stratified training, validation, and testing datasets.
- Saving processed datasets in Parquet format.

The final sampled dataset contains 149,998 reviews.

### 2. Review-Level Sentiment Classification

Two machine learning approaches were implemented and evaluated.

**Model 1: TF-IDF + Logistic Regression**

A traditional machine learning baseline was developed using:

- TF-IDF vectorization for text representation.
- Logistic Regression for sentiment classification.
- Class weighting to address class imbalance.

**Model 2: Fine-Tuned DistilBERT**

A transformer-based model was fine-tuned using the `distilbert/distilbert-base-uncased` pretrained checkpoint.

Key training configurations:

| Parameter | Configuration |
|---|---|
| Pretrained Model | DistilBERT Base Uncased |
| Maximum Sequence Length | 256 |
| Number of Classes | 3 |
| Learning Rate | 2e-5 |
| Training Epochs | 3 |
| Batch Size | 16 |
| Gradient Accumulation | 2 |
| Mixed Precision | FP16 |
| Loss Function | Weighted Cross-Entropy |
| Model Selection | Validation Macro F1 |

Training was conducted using Google Colab with a Tesla T4 GPU. The best checkpoint was selected based on validation macro F1.

### 3. Aspect-Based Sentiment Analysis

Review-level sentiment classification provides an overall opinion of a product, but it does not necessarily explain which product features customers appreciate or criticize.

To address this limitation, an Aspect-Based Sentiment Analysis pipeline was developed.

The pipeline consists of two main components.

**Aspect Extraction**

The aspect extraction module uses:

- A curated vocabulary of product-related aspects.
- spaCy PhraseMatcher for identifying known aspect expressions.
- Noun-chunk extraction for discovering additional candidate aspects.
- Contextual rules to improve the identification of relevant phrases.

**Aspect-Level Sentiment Classification**

The extracted aspects are analyzed using the pretrained model:

`yangheng/deberta-v3-base-absa-v1.1`

The model predicts sentiment associated with an aspect in its review context.

For example:

| Customer Review | Extracted Aspect | Predicted Sentiment |
|---|---|---|
| The battery life is terrible. | Battery Life | Negative |
| The camera quality is excellent. | Camera Quality | Positive |
| The speakers sound awful. | Speakers | Negative |

The ABSA pipeline was evaluated through a 300-review pilot sample, containing 100 reviews from each category.

The pilot generated 251 aspect-level predictions from 148 reviews.

**Important:** The aspect extraction and sentiment predictions have not been validated against a manually annotated benchmark. Model confidence scores should not be interpreted as measured prediction accuracy.

## Model Evaluation

The two review-level sentiment classification models were evaluated on the same held-out test set.

### Performance Comparison

| Evaluation Metric | TF-IDF + Logistic Regression | Fine-Tuned DistilBERT |
|---|---:|---:|
| Accuracy | 87.40% | 91.35% |
| Macro Precision | 70.57% | 76.24% |
| Macro Recall | 76.97% | 79.09% |
| Macro F1-Score | 72.93% | 77.58% |
| Weighted F1-Score | 88.48% | 91.58% |

### Class-Wise F1-Score

| Sentiment Class | TF-IDF + Logistic Regression | Fine-Tuned DistilBERT |
|---|---:|---:|
| Negative | 0.80 | 0.86 |
| Neutral | 0.44 | 0.50 |
| Positive | 0.94 | 0.97 |

### Key Observations

- DistilBERT outperformed the traditional baseline across all reported evaluation metrics.
- The transformer model achieved a test accuracy of 91.35% and a macro F1-score of 77.58%.
- Positive reviews were classified more effectively than negative and neutral reviews.
- Neutral sentiment remained the most challenging class for both models.
- The results demonstrate the benefit of contextual language representations for this classification task.

The evaluation results are based on weakly supervised rating-derived labels. They should not be interpreted as equivalent to performance against human-annotated sentiment labels.

## Interactive Dashboard

The Streamlit dashboard provides an interactive interface for exploring the results of the analysis.

### Dashboard Sections

**1. Executive Overview**

Provides a high-level summary of review activity, sentiment distribution, category performance, and important analytical observations.

**2. Sentiment Analytics**

Organized into three analytical tabs:

- **Model Evaluation:** Compares TF-IDF and DistilBERT performance using evaluation metrics and class-level results.
- **Training Insights:** Explores training progression, validation performance, loss curves, and confusion matrix results.
- **Aspect Analytics:** Examines aspect-level sentiment distributions, category comparisons, and prediction confidence.

**3. Aspect Intelligence**

Explores extracted product aspects, their sentiment distributions, and recurring customer concerns.

**4. Product Comparison**

Provides product-level analytical views where sufficient review data is available. Product-level comparisons should be interpreted cautiously because the pilot contains very few reviews per product.

**5. Review Engagement**

Explores review verification status, helpful votes, and engagement-related patterns.

**6. Review Explorer**

Allows users to filter and inspect individual customer reviews and their associated sentiment predictions.

**7. AI Insights**

Presents rule-based observations derived from the available review and aspect analytics.

**8. Live AI Analysis**

Allows users to enter review text and obtain sentiment and aspect-level predictions.

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- Git
- pip

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd review-intelligence
```

Replace the placeholder with your repository URL.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Data

Download the required Amazon Reviews 2023 data independently and place the raw files in the appropriate directories under:

```text
data/raw/
```

Prepare the processed datasets and model artifacts according to the project pipeline.

The dashboard requires the processed review datasets, ABSA outputs, and evaluation artifacts to be available in their expected project locations.

### 5. Run the Dashboard

From the project root directory:

```bash
streamlit run app/dashboard.py
```

The dashboard will open in your browser.

## Repository Artifacts

The project uses several generated artifacts for evaluation and dashboard visualization.

Examples include:

- `reports/baseline_test_results.json`
- `reports/distilbert_test_results.json`
- `reports/distilbert_training_history.json`
- `reports/figures/distilbert_test_confusion_matrix.png`
- `data/processed/representative/all_categories_sample.parquet`
- `data/processed/absa/absa_pilot_results_enriched.parquet`
- `data/processed/absa/absa_pilot_summary_enriched.csv`

Large datasets, model weights, and other generated artifacts are excluded from version control to keep the repository manageable.

## Limitations and Future Improvements

Although the project demonstrates an end-to-end customer review analytics workflow, several limitations remain.

### Current Limitations

- **Weak Supervision:** Sentiment labels are derived from numerical ratings rather than manual text annotation.
- **Neutral Sentiment Classification:** Neutral reviews remain comparatively difficult to classify.
- **Limited ABSA Validation:** The aspect extraction and sentiment pipeline has not been evaluated against a manually annotated benchmark.
- **Small ABSA Pilot:** Aspect-level analysis is based on 300 sampled reviews and should not be generalized to the entire Amazon review population.
- **Product-Level Coverage:** Most products in the pilot have only one associated review, limiting reliable product-level comparisons.
- **Sampling Strategy:** Equal category sampling supports category comparisons but does not represent the original category proportions.
- **Generalization:** The random train/test split does not establish performance on entirely unseen product categories.

### Future Improvements

- Develop a manually annotated dataset for more reliable ABSA evaluation.
- Improve neutral sentiment classification through targeted data collection and model experimentation.
- Expand aspect extraction using more advanced linguistic and contextual methods.
- Evaluate model generalization across unseen product categories.
- Increase the ABSA sample size to support more reliable category-level insights.
- Introduce temporal sentiment analysis to investigate changes in customer satisfaction.
- Explore deployment and model-serving optimizations for larger-scale inference.

## Key Takeaways

This project demonstrates how machine learning and NLP can be combined with data analytics to extract actionable information from customer reviews.

It brings together traditional machine learning, transformer-based sentiment classification, aspect extraction, aspect-level sentiment analysis, and interactive visualization in a single analytical platform.

Beyond predicting whether a review is positive or negative, the platform aims to provide a more detailed understanding of **what customers like, what they dislike, and which product features influence their experiences.**

---

## Author

**Md. Zobayer**

Computer Science and Engineering Graduate  
Interests: Data Science, Machine Learning, Natural Language Processing, and Business Analytics.

[GitHub](<YOUR_GITHUB_PROFILE_URL>) · [LinkedIn](<YOUR_LINKEDIN_PROFILE_URL>)