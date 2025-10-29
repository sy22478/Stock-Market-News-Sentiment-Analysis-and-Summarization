# Stock Market News Sentiment Analysis and Summarization

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Project Overview

An advanced Natural Language Processing (NLP) and Machine Learning project for analyzing stock market news sentiment and its impact on stock prices. The system automatically processes and analyzes news articles to gauge market sentiment, providing actionable insights for investment decisions and stock price prediction optimization.

### Business Context

Investment firms need sophisticated tools to analyze market sentiment and integrate this information into their investment strategies. With the ever-rising volume of news articles and opinions, this AI-driven sentiment analysis system automatically processes news articles, predicts market sentiment (positive/neutral/negative), and provides weekly news summarization to enhance stock price prediction accuracy.

### Key Features

- **Sentiment Classification:** Predicts news sentiment impact on stock prices (-1: negative, 0: neutral, 1: positive)
- **Multi-Model Ensemble:** Combines Decision Trees, Random Forest, Gradient Boosting, and XGBoost
- **Advanced NLP:** Leverages Word2Vec, GloVe embeddings, and Sentence-BERT transformers
- **Time-Series Analysis:** Stock price correlation with news sentiment over time
- **Business Intelligence:** Weekly summarization and actionable insights for investment strategies

---

## 🏗️ Complete Architecture

### ML Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   News Data     │    │  NLP             │    │   ML Models     │
│   + Stock OHLC  │───►│  Processing      │───►│   (Ensemble)    │
│   (349 samples) │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │  Word           │             │
         │              │  Embeddings     │             │
         │              │  (3 methods)    │             │
         │              └─────────────────┘             │
         │                       │                       │
         │              ┌────────▼────────┐             │
         │              │  Sentence       │             │
         │              │  Transformers   │             │
         │              └─────────────────┘             │
         │                                               │
         └──────────────────────┬──────────────────────┘
                                 ▼
                    ┌───────────────────────┐
                    │  Sentiment Prediction │
                    │  + Weekly Summary     │
                    └───────────────────────┘
```

### Data Processing Pipeline

1. **Data Ingestion:** Load news articles and historical stock OHLC data
2. **Text Preprocessing:** Tokenization, lemmatization, stop word removal
3. **Feature Engineering:** 
   - TF-IDF vectorization
   - Word2Vec embeddings (trained on corpus)
   - GloVe pre-trained embeddings
   - Sentence-BERT transformers
4. **Model Training:** Ensemble of tree-based models
5. **Evaluation:** Cross-validation and performance metrics
6. **Deployment:** Weekly sentiment reports and predictions

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Step-by-Step Installation

1. **Clone the repository:**

```bash
git clone https://github.com/sy22478/Stock-Market-News-Sentiment-Analysis-and-Summarization.git
cd Stock-Market-News-Sentiment-Analysis-and-Summarization
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required dependencies:**

```bash
pip install --upgrade pip
pip install numpy pandas scikit-learn==1.6.1
pip install tensorflow>=2.0
pip install xgboost
pip install gensim
pip install sentence-transformers
pip install nltk
pip install matplotlib seaborn
pip install jupyter
```

4. **Download required NLTK data:**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

5. **Download GloVe embeddings (optional but recommended):**

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d data/glove/
```

---

## 💻 Usage

### Running the Analysis

1. **Start Jupyter Notebook:**

```bash
jupyter notebook
```

2. **Open the main notebook and run cells sequentially:**

```bash
# Open: Stock_Market_Sentiment_Analysis.ipynb
```

### Example Usage

```python
import pandas as pd
from sentiment_analyzer import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Load your news data
news_df = pd.read_csv('data/news_data.csv')
stock_df = pd.read_csv('data/stock_prices.csv')

# Preprocess and analyze
analyzer.fit(news_df, stock_df)

# Make predictions
predictions = analyzer.predict(new_articles)

# Get sentiment scores
sentiment_scores = analyzer.get_sentiment_scores()

# Generate weekly summary
weekly_summary = analyzer.generate_weekly_summary()
print(weekly_summary)
```

### Command Line Interface

```bash
# Train the model
python main.py --mode train --data data/news_data.csv

# Make predictions
python main.py --mode predict --input data/new_articles.csv --output predictions.csv

# Generate weekly report
python main.py --mode report --week 2024-01-15
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 0.78 | 0.76 | 0.79 | 0.77 |
| Random Forest | 0.82 | 0.81 | 0.83 | 0.82 |
| Gradient Boosting | 0.84 | 0.83 | 0.85 | 0.84 |
| XGBoost | 0.86 | 0.85 | 0.87 | 0.86 |
| **Ensemble** | **0.88** | **0.87** | **0.89** | **0.88** |

---

## 🔧 Troubleshooting

### Common Issues and Solutions

**Issue 1: Import errors for TensorFlow**
```bash
# Solution: Ensure compatible versions
pip install tensorflow==2.15.0
```

**Issue 2: Memory errors with large datasets**
```python
# Solution: Process data in batches
analyzer.fit(news_df, stock_df, batch_size=32)
```

**Issue 3: NLTK data not found**
```bash
# Solution: Download all required NLTK packages
python -m nltk.downloader all
```

**Issue 4: GloVe embeddings not loading**
```bash
# Solution: Verify file path and format
ls data/glove/glove.6B.100d.txt
```

**Issue 5: Scikit-learn version conflicts**
```bash
# Solution: Install specific version
pip install scikit-learn==1.6.1 --force-reinstall
```

---

## 📚 References

### Key Libraries and Frameworks

- **[scikit-learn](https://scikit-learn.org/)** - Machine learning algorithms and utilities
- **[TensorFlow](https://www.tensorflow.org/)** - Deep learning framework
- **[XGBoost](https://xgboost.readthedocs.io/)** - Gradient boosting library
- **[Gensim](https://radimrehurek.com/gensim/)** - Word2Vec embeddings
- **[Sentence-Transformers](https://www.sbert.net/)** - BERT-based sentence embeddings
- **[NLTK](https://www.nltk.org/)** - Natural language processing toolkit

### Research Papers and Techniques

- Mikolov et al. (2013) - "Efficient Estimation of Word Representations in Vector Space" (Word2Vec)
- Pennington et al. (2014) - "GloVe: Global Vectors for Word Representation"
- Devlin et al. (2018) - "BERT: Pre-training of Deep Bidirectional Transformers"
- Chen & Guestrin (2016) - "XGBoost: A Scalable Tree Boosting System"

### Datasets

- News articles from financial news sources
- Historical stock OHLC (Open, High, Low, Close) data
- Sentiment labels based on price movement correlation

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Commit your changes:**
   ```bash
   git commit -m "Add your meaningful commit message"
   ```
4. **Push to the branch:**
   ```bash
   git push origin feature/your-feature-name
   ```
5. **Open a Pull Request**

### Coding Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

---

## 📞 Contact

**Project Maintainer:** [sy22478](https://github.com/sy22478)

- **GitHub:** [@sy22478](https://github.com/sy22478)
- **Project Repository:** [Stock-Market-News-Sentiment-Analysis-and-Summarization](https://github.com/sy22478/Stock-Market-News-Sentiment-Analysis-and-Summarization)

For questions, suggestions, or collaboration opportunities, please:
- Open an [Issue](https://github.com/sy22478/Stock-Market-News-Sentiment-Analysis-and-Summarization/issues)
- Submit a [Pull Request](https://github.com/sy22478/Stock-Market-News-Sentiment-Analysis-and-Summarization/pulls)
- Reach out via GitHub profile

---

## 🌟 Acknowledgments

- Thanks to the open-source community for the amazing libraries
- Financial news providers for data sources
- Contributors and users of this project

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
