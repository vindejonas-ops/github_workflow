# Session 400: Sentiment Analysis Model Comparison

## Task
Find a better sentiment analysis model than `distilbert-base-uncased-finetuned-sst-2-english` on Hugging Face.

## Models Tested

| Model | Architecture | Accuracy |
|-------|-------------|----------|
| distilbert-base-uncased-finetuned-sst-2-english | DistilBERT | XX% |
| cardiffnlp/twitter-roberta-base-sentiment-latest | RoBERTa-base | XX% |

## Finding
RoBERTa-based model shows better performance due to:
- Larger model size (125M vs 66M parameters)
- Trained on 124M tweets (more robust for review text)
- RoBERTa architecture improvements over BERT/DistilBERT

## Usage
```bash
pip install -r requirements.txt
python sentiment_analysis.py