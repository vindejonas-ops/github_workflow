"""
Session 400: 情感分析模型对比
对比 Baseline (DistilBERT) vs Improved (RoBERTa)
"""

from transformers import pipeline
import time

# ========== 模型配置 ==========
MODELS = {
    "baseline": {
        "name": "distilbert-base-uncased-finetuned-sst-2-english",
        "description": "DistilBERT (small, 66M params)"
    },
    "improved": {
        "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "description": "RoBERTa-base (larger, 125M params, trained on 124M tweets)"
    }
}

def load_classifier(model_name):
    """加载模型"""
    print(f"Loading {model_name}...")
    return pipeline(
        task="sentiment-analysis",
        model=model_name,
        tokenizer=model_name,
        framework="pt"
    )

def predict_binary(classifier, text):
    """
    统一为二分类输出 (positive/negative)
    RoBERTa 输出3类，neutral 映射为 negative
    """
    result = classifier(text)[0]
    label = result['label'].lower()
    
    # 标签映射
    if label == 'neutral':
        return 'negative'  # 保守策略：中性视为负面
    return label

def evaluate(classifier, texts, labels, model_key):
    """评估准确率"""
    correct = 0
    start = time.time()
    
    for text, true_label in zip(texts, labels):
        if model_key == "baseline":
            # Baseline 直接输出大写标签
            result = classifier(text)[0]
            pred = result['label'].lower()
        else:
            # Improved 需要映射
            pred = predict_binary(classifier, text)
        
        if pred == true_label.lower():
            correct += 1
    
    accuracy = correct / len(texts) * 100
    elapsed = time.time() - start
    
    return {
        "accuracy": accuracy,
        "time": elapsed,
        "correct": correct,
        "total": len(texts)
    }

# ========== 测试数据（前50条评论示例，替换为你的真实数据）==========
TEST_TEXTS = [
    "This movie is absolutely fantastic! Best acting I've seen.",
    "Terrible film. Complete waste of time and money.",
    "The plot was okay but the ending was disappointing.",
    "A masterpiece of modern cinema. Highly recommended!",
    "Boring and predictable. I fell asleep halfway through.",
    # ... 继续添加直到50条
]

TEST_LABELS = [
    "positive", "negative", "negative", "positive", "negative",
    # ... 对应50条标签
]

# ========== 主程序 ==========
if __name__ == "__main__":
    print("=" * 70)
    print("Session 400: Sentiment Analysis Model Comparison")
    print("=" * 70)
    
    results = {}
    
    for model_key, config in MODELS.items():
        print(f"\n>>> Testing: {config['description']}")
        print(f"Model: {config['name']}")
        
        # 加载模型
        classifier = load_classifier(config['name'])
        
        # 评估
        result = evaluate(classifier, TEST_TEXTS, TEST_LABELS, model_key)
        results[model_key] = result
        
        print(f"Accuracy: {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")
        print(f"Time: {result['time']:.2f}s")
    
    # ========== 输出对比结果 ==========
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    baseline_acc = results['baseline']['accuracy']
    improved_acc = results['improved']['accuracy']
    improvement = improved_acc - baseline_acc
    
    print(f"Baseline (DistilBERT):  {baseline_acc:.2f}%")
    print(f"Improved (RoBERTa):     {improved_acc:.2f}%")
    print(f"Improvement:            +{improvement:.2f}%")
    
    print("\n" + "=" * 70)
    print("FINDINGS FOR WECHAT GROUP:")
    print("=" * 70)
    print(f"Model: cardiffnlp/twitter-roberta-base-sentiment-latest")
    print(f"Architecture: RoBERTa-base (125M parameters)")
    print(f"Training Data: 124M tweets, fine-tuned on TweetEval")
    print(f"Accuracy: {improved_acc:.2f}% (vs {baseline_acc:.2f}% baseline)")
    print(f"Status: {'BETTER' if improvement > 0 else 'WORSE'} than baseline")
    print("=" * 70)