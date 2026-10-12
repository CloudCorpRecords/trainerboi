import os
import json
import numpy as np
import pandas as pd

os.makedirs("data/samples", exist_ok=True)

# 1. Customer Churn Tabular Dataset
print("--> Generating sample_customer_churn.csv...")
np.random.seed(42)
n_samples = 2000
age = np.random.randint(18, 75, n_samples)
tenure_months = np.random.randint(1, 72, n_samples)
monthly_spend = np.round(np.random.uniform(20.0, 150.0, n_samples), 2)
support_calls = np.random.poisson(lam=1.8, size=n_samples)
contract_type = np.random.choice(["month-to-month", "one-year", "two-year"], size=n_samples, p=[0.5, 0.3, 0.2])
has_fiber = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
payment_method = np.random.choice(["credit_card", "bank_transfer", "electronic_check"], size=n_samples)

# Calculate realistic churn probability
risk = (
    (support_calls * 0.4) +
    (monthly_spend / 100.0) -
    (tenure_months / 30.0) +
    (1.2 * (contract_type == "month-to-month")) +
    np.random.normal(0, 0.5, n_samples)
)
churn = (risk > 1.0).astype(int)

df = pd.DataFrame({
    "age": age,
    "tenure_months": tenure_months,
    "monthly_spend": monthly_spend,
    "support_calls": support_calls,
    "contract_type": contract_type,
    "has_fiber": has_fiber,
    "payment_method": payment_method,
    "churn": churn
})
df.to_csv("data/samples/customer_churn.csv", index=False)
print("Saved data/samples/customer_churn.csv")

# 2. General Instruction Chat Dataset (JSONL)
print("--> Generating sample_chat_qa.jsonl...")
chat_samples = [
    {
        "messages": [
            {"role": "user", "content": "What is the difference between supervised and unsupervised learning?"},
            {"role": "assistant", "content": "In supervised learning, models train on input data paired with correct target labels (e.g. classification or regression). In unsupervised learning, the model discovers hidden patterns, clusters, or structures directly from unlabeled data (e.g. k-means clustering, dimensionality reduction)."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How does LoRA work in fine-tuning large language models?"},
            {"role": "assistant", "content": "LoRA (Low-Rank Adaptation) freezes the pretrained model weights and injects trainable rank-decomposition matrices into each transformer layer. This dramatically reduces the number of trainable parameters (often by 99%+) and VRAM usage while maintaining full model accuracy."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is mixed precision training and why should I use BF16?"},
            {"role": "assistant", "content": "Mixed precision training uses 16-bit floating point formats (FP16 or BF16) instead of 32-bit floats. BF16 (Bfloat16) matches the dynamic exponent range of FP32, preventing underflow/overflow errors without needing loss scaling. It doubles training speed and halves VRAM usage."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Explain gradient descent in simple terms."},
            {"role": "assistant", "content": "Imagine you are standing on a foggy mountain trying to reach the lowest valley. You cannot see the entire landscape, but you can feel the slope of the ground beneath your feet. Gradient descent takes small steps downhill (opposite to the gradient of the loss) until it reaches the lowest point."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What does learning rate mean in machine learning?"},
            {"role": "assistant", "content": "The learning rate is a hyperparameter that controls how much the model updates its weights with each training step. A learning rate that is too high causes training to diverge or oscillate wildly, while a rate that is too low causes training to progress too slowly."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Write a clean Python function to calculate the Fibonacci series up to n numbers."},
            {"role": "assistant", "content": "Here is a Python function:\n\n```python\ndef fibonacci(n: int) -> list[int]:\n    if n <= 0:\n        return []\n    if n == 1:\n        return [0]\n    seq = [0, 1]\n    while len(seq) < n:\n        seq.append(seq[-1] + seq[-2])\n    return seq\n```"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is overfitting and how do I prevent it?"},
            {"role": "assistant", "content": "Overfitting happens when a model learns the noise and exact training examples by heart rather than generalizing to unseen data. You can prevent it with regularization (L1/L2), dropout, data augmentation, early stopping, and using validation sets."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is an epoch in deep learning?"},
            {"role": "assistant", "content": "One epoch means the neural network has processed every single training example in the dataset exactly once. Training typically runs for multiple epochs (e.g., 3 to 10) so the model has enough opportunities to refine its weights."}
        ]
    }
]

with open("data/samples/sample_chat_qa.jsonl", "w", encoding="utf-8") as f:
    for item in chat_samples:
        f.write(json.dumps(item) + "\n")
print("Saved data/samples/sample_chat_qa.jsonl")
