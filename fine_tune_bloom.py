"""
fine_tune_bloom.py — BloomBERT-style fine-tuning for Mongolian classroom data.

BloomBERT (https://github.com/RyanLauQF/BloomBERT) яг ижил architecture:
  - Transformer backbone (DistilBERT → бид XLM-RoBERTa ашиглана, Монгол дэмжихийн тулд)
  - Linear classification head (dropout → linear → 6 класс)
  - End-to-end fine-tuning (бүх жинг хамт сургана)

Зарчмын ялгаа vs одоогийн ml_classify.py:
  - ml_classify.py:   XLM-R embeddings → Logistic Regression (2 алхамт, хөлдөөсөн BERT)
  - энэ файл:         XLM-R + head → end-to-end gradient descent (BloomBERT ижил)

Ажиллуулах:
  pip install transformers torch scikit-learn
  python fine_tune_bloom.py --epochs 10 --output bloom_mongolian_model

Сургасны дараа ml_classify.py-д ачааллах:
  BLOOM_BERT_MODEL = './bloom_mongolian_model'
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

# ── Training data (ml_classify.py-аас авна) ───────────────────────────────────
from ml_classify import BLOOM_TRAIN, BLOOM_KEYWORDS, _preprocess

LABEL2ID = {'Q1': 0, 'Q2': 1, 'Q3': 2, 'Q4': 3, 'Q5': 4, 'Q6': 5}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 6


# ── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset(extra_json=None):
    """Build (texts, labels) from BLOOM_TRAIN + optional extra JSON file.

    extra_json: autolabel_cache.json-ийн замыг өгвөл нэмж авна.
    """
    texts, labels = [], []
    extra = {}
    if extra_json and Path(extra_json).exists():
        with open(extra_json, encoding='utf-8') as f:
            extra = json.load(f)

    for cls, examples in BLOOM_TRAIN.items():
        for t in examples:
            texts.append(t)
            labels.append(LABEL2ID[cls])
        for t in extra.get(cls, []):
            texts.append(t)
            labels.append(LABEL2ID[cls])

    print(f"[dataset] Нийт {len(texts)} жишээ, "
          f"{sum(1 for _ in extra.values() if _)} auto-label нэмсэн")
    return texts, labels


# ── BloomBERT Architecture ────────────────────────────────────────────────────

def build_model(model_name='xlm-roberta-base', num_labels=NUM_LABELS):
    """BloomBERT-ийн яг ижил: Transformer + linear head.

    BloomBERT код (TensorFlow):
        base_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        x = base_model(input_ids, attention_mask=attention_mask)[0][:,0,:]
        x = Dropout(0.3)(x)
        output = Dense(num_labels, activation='softmax')(x)

    Бид PyTorch + XLM-RoBERTa ашиглана (Монгол дэмжихийн тулд).
    """
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError:
        raise ImportError("pip install transformers torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        hidden_dropout_prob=0.3,       # BloomBERT: Dropout(0.3)
        attention_probs_dropout_prob=0.3,
        ignore_mismatched_sizes=True,
    )
    return model


# ── Training loop (BloomBERT style: 50 epochs → бид early stopping нэмнэ) ────

def train(
    model_name='xlm-roberta-base',
    output_dir='bloom_mongolian_model',
    epochs=50,          # BloomBERT: 50 epochs
    batch_size=16,
    lr=2e-5,
    val_split=0.15,
    extra_json='autolabel_cache.json',
    seed=42,
):
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader, random_split
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup
        from torch.optim import AdamW
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError as e:
        raise ImportError(f"pip install transformers torch scikit-learn — {e}")

    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    texts, labels = build_dataset(extra_json)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    class BloomDataset(Dataset):
        def __init__(self, texts, labels, tok, max_len=128):
            self.enc = tok(texts, padding=True, truncation=True,
                           max_length=max_len, return_tensors='pt')
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self): return len(self.labels)

        def __getitem__(self, i):
            return {k: v[i] for k, v in self.enc.items()}, self.labels[i]

    dataset = BloomDataset(texts, labels, tokenizer)

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # ── BloomBERT: 50 epochs, best val acc checkpoint ─────────────────────────
    best_val_acc = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for batch, batch_labels in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_labels = batch_labels.to(device)
            out = model(**batch, labels=batch_labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_loss += loss.item()

        # Validate
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for batch, batch_labels in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds_all.extend(logits.argmax(-1).cpu().tolist())
                labels_all.extend(batch_labels.tolist())

        val_acc = accuracy_score(labels_all, preds_all)
        avg_loss = train_loss / len(train_loader)
        history.append({'epoch': epoch, 'loss': round(avg_loss, 4), 'val_acc': round(val_acc, 4)})

        print(f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | val_acc={val_acc:.4f}"
              + (" ← BEST" if val_acc > best_val_acc else ""))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    # ── Final report ──────────────────────────────────────────────────────────
    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch, batch_labels in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds_all.extend(logits.argmax(-1).cpu().tolist())
            labels_all.extend(batch_labels.tolist())

    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(labels_all, preds_all, target_names=label_names))
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir}/")

    # Save training history
    with open(f"{output_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    return best_val_acc


# ── Integration helper ────────────────────────────────────────────────────────

def update_ml_classify(model_dir='bloom_mongolian_model'):
    """Сургасан загварыг ml_classify.py-д ашиглахаар тохируулна.

    ml_classify.py-д BLOOM_BERT_MODEL = './bloom_mongolian_model' гэж өөрчлөнө.
    """
    ml_path = Path(__file__).parent / 'ml_classify.py'
    content = ml_path.read_text(encoding='utf-8')
    old = "BLOOM_BERT_MODEL = 'xlm-roberta-base'"
    new = f"BLOOM_BERT_MODEL = './{model_dir}'"
    if old in content:
        ml_path.write_text(content.replace(old, new), encoding='utf-8')
        print(f"✅ ml_classify.py шинэчлэгдлээ: BLOOM_BERT_MODEL = './{model_dir}'")
    else:
        print(f"⚠️  ml_classify.py-д '{old}' олдсонгүй. Гараар шинэчилнэ үү.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='BloomBERT-style fine-tuning for Mongolian Bloom taxonomy classification'
    )
    parser.add_argument('--model',   default='xlm-roberta-base',
                        help='HuggingFace model name (default: xlm-roberta-base)')
    parser.add_argument('--output',  default='bloom_mongolian_model',
                        help='Output directory for fine-tuned model')
    parser.add_argument('--epochs',  type=int, default=50,
                        help='Training epochs (BloomBERT: 50)')
    parser.add_argument('--batch',   type=int, default=16)
    parser.add_argument('--lr',      type=float, default=2e-5)
    parser.add_argument('--extra',   default='autolabel_cache.json',
                        help='Path to autolabel_cache.json for extra training data')
    parser.add_argument('--update-config', action='store_true',
                        help='After training, update BLOOM_BERT_MODEL in ml_classify.py')
    args = parser.parse_args()

    print("═" * 60)
    print("  BloomBERT-style Fine-Tuning — Монгол ангийн өрөөний өгөгдөл")
    print("═" * 60)
    print(f"  Backbone:  {args.model}")
    print(f"  Epochs:    {args.epochs} (BloomBERT ижил)")
    print(f"  Output:    {args.output}/")
    print("═" * 60)

    best_acc = train(
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        extra_json=args.extra,
    )

    if args.update_config:
        update_ml_classify(args.output)

    print(f"\n🎉 Дууслаа! Best val accuracy: {best_acc:.4f}")
    print(f"\nДараагийн алхам:")
    print(f"  ml_classify.py-д: BLOOM_BERT_MODEL = './{args.output}'")
    print(f"  эсвэл: python fine_tune_bloom.py --update-config")
