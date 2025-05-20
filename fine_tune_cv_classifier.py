import pandas as pd
import torch
import shutil
import os
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_recall_fscore_support
import torch.nn as nn
import nltk
import random
from nltk.corpus import stopwords


try:
    stopwords.words('turkish')
except:
    nltk.download('stopwords')
    nltk.download('punkt')

print("🚀 CV Sınıflandırıcı Eğitim Başlatılıyor...")


try:
    df = pd.read_csv("toplu_cv_analiz.csv", encoding="utf-8")
    print(f"✅ Veri başarıyla yüklendi. Toplam {len(df)} örnek bulundu.")
except Exception as e:
    print(f"❌ Veri yüklenirken hata: {e}")
    raise e

if "is_trained" in df.columns:
    untrained_count = len(df[df["is_trained"] != 1])
    df = df[df["is_trained"] != 1]
    print(f"📊 {untrained_count} eğitilmemiş örnek seçildi.")
else:
    print("⚠️ 'is_trained' sütunu bulunamadı, tüm veriler kullanılacak.")

label_cols = [col for col in df.columns if col not in ['dosya', 'cümle', 'is_trained']]

print(f"🏷️ Etiketler: {label_cols}")

print("\n📊 Etiket dağılımı:")
for label in label_cols:
    if label in df.columns:
        positive_count = sum(df[label] > 0.5)
        print(f"  {label}: {positive_count}/{len(df)} ({positive_count/len(df)*100:.2f}%)")

class_weights = []
for label in label_cols:
    if label in df.columns:
        pos = sum(df[label])
        neg = len(df) - pos
        # Dengesizlik durumunda ağırlık hesapla (pozitif)
        weight = neg / pos if pos > 0 else 1.0
        class_weights.append(weight)
    else:
        class_weights.append(1.0)
        print(f"⚠️ '{label}' sütunu veride bulunamadı!")

print(f"⚖️ Sınıf ağırlıkları: {[round(w, 2) for w in class_weights]}")

#tür dönüşm
for col in label_cols:
    if col in df.columns:
        if df[col].isnull().any():
            print(f"⚠️ '{col}' sütununda NaN bulundu. 0 ile dolduruluyor.")
            df[col].fillna(0, inplace=True)
        df[col] = df[col].astype(float)

#birleştir
df["labels"] = df[label_cols].values.tolist()
df.dropna(subset=['cümle'], inplace=True)
df = df[["cümle", "labels"]].rename(columns={"cümle": "text"})

#dataup
print("\n🔄 Veri artırma işlemi başlatılıyor...")

try:
    turkish_stopwords = set(stopwords.words('turkish'))
except:
    turkish_stopwords = set()
    print("⚠️ Türkçe stopwords yüklenemedi.")
#randomize
def augment_text(text):
    words = text.split()
    if len(words) <= 3:
        return text
        
    
    if random.random() < 0.3:
        dropout_ratio = random.uniform(0.1, 0.2)
        words = [w for w in words if random.random() > dropout_ratio or w.lower() in turkish_stopwords]
    
    
    if random.random() < 0.2:
        chunks = [words[i:i+3] for i in range(0, len(words), 3)]
        random.shuffle(chunks)
        words = [word for chunk in chunks for word in chunk]
    
    return " ".join(words)

augmented_texts = []
augmented_labels = []

#3ornek
minority_threshold = max(3, len(df) * 0.2)

for label_idx, label in enumerate(label_cols):
    if label in df.columns:
       
        minority_samples = df[df.apply(lambda row: row["labels"][label_idx] > 0.5, axis=1)]
        
        if len(minority_samples) < minority_threshold:
            print(f"🔍 '{label}' etiketi için veri artırma yapılıyor ({len(minority_samples)} → ~{min(len(minority_samples)*3, minority_threshold)})")

            for _, row in minority_samples.iterrows():
                augmentation_count = min(3, int(minority_threshold / len(minority_samples)) + 1)
                for _ in range(augmentation_count):
                    augmented_text = augment_text(row['text'])
                    augmented_texts.append(augmented_text)
                    augmented_labels.append(row['labels'])

#ornek ekle
if augmented_texts:
    aug_df = pd.DataFrame({"text": augmented_texts, "labels": augmented_labels})
    df = pd.concat([df, aug_df], ignore_index=True)
    print(f"✅ Veri artırma sonrası toplam örnek sayısı: {len(df)}")
else:
    print("ℹ️ Veri artırmaya gerek duyulmadı.")


full_dataset = Dataset.from_pandas(df)

print("\n🤖 Model hazırlanıyor...")

#token
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = BertTokenizerFast.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

full_dataset = full_dataset.map(tokenize, batched=True)

#egitim
split = full_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"📚 Eğitim seti: {len(train_dataset)} örnek")
print(f"🔍 Değerlendirme seti: {len(eval_dataset)} örnek")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

#tanılama
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)

#loss
class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.weights = torch.tensor(weights)
        
    def forward(self, logits, targets):
        loss = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
        weighted_loss = loss * self.weights.to(logits.device).view(1, -1)
        return weighted_loss.mean()

#egitimsınıf
class MultiLabelTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            loss_fct = WeightedBCEWithLogitsLoss(self.class_weights)
        else:
            loss_fct = nn.BCEWithLogitsLoss()
            
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

#metrik
def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    
    #0.5
    thresholds = np.arange(0.3, 0.7, 0.05)
    best_f1s = []
    best_thresholds = []
    
    for i in range(probs.shape[1]):
        f1s = []
        for threshold in thresholds:
            y_pred = (probs[:, i] >= threshold).int().numpy()
            y_true = (labels[:, i] > 0.5).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1s.append(f1)
        
        best_idx = np.argmax(f1s)
        best_f1s.append(f1s[best_idx])
        best_thresholds.append(thresholds[best_idx])
    
    y_pred = np.zeros_like(probs.numpy())
    for i in range(probs.shape[1]):
        y_pred[:, i] = (probs[:, i] >= best_thresholds[i]).int().numpy()
    
    y_true = (labels > 0.5).astype(int)
    
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    #hesap
    class_stats = {}
    for i, label in enumerate(label_cols):
        class_stats[f"{label}_f1"] = f1_score(y_true=y_true[:, i], y_pred=y_pred[:, i], zero_division=0)
        class_stats[f"{label}_precision"] = precision_recall_fscore_support(y_true=y_true[:, i], y_pred=y_pred[:, i], average='binary', zero_division=0)[0]
        class_stats[f"{label}_recall"] = precision_recall_fscore_support(y_true=y_true[:, i], y_pred=y_pred[:, i], average='binary', zero_division=0)[1]
        class_stats[f"{label}_thresh"] = best_thresholds[i]
    
    #kayıt
    try:
        os.makedirs("cv_classifier_model", exist_ok=True)
        with open("cv_classifier_model/thresholds.txt", "w", encoding="utf-8") as f:
            for i, label in enumerate(label_cols):
                f.write(f"{label}:{best_thresholds[i]}\n")
    except Exception as e:
        print(f"Eşik değerleri kaydedilirken hata: {e}")
    
    metrics = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'accuracy': accuracy,
        **class_stats
    }  
    return metrics

print("\n⚙️ Eğitim parametreleri ayarlanıyor...")

#GPU
gpu_available = torch.cuda.is_available()
print(f"🖥️ GPU Kullanılabilir: {gpu_available}")

training_args = TrainingArguments(
    output_dir="./cv_classifier_model",
    eval_strategy="epoch",
    learning_rate=2e-5, 
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    fp16=gpu_available,
    report_to="none"
)

print("\n🏋️ Eğitim başlatılıyor...")

trainer = MultiLabelTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
    class_weights=class_weights
)
try:
    trainer.train()
    print("✅ Eğitim başarıyla tamamlandı.")
except Exception as e:
    print(f"❌ Eğitim sırasında bir hata oluştu: {e}")
    raise e

print("\n💾 En iyi model kaydediliyor...")
trainer.save_model("./cv_classifier_model")
tokenizer.save_pretrained("./cv_classifier_model")

with open("./cv_classifier_model/labels.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(label_cols))

print("✅ Model, tokenizer ve etiketler başarıyla kaydedildi: ./cv_classifier_model")

print("\n📊 Model değerlendiriliyor...")
eval_results = trainer.evaluate()

print("\n📈 Değerlendirme Sonuçları:")
for key, value in eval_results.items():
    print(f"  {key}: {value:.4f}")

#threshold
print("\n🎯 Optimum Eşik Değerleri:")
for i, label in enumerate(label_cols):
    key = f"{label}_thresh"
    if key in eval_results:
        print(f"  {label}: {eval_results[key]:.2f}")

print("\n🧹 Temizlik işlemleri yapılıyor...")

#checkpoint
output_dir = training_args.output_dir
for subfolder in os.listdir(output_dir):
    if subfolder.startswith("checkpoint-"):
        shutil.rmtree(os.path.join(output_dir, subfolder), ignore_errors=True)

print("✅ Tüm ara checkpoint klasörleri silindi.")

#update
try:
    full_df = pd.read_csv("toplu_cv_analiz.csv", encoding="utf-8")
    mask = full_df["is_trained"] != 1
    full_df.loc[mask, "is_trained"] = 1
    full_df.to_csv("toplu_cv_analiz.csv", index=False, encoding="utf-8")
    print("✅ Eğitim sonrası is_trained sütunu güncellendi.")
except Exception as e:
    print(f"❌ is_trained güncellenirken hata: {e}")

print("\n🎉 İşlem tamamlandı. Model kullanıma hazır: ./cv_classifier_model")