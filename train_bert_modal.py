import os
import modal

app = modal.App("bert-safety-training")
volume = modal.Volume.from_name("bert-safety-checkpoints", create_if_missing=True)

# Remote container image with PyTorch CUDA & HuggingFace Transformers
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "scikit-learn",
        "pandas",
        "numpy",
        "scipy",
        "accelerate"
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,                  # 2 hours — 3 configs × 2 epochs on 74k rows needs margin
    volumes={"/vol": volume},
)
def train_bert_remote(train_csv: str, val_csv: str, test_csv: str):
    import io
    import torch
    import numpy as np
    import pandas as pd
    from scipy.special import softmax
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    from transformers import (
        AutoTokenizer,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments
    )

    # Verify GPU availability before proceeding
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  WARNING: No GPU detected — training will be extremely slow on CPU!")

    print("🚀 [Modal GPU] Loading cleaned datasets into GPU memory...")
    train_df = pd.read_csv(io.StringIO(train_csv)).dropna(subset=["clean_prompt"])
    val_df = pd.read_csv(io.StringIO(val_csv)).dropna(subset=["clean_prompt"])
    test_df = pd.read_csv(io.StringIO(test_csv)).dropna(subset=["clean_prompt"])

    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    print(f"Dataset split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Tokenization
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    MAX_LEN_BERT = 128

    def encode_bert_texts(texts):
        return bert_tokenizer(
            texts.tolist(),
            padding=True,
            truncation=True,
            max_length=MAX_LEN_BERT,
            return_tensors="pt"
        )

    print("⚡ [Modal GPU] Tokenizing prompt sequences...")
    X_train_bert = encode_bert_texts(train_df["clean_prompt"])
    X_val_bert = encode_bert_texts(val_df["clean_prompt"])
    X_test_bert = encode_bert_texts(test_df["clean_prompt"])

    class WildGuardDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = WildGuardDataset(X_train_bert, y_train)
    val_dataset = WildGuardDataset(X_val_bert, y_val)
    test_dataset = WildGuardDataset(X_test_bert, y_test)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"macro_f1": f1_score(labels, predictions, average="macro")}

    # 3 Hyperparameter Configurations per course guidelines
    bert_configs = [
        {"learning_rate": 2e-5, "batch_size": 16},
        {"learning_rate": 3e-5, "batch_size": 32},
        {"learning_rate": 5e-5, "batch_size": 16}
    ]

    best_bert_f1 = 0
    best_bert_model = None
    tuning_log = []

    for i, config in enumerate(bert_configs):
        config_str = f"lr={config['learning_rate']}, batch_size={config['batch_size']}"
        print(f"\n=======================================================")
        print(f" [Modal GPU] Training Configuration {i+1}/3: {config_str}")
        print(f"=======================================================")

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

        training_args = TrainingArguments(
            output_dir=f"./results_bert_{i}",
            num_train_epochs=2,
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        eval_results = trainer.evaluate()
        val_macro_f1 = eval_results["eval_macro_f1"]
        print(f"--> Config {i+1} Validation Macro F1: {val_macro_f1:.4f}")

        tuning_log.append(["BERT Base", config_str, val_macro_f1])

        if val_macro_f1 > best_bert_f1:
            best_bert_f1 = val_macro_f1
            best_bert_model = model

    # Persist best model to Volume so it survives container teardown
    best_bert_model.save_pretrained("/vol/bert_best_model")
    bert_tokenizer.save_pretrained("/vol/bert_best_model")
    volume.commit()
    print(f"💾 Best model saved to Volume (Val Macro F1: {best_bert_f1:.4f})")

    # Test Set Prediction & Probability Generation
    print("\n=======================================================")
    print(" [Modal GPU] Generating Final Test Predictions & Probabilities")
    print("=======================================================")
    test_trainer = Trainer(model=best_bert_model)
    predictions = test_trainer.predict(test_dataset)
    logits = predictions.predictions
    probs = softmax(logits, axis=1)
    test_preds = np.argmax(logits, axis=1)

    acc = accuracy_score(y_test, test_preds)
    macro_f1 = f1_score(y_test, test_preds, average="macro")
    print(f"Final Test Accuracy: {acc * 100:.2f}% | Test Macro F1: {macro_f1:.4f}")

    class_names = ["Benign_Vanilla", "Benign_Adversarial", "Harmful_Vanilla", "Harmful_Adversarial"]
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=class_names))

    tuning_df = pd.DataFrame(tuning_log, columns=["Model", "Config", "Val F1"])
    bert_predictions_df = pd.DataFrame({
        "bert_preds": test_preds,
        "prob_0": probs[:, 0],
        "prob_1": probs[:, 1],
        "prob_2": probs[:, 2],
        "prob_3": probs[:, 3]
    })

    return tuning_df.to_csv(index=False), bert_predictions_df.to_csv(index=False)


@app.local_entrypoint()
def main():
    train_file = "data/train_clean.csv" if os.path.exists("data/train_clean.csv") else "train_clean.csv"
    val_file = "data/val_clean.csv" if os.path.exists("data/val_clean.csv") else "val_clean.csv"
    test_file = "data/test_clean.csv" if os.path.exists("data/test_clean.csv") else "test_clean.csv"

    if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
        raise FileNotFoundError(
            "Cleaned datasets not found. Please ensure train_clean.csv, val_clean.csv, "
            "and test_clean.csv exist in the 'data/' directory or project root."
        )

    print(f"Reading dataset files from '{train_file}', '{val_file}', '{test_file}'...")
    with open(train_file, "r") as f:
        train_csv = f.read()
    with open(val_file, "r") as f:
        val_csv = f.read()
    with open(test_file, "r") as f:
        test_csv = f.read()

    print("🚀 Launching remote training on Modal A10G Cloud GPU...")
    tuning_csv_str, preds_csv_str = train_bert_remote.remote(train_csv, val_csv, test_csv)

    os.makedirs("BERT Data", exist_ok=True)

    with open("BERT Data/bert_tuning_log.csv", "w") as f:
        f.write(tuning_csv_str)
    with open("BERT Data/bert_predictions.csv", "w") as f:
        f.write(preds_csv_str)

    # Also write to root for convenience
    with open("bert_tuning_log.csv", "w") as f:
        f.write(tuning_csv_str)
    with open("bert_predictions.csv", "w") as f:
        f.write(preds_csv_str)

    print("\n✅ Training complete! Output files saved to 'BERT Data/' and project root:")
    print("   - BERT Data/bert_tuning_log.csv")
    print("   - BERT Data/bert_predictions.csv")
