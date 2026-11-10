"""Four-class customer support classifier using DistilBERT.

Works in a Python script, Jupyter, or Google Colab with recent Hugging Face
Transformers. Also handles the older Trainer argument names used by 4.40.x.

Install (PowerShell/terminal):
    python -m pip install -U torch transformers datasets accelerate scikit-learn pandas

Install (Google Colab *notebook cell*):
    %pip install -U transformers datasets accelerate scikit-learn pandas

The 30-row dataset is educational, not sufficient for reliable production metrics.
"""

# %% Imports and environment
import inspect

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
else:
    print("No CUDA GPU found. Training will run on CPU (slower).")
    print("In Google Colab, select Runtime > Change runtime type > GPU to use one.")

use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

# %% Prepare the four-label customer support dataset
RAW_DATA = [
    ("My payment failed and I was charged twice", "billing"),
    ("I cannot log into my account after the password reset", "technical"),
    ("How do I update my credit card on file?", "billing"),
    ("The API keeps returning 500 errors on every request", "technical"),
    ("Can you add dark mode to the dashboard?", "feature_request"),
    ("I need to change the email address on my account", "account"),
    ("Why was I billed for a Pro plan when I am on Starter?", "billing"),
    ("The export feature is not working, it just spins forever", "technical"),
    ("I want to add two more users to my team", "account"),
    ("It would be great if you could support CSV imports", "feature_request"),
    ("My invoice for last month shows the wrong amount", "billing"),
    ("The dashboard is very slow to load on Firefox", "technical"),
    ("How do I delete my account permanently?", "account"),
    ("Please add a keyboard shortcut for the search function", "feature_request"),
    ("I upgraded to Pro but my limits are still showing Starter", "billing"),
    ("The webhook is not firing even though I have it configured correctly", "technical"),
    ("Can I have two separate workspaces under one account?", "account"),
    ("Would it be possible to integrate with Zapier?", "feature_request"),
    ("I was charged after I cancelled my subscription", "billing"),
    ("Authentication tokens keep expiring after 5 minutes", "technical"),
    ("I need to transfer account ownership to a colleague", "account"),
    ("A bulk edit feature for records would save us hours each week", "feature_request"),
    ("The discount code I received is not being accepted at checkout", "billing"),
    ("Error message ERR-4821 appears whenever I try to upload a file", "technical"),
    ("How do I revoke access for a former team member?", "account"),
    ("Support for custom branding on exported reports would be valuable", "feature_request"),
    ("I received a refund confirmation but the money has not appeared", "billing"),
    ("The mobile app crashes immediately on iOS 17", "technical"),
    ("I want to merge two accounts that belong to the same organization", "account"),
    ("An offline mode for the mobile app would help our field team", "feature_request"),
]

LABEL_NAMES = ["billing", "technical", "account", "feature_request"]
LABEL2ID = {label: index for index, label in enumerate(LABEL_NAMES)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}

texts = [text for text, _ in RAW_DATA]
labels = [LABEL2ID[label] for _, label in RAW_DATA]

# Stratified splitting preserves representation of the four labels.
train_texts, eval_texts, train_labels, eval_labels = train_test_split(
    texts,
    labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels,
)

dataset = DatasetDict(
    {
        "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
        "validation": Dataset.from_dict({"text": eval_texts, "label": eval_labels}),
    }
)
print(f"Dataset prepared: {len(dataset['train'])} training, {len(dataset['validation'])} validation examples.")
print("Training labels:", pd.Series(train_labels).map(ID2LABEL).value_counts().to_dict())
print("Validation labels:", pd.Series(eval_labels).map(ID2LABEL).value_counts().to_dict())

# %% Tokenization
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True, max_length=128)


tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format(type="torch")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print("Tokenization complete.")
print("Train features:", tokenized_dataset["train"].features)
print("First input length:", len(tokenized_dataset["train"][0]["input_ids"]))

# %% Shared helpers and evaluation metrics

def compute_metrics(eval_pred):
    """Compute classification metrics from Hugging Face EvalPrediction."""
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    references = eval_pred.label_ids
    return {
        "accuracy": float(accuracy_score(references, predictions)),
        "f1_weighted": float(
            f1_score(
                references,
                predictions,
                labels=list(range(len(LABEL_NAMES))),
                average="weighted",
                zero_division=0,
            )
        ),
    }


def create_training_arguments(**kwargs):
    """Support Transformers 4.40 and 5.x without deprecated warmup arguments.

    warmup_fraction=0.1 means 10% of optimizer updates are warmup.
    Transformers 4.x uses warmup_ratio; Transformers >=5.2 uses a fractional
    warmup_steps value for the same behavior.
    """
    accepted = inspect.signature(TrainingArguments).parameters

    if "warmup_fraction" in kwargs:
        fraction = kwargs.pop("warmup_fraction")
        if not 0 <= fraction < 1:
            raise ValueError("warmup_fraction must be between 0 and 1 (exclusive)")
        if "warmup_ratio" in accepted:
            kwargs["warmup_ratio"] = fraction
        elif "warmup_steps" in accepted:
            kwargs["warmup_steps"] = fraction
        else:
            raise RuntimeError("Installed Transformers has no warmup setting")

    if "eval_strategy" in kwargs and "eval_strategy" not in accepted:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def create_trainer(**kwargs):
    """Handle the modern 'processing_class' vs old 'tokenizer' argument."""
    arg_name = (
        "processing_class"
        if "processing_class" in inspect.signature(Trainer.__init__).parameters
        else "tokenizer"
    )
    return Trainer(**kwargs, **{arg_name: tokenizer})


print("Evaluation metrics defined: accuracy and weighted F1.")

# %% Full fine-tuning
set_seed(SEED)  # Same randomly initialized head for a fairer comparison.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_NAMES),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
).to(device)

print(f"Model: {MODEL_NAME}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

training_args = create_training_arguments(
    output_dir="./support_classifier",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    warmup_fraction=0.1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    logging_strategy="epoch",
    report_to="none",
    bf16=use_bf16,
    fp16=use_fp16,
    dataloader_pin_memory=torch.cuda.is_available(),
    seed=SEED,
)

trainer = create_trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting full fine-tuning...")
train_result = trainer.train()
print("Full fine-tuning complete.")
print(f"Training loss: {train_result.training_loss:.4f}")
print(f"Runtime (s): {train_result.metrics.get('train_runtime', float('nan')):.1f}")
print(f"Samples/second: {train_result.metrics.get('train_samples_per_second', float('nan')):.2f}")

# Save trained model and tokenizer after selecting the best checkpoint.
trainer.save_model("./support_classifier/best_model")
tokenizer.save_pretrained("./support_classifier/best_model")

# %% Evaluate and create classification report
full_results = trainer.evaluate()
print("\nFull fine-tuning evaluation:")
for metric in ("eval_accuracy", "eval_f1_weighted", "eval_loss"):
    print(f"  {metric}: {full_results[metric]:.4f}")

prediction_output = trainer.predict(tokenized_dataset["validation"])
logits = prediction_output.predictions
if isinstance(logits, tuple):
    logits = logits[0]
predicted_ids = np.argmax(logits, axis=-1)
true_ids = np.asarray(prediction_output.label_ids)

print("\nClassification report:")
print(
    classification_report(
        true_ids,
        predicted_ids,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
)
print("Confusion matrix (rows=true, columns=predicted):")
conf_matrix = confusion_matrix(
    true_ids, predicted_ids, labels=list(range(len(LABEL_NAMES)))
)
print(pd.DataFrame(conf_matrix, index=LABEL_NAMES, columns=LABEL_NAMES))

# %% Inference on new queries

def classify_query(query: str) -> dict:
    """Predict a category using the trained full fine-tuning model."""
    trained_model = trainer.model
    trained_model.eval()
    model_device = next(trained_model.parameters()).device
    encoded = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
    encoded = {key: value.to(model_device) for key, value in encoded.items()}

    with torch.inference_mode():
        logits = trained_model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(probabilities.argmax().item())
    return {
        "query": query,
        "predicted_label": ID2LABEL[predicted_id],
        "confidence": round(float(probabilities[predicted_id].item()), 4),
        "all_scores": {
            ID2LABEL[i]: round(float(prob.item()), 4)
            for i, prob in enumerate(probabilities)
        },
    }


test_queries = [
    "I was charged twice for the same month",
    "The search bar is broken and returns no results",
    "How can I change my username?",
    "It would be amazing to have a Slack notification integration",
    "My team cannot see the shared workspace I created",
]
print("\nInference on new queries:")
for query in test_queries:
    result = classify_query(query)
    print(f"Query: {result['query']}")
    print(f"Prediction: {result['predicted_label']} (confidence: {result['confidence']:.4f})")
    print(f"All scores: {result['all_scores']}\n")

# %% Frozen-backbone comparison
# DistilBERT has a pre_classifier AND a classifier. Train both head layers;
# otherwise the pre_classifier may remain randomly initialized.
set_seed(SEED)
model_frozen = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_NAMES),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
).to(device)

for parameter in model_frozen.distilbert.parameters():
    parameter.requires_grad = False

frozen_trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
frozen_total = sum(p.numel() for p in model_frozen.parameters())
print("\nFrozen-backbone model:")
print(f"  Total parameters: {frozen_total:,}")
print(f"  Trainable (classification head): {frozen_trainable:,}")
print(f"  Frozen: {frozen_total - frozen_trainable:,}")
print(f"  Trainable fraction: {frozen_trainable / frozen_total:.4%}")

frozen_args = create_training_arguments(
    output_dir="./support_classifier_frozen",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-3,
    warmup_fraction=0.1,
    eval_strategy="epoch",
    save_strategy="no",
    logging_strategy="epoch",
    report_to="none",
    bf16=use_bf16,
    fp16=use_fp16,
    dataloader_pin_memory=torch.cuda.is_available(),
    seed=SEED,
)

frozen_trainer = create_trainer(
    model=model_frozen,
    args=frozen_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Training frozen-backbone classifier...")
frozen_trainer.train()
frozen_results = frozen_trainer.evaluate()

print("\nComparison:")
print(
    "Full fine-tuning: "
    f"Accuracy={full_results['eval_accuracy']:.4f} "
    f"F1={full_results['eval_f1_weighted']:.4f}"
)
print(
    "Frozen backbone: "
    f"Accuracy={frozen_results['eval_accuracy']:.4f} "
    f"F1={frozen_results['eval_f1_weighted']:.4f}"
)
print("Note: Only 6 validation examples. Scores are not statistically reliable.")
