# Transformer Fine-Tuning and LLM Preference Alignment

PyTorch and Hugging Face examples covering **DistilBERT customer-support classification** and **LLM supervised fine-tuning (SFT) followed by Direct Preference Optimization (DPO)**. An Odds Ratio Preference Optimization (ORPO) configuration is also shown for comparison.

> **Project status:** The DistilBERT Python script passes syntax validation but has not been end-to-end trained as part of this repository preparation. The supplied `DPO_PPO.py` is **not yet runnable**: it contains syntax/indentation errors and uses trainer arguments that need to be migrated to current TRL/Transformers APIs. Despite its filename, **PPO is not implemented**. See [Known issues](#known-issues-and-limitations).

## Project files

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── DistilBERT_Colab_Fixed_2026.py
├── DistilBERT_Colab_Fixed_2026.ipynb
└── DPO_SFT.py
```

- **`DistilBERT_Colab_Fixed_2026.py`** — Full fine-tuning, evaluation, inference, and a frozen-backbone comparison for a four-class customer-support classifier.
- **`DistilBERT_Colab_Fixed_2026.ipynb`** — A proper Jupyter Notebook generated from the Python script, organized into eight code sections.
- **`DPO_SFT.py`** — An *unmodified, work-in-progress* SFT/DPO example based on Microsoft Phi-2 with LoRA and 4-bit quantization. Contains a textual ORPO training pattern; no executable PPO algorithm.
- **`requirements.txt`** — Unified installation dependencies for both examples. Installing them alone does not fix errors in `DPO_PPO.py`.

## What the code demonstrates

### 1. DistilBERT customer-support classifier

**Model:** [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)  
**Frameworks:** PyTorch, Transformers `Trainer`, Hugging Face Datasets, scikit-learn.

The code uses 30 embedded, manually labeled support requests across four classes:

| Label | Typical request |
| --- | --- |
| `billing` | Duplicate charges, invoices, payment problems |
| `technical` | API errors, bugs, app crashes |
| `account` | User management and account settings |
| `feature_request` | New integrations or requested functionality |

Workflow:

1. Split examples into training (24) and validation (6) sets with stratification.
2. Tokenize with `AutoTokenizer` and dynamic padding.
3. Fine-tune all model weights with `Trainer` and a classification head.
4. Report validation accuracy, weighted F1, classification report, and confusion matrix.
5. Predict categories for example support requests.
6. Train a second model with the DistilBERT backbone frozen, then compare validation results.

**Important:** With just 30 samples and only six validation cases, the evaluation scores are educational demonstrations, not reliable estimates of real-world performance. The script prints metrics at runtime; no fixed accuracy claim is made here.

### 2. LLM SFT → DPO experiment (work in progress)

**Base model:** [`microsoft/phi-2`](https://huggingface.co/microsoft/phi-2) (approximately 2.7 billion parameters).  
**Libraries:** Transformers, PEFT/LoRA, TRL, Hugging Face Datasets, bitsandbytes, pandas.

Intended workflow in `DPO_PPO.py`:

1. Build six chat-style **SFT** instruction/response examples.
2. Load Phi-2 with 4-bit NF4 quantization and prepare LoRA adapters.
3. Attempt instruction tuning with `SFTTrainer`.
4. Create five **DPO preference pairs** (`prompt`, `chosen`, `rejected`).
5. Attempt preference optimization using `DPOTrainer`.
6. Compare qualitative generated responses and analyze response lengths.
7. Print an **ORPO** code example for single-stage alignment.

The source does not define or train a PPO reward model or PPO policy. ORPO training is not executed either.

## Installation

### Python environment (Windows PowerShell)

Python **3.12** is recommended to reduce package compatibility issues.

```powershell
cd path\to\your\repository
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If your machine has a CUDA-enabled NVIDIA GPU, install a **CUDA-enabled PyTorch build** that matches your platform following the [official PyTorch selector](https://pytorch.org/get-started/locally/) **before** installing the remaining dependencies. Check GPU availability:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A working GPU driver does not guarantee a CUDA-enabled PyTorch build. CPU execution works for the DistilBERT script but can be much slower. The Phi-2 4-bit experiment requires an appropriately configured accelerator and more memory than is comfortably available on many 4 GB consumer GPUs.

### Google Colab / Jupyter

Open the `DistilBERT_Colab_Fixed_2026.ipynb` notebook. In a notebook cell, install the dependencies using:

```python
%pip install -r requirements.txt
```

Upload `requirements.txt` into Colab alongside the notebook, or install the dependencies directly. Select a GPU runtime if available. Restart the kernel after installation if imports fail.

For the CLI script, use a terminal—not direct bare Python statements typed into PowerShell.

## Running the DistilBERT project

Run the Python script:

```powershell
python DistilBERT_Colab_Fixed_2026.py
```

Or launch JupyterLab and open the notebook:

```powershell
jupyter lab
```

Run cells from top to bottom. The first run downloads `distilbert-base-uncased` model files from Hugging Face unless cached.

The script writes training artifacts into:

```text
support_classifier/
└── best_model/
support_classifier_frozen/
```

The `best_model` folder contains the saved fine-tuned model and tokenizer. Training checkpoints and generated model weights are ignored by the included `.gitignore` so GitHub does not receive large binaries by accident.

## Running the SFT/DPO project

**Do not run `DPO_SFT.py` unchanged and expect training to start.** First fix the syntax errors and migrate the trainer calls. A syntax check exposes the first issue:

```powershell
python -m py_compile DPO_PPO.py
```

After correction, a typical command would be:

```powershell
python DPO_SFT.py
```

This command is documentation for a **future corrected version**, not a verified working run of the current file.

## Known issues and limitations

- **`DPO_SFT.py` syntax:** The uploaded file has an unterminated multiline string at approximately line 44, widespread missing indentation, and other issues. Python cannot execute it as uploaded.
- **TRL API compatibility:** The SFT and DPO calls use older patterns such as `tokenizer=`, `max_seq_length=`, and a `TrainingArguments` configuration with `warmup_ratio`. Current TRL commonly uses `processing_class` and `SFTConfig`/`DPOConfig` with updated fields. Update the source based on the exact installed versions rather than blindly upgrading packages.
- **Prompt formatting:** The manual `<|system|>`, `<|user|>`, `<|assistant|>` delimiters are not a verified Phi-2 chat template; correct prompt and tokenizer formatting are needed.
- **`bitsandbytes` / CUDA:** 4-bit NF4 training depends on supported hardware, CUDA-enabled PyTorch and compatible native binaries. A 4 GB GPU can still run out of memory with Phi-2.
- **Datasets:** Both projects use tiny, embedded educational examples. Their outputs should not be presented as production-quality model results.
- **Notebook correction:** An uploaded file named `DistilBERT_Colab_Fixed_2026(1).ipynb` contained **Python source instead of notebook JSON**. The valid `.ipynb` in this repository is generated from the matching `.py` file.
- **PPO:** Not implemented. ORPO is a demonstration string, not a completed training stage.

## Recommended next improvements

- Fix and modernize `DPO_SFT.py` against a documented set of compatible `transformers`, `trl`, `peft` and `bitsandbytes` versions.
- Replace placeholder chat delimiters with a tokenizer-aware prompt template and add held-out evaluation.
- Expand the DistilBERT dataset and use multiple random seeds or cross-validation.
- Add automated tests and reproducible version locks after the pipelines pass full integration tests.
- Avoid committing API keys, local caches, trained model weights or sensitive datasets.

## References

- [PyTorch](https://pytorch.org/docs/stable/index.html)
- [Hugging Face Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Hugging Face TRL DPOTrainer](https://huggingface.co/docs/trl/dpo_trainer)
- [PEFT / LoRA](https://huggingface.co/docs/peft/index)
- [bitsandbytes installation](https://huggingface.co/docs/bitsandbytes/installation)

## License

No license is supplied with the original project. Add an appropriate `LICENSE` file if you want to grant others permission to reuse or modify the code. Third-party model and dataset licenses remain applicable.
