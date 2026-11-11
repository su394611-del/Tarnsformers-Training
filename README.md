# Transformers Training — DistilBERT Classifier and Phi-2 Alignment Draft

 PyTorch/Hugging Face  **four-class customer-support text classification** and a separate **unfinished Phi-2 SFT/DPO design**.

> **Implementation status:** `DistilBERT_Colab_Fixed_2026.py` and its Jupyter notebook parse successfully. The classifier has not been run end-to-end as part of this packaging task. `DPO_SFT.py` is preserved **unchanged** but has Python syntax/indentation errors and cannot run. ORPO is only an example string that is not executed.

## Documentation

- [Wiki Home](wiki/Home.md) — full documentation set, ready for publication to GitHub Wiki
- [High-Level Design](wiki/System-Architecture.md) — implemented vs planned components
- [Low-Level Design](wiki/Low-Level-Design.md) — functions and library dependencies
- [Mermaid diagram sources](diagrams/) — editable `.mmd` files
- [Validation report](docs/VALIDATION.md) — what was and was not tested
- [Original supplied README](docs/ORIGINAL_README.md) — preserved for provenance

## Main source files

| File | Purpose |
|---|---|
| `DistilBERT_Colab_Fixed_2026.py` | Pretrained DistilBERT, 30 labeled examples, full fine-tuning, frozen-backbone comparison, evaluation and inference |
| `DistilBERT_Colab_Fixed_2026.ipynb` | Jupyter notebook companion |
| `DPO_SFT.py` | **Incomplete** Phi-2 LoRA/SFT/DPO draft; do not execute as-is |
| `requirements.txt` | Original unified requirements |
| `requirements_distilbert.txt` | Smaller classifier-only install |

## Run the classifier (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements_distilbert.txt
python DistilBERT_Colab_Fixed_2026.py
```

The first execution downloads the pretrained `distilbert-base-uncased` model unless cached. The 30-row dataset is embedded in source and split 24/6 for training/validation. Saved full-model output is placed under `support_classifier/best_model/`. No fixed accuracy figures are claimed from this tiny dataset.

## Publish the Wiki

1. Push the **contents of this extracted folder** (not the ZIP archive as a single code file) to your normal GitHub repository.
2. Enable **Wikis** in GitHub **Settings → General → Features** and save an initial Wiki `Home` page.
3. From this extracted folder's root, run:

```powershell
.\scripts\publish-wiki.cmd -Owner "YOUR_GITHUB_USERNAME" -Repository "YOUR_REPOSITORY_NAME" -DryRun
.\scripts\publish-wiki.cmd -Owner "YOUR_GITHUB_USERNAME" -Repository "YOUR_REPOSITORY_NAME"
```

The publisher uses a fresh temporary Git checkout, previews changes, and asks you to type `PUBLISH` before pushing. It publishes Markdown files from `wiki/` into the **separate** `.wiki.git` repository. Inspect `scripts/publish-wiki.ps1` before using it, particularly if a script's provenance is uncertain.

## Validation

Run `python scripts/check-package.py`. It checks source presence/AST syntax, notebook JSON and code-cell syntax, Mermaid blocks, cross-page links, and Markdown pages. It intentionally reports the existing `DPO_SFT.py` parse failure as an **expected source limitation**.

## Privacy and license

Review before publishing; do not commit credentials, confidential examples, cached model weights, `.venv`, or large checkpoints. **No LICENSE was supplied**, so no license has been assigned by this packaging task. Verify any third-party pretrained model license separately.
