# DSA4212: Compute-Efficient Training of Small Transformer Language Models

**Author:** SONG MINGJUN (A034593Y)  
**Repository:** https://github.com/JS-2028/dsa4212-small-llm

## Project summary
This project trains decoder-only Transformer language models from random initialisation on TinyStories using PyTorch and a Google Colab NVIDIA Tesla T4 GPU. It investigates training duration (E1), training-data quantity (E2), and model capacity under a fixed wall-clock budget (E3).

## Main findings
- E1 (exploratory): validation loss was 3.9758 after 500 updates and 2.5115 after 5,000 updates. The schedules and validation sampling differed, so this is not a fully controlled causal comparison.
- E2 (three seeds per condition): mean validation losses were 2.5555 (10k stories), 2.5160 (50k), and 2.5150 (100k). Gains beyond 50k were marginal under 5,000 updates.
- E3 (one seed per size, ~150 s): validation losses were 2.4671 (5.27M parameters), 2.6092 (13.82M), and 2.8401 (29.45M). Larger models completed fewer updates in the same approximate time.

See `report/report.pdf` for the full discussion and limitations.

## Environment
- Python 3.13.18
- PyTorch 2.14.1+cu130
- NVIDIA Tesla T4, 15.64 GB reported VRAM
- Hugging Face `datasets`, `tokenizers`, NumPy, pandas, matplotlib, tqdm

## Reproducibility
The experiments were conducted in Google Colab. The training pipeline used:
- TinyStories; a byte-level BPE tokenizer with vocabulary size 8,000 trained on a 200,000-story training-only sample.
- EOS appended to each story.
- Context length 256, batch size 32.
- AdamW, FP16 autocast and gradient scaling, gradient clipping.
- Baseline architecture: 4 Transformer blocks, embedding dimension 256, 4 heads, tied token embedding/output weights.

The report documents the specific configurations and caveats for E1–E3. The original Colab notebook should be exported and added to `notebooks/` if available; this repository snapshot includes the report, summary data, and figures.

## Repository contents
- `authors.md`: author and student ID.
- `report/`: PDF report, LaTeX source, bibliography, and figures.
- `results/`: experiment summary CSV files.
- `configs/`: experiment configuration summaries.
- `code/`: model implementation and training code.

## Important limitations
The held-out test split was reserved but not evaluated before submission. E2 used three seeds per condition and fixed validation batches; E3 used one seed per model size. See the report for details.

## Execution notebook
The complete executed Google Colab notebook should be exported as `code/experiments.ipynb`. The standalone model architecture is in `code/model.py`.

## Results files
- `results/e1_training_duration.csv`: E1 exploratory duration comparison.
- `results/e2_all_runs.csv`: individual E2 runs (original run timing/memory were not recorded and are blank).
- `results/e2_replication.csv`: E2 mean and standard deviation across three seeds.
- `results/e3_model_scaling.csv`: E3 wall-clock model-scaling results.

