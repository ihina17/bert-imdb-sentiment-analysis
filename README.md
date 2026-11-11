# IMDb Sentiment Analysis with Fine-Tuned BERT

EDS 6352 — Natural Language Processing, Assignment 4 (Fall 2026).

This project evaluates four fine-tuned `bert-base-uncased` sentiment classifiers on IMDb movie reviews, then compares them with the hand-implemented Naïve Bayes baseline from Assignment 1.

## Models

1. **Model 1 — Linear:** standard BERT sequence classifier with a linear classification head.
2. **Model 2 — ReLU:** linear (768 → 256), ReLU, linear (256 → 2).
3. **Model 3 — GELU:** linear (768 → 256), GELU, linear (256 → 2).
4. **Model 4 — Last-four-layer [CLS] average:** averages [CLS] outputs from BERT transformer blocks 9–12 and applies a linear head.

## Dataset and training

- 40,000 initial IMDb reviews; 277 duplicate review texts removed.
- 800 training, 200 validation, 400 held-out test reviews (200 per test class).
- Sequence length 384, batch size 8, learning rate 2e-5, 3 epochs, AdamW.
- BERT fine-tuned on CPU; checkpoints selected by validation accuracy.

## Results

| Model | Best validation accuracy | Test accuracy | Test macro F1 |
| --- | ---: | ---: | ---: |
| Model 1 — Linear | 90.00% | 87.25% | 0.8725 |
| Model 2 — ReLU | 89.00% | **88.75%** | **0.8875** |
| Model 3 — GELU | 89.50% | 86.25% | 0.8625 |
| Model 4 — Last-four-layer average | **91.50%** | 86.00% | 0.8598 |
| Assignment 1 Naïve Bayes (best scenario) | — | 77.25% | 0.7725 |

Model 4 was chosen by the highest validation accuracy, while Model 2 achieved the highest observed test accuracy. The distinction matters: test performance was not used to select the final model.

## Files

- `Mahajan_HW4.ipynb`: full experiment, Markdown explanations, training code, recorded outputs, plots, and classification reports.
- `requirements.txt`: primary Python dependencies.
- `.gitignore`: keeps large checkpoints and local dataset files out of Git.

## Running the notebook

1. Obtain the IMDb dataset used in the class assignment and place it beside the notebook as `movie.csv`.
2. Install dependencies from `requirements.txt`.
3. Open `Mahajan_HW4.ipynb` in Jupyter and run cells in order.

The notebook includes recorded results from completed runs. Re-running the training sections requires downloading `bert-base-uncased` and fine-tuning all four models, which was computationally expensive on CPU. Saved model checkpoints are **not** included in the repository.

The dataset's exact sampling/splitting procedure and model-specific evaluation are documented in the notebook.

## Note

This is a course assignment repository. Results are from one experiment with small validation/test sets and should not be treated as general benchmark claims.
