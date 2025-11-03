# Context-Aware Next-Word Prediction (WikiText-2)

This folder contains the materials I will submit for the CST435 final project. The deliverable trains a recurrent neural network on WikiText-2, persists the resulting tokenizer/model artifacts, and exposes an interactive Streamlit demo for next-word suggestions.

## Repository Layout
- `main_final.ipynb` – end-to-end notebook: data prep, training, evaluation, and packaging.
- `requirements.txt` – Python dependencies used across the notebook, scripts, and Streamlit app.
- `tokenizer_utils.py` – shared preprocessing, tokenizer persistence, and sequence helpers.
- `prepare_tokenizer.py` – CLI utility to build the tokenizer artifacts without training the full model.
- `train_model.py` – script version of the notebook training loop; exports the best weights and training history.
- `streamlit_app.py` – interactive next-word demo powered by the saved model/tokenizer.
- `artifacts/tokenizer.json` – serialized tokenizer produced during training.
- `best_model.keras` – saved Keras model used by the Streamlit app (same weights as `best_wikitext2_lstm.keras` referenced in the notebook).

## 1. Environment Setup
1. Use Python 3.10 or 3.11 (tested with 3.11).
2. Create and activate a virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the project dependencies.
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 2. Download Required Data (GloVe)
The model initialises its embedding layer with the 100-dimensional GloVe vectors.

Two options:

- **Include the file before zipping**: place `glove.6B.100d.txt` under `glove_6B/` (preferred for a self-contained submission).
- **Or regenerate it locally**: run the snippet below from the project root.
  ```bash
  mkdir -p glove_6B
  curl -L -o glove.6B.zip https://nlp.stanford.edu/data/glove.6B.zip
  unzip -j glove.6B.zip glove.6B.100d.txt -d glove_6B
  ```

Both `train_model.py` and `streamlit_app.py` will raise a `FileNotFoundError` if the vector file is missing.

## 3. Optional: Prepare Tokenizer Only
If you want to (re)generate the tokenizer without training the RNN:
```bash
python prepare_tokenizer.py --vocab-size 20000 --output artifacts/tokenizer.json
```
This script downloads WikiText-2 via the Hugging Face `datasets` library, cleans the text, and saves the tokenizer payload for reuse.

## 4. Full Model Training Script
Run the script to reproduce training outside the notebook:
```bash
python train_model.py \
  --seq-len 10 \
  --vocab-size 20000 \
  --glove-path glove_6B/glove.6B.100d.txt \
  --epochs 15 \
  --batch-size 256 \
  --model-output best_wikitext2_lstm.keras
```
Outputs:
- `best_wikitext2_lstm.keras` (best validation accuracy checkpoint; identical weights to `best_model.keras`).
- `best_wikitext2_lstm.history.json` (training metrics per epoch).
- `artifacts/tokenizer.json` (overwritten with the tokenizer fit during training).

## 5. Streamlit Demo
Launch the interactive UI once the model/tokenizer artifacts and GloVe vectors are ready:
```bash
streamlit run streamlit_app.py
```
Features:
- Enter a prompt and view top-k next-word suggestions.
- Adjust temperature, top-k, and continuation length.
- Inspect sampling traces and probability tables.

## 6. Notebook Usage
Open `main_final.ipynb` in Jupyter for a narrative walkthrough of the project. The notebook mirrors the script functionality and includes:
- Dataset exploration and preprocessing visuals.
- Model definition, training, and evaluation plots.
- Qualitative decoding examples and packaging steps.

## 7. Troubleshooting
- **Missing GloVe vectors**: ensure `glove_6B/glove.6B.100d.txt` exists; re-run the download command if necessary.
- **Datasets download blocked**: the Hugging Face loader caches under `artifacts/hf_cache`. If running offline, pre-populate that cache or toggle `HF_DATASETS_OFFLINE=1` once the data is cached.
- **TensorFlow GPU warnings**: the project trains on CPU if no GPU is available; warnings can be ignored.
