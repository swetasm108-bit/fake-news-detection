# Fake News Detection

Binary text classifier (real vs fake news) using TF-IDF + LinearSVC.

## Setup
1. `pip install -r requirements.txt`
2. Download "Fake and Real News Dataset" from Kaggle
   (https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
   and put `Fake.csv` and `True.csv` inside `data/`.
3. `python train.py`
4. `python predict.py "some news text"`

## Method
- Title + text combined, lowercased, URLs and symbols removed
- Removed the "(Reuters) -" source tag to avoid label leakage
- Duplicates dropped, stratified 80/20 split, TF-IDF fit on train only
- Unigrams + bigrams, LinearSVC

## Results
Run `train.py` and paste your own numbers here (accuracy, precision, recall, F1).
