"""Usage: python predict.py "paste the news text here" """
import sys

import joblib

from train import MODEL_PATH, clean_text

if len(sys.argv) < 2:
    sys.exit('Usage: python predict.py "news text"')

model = joblib.load(MODEL_PATH)
text = clean_text(sys.argv[1])
label = model.predict([text])[0]
score = model.decision_function([text])[0]   # distance from the decision boundary

print("Prediction:", "REAL" if label == 1 else "FAKE")
print(f"Confidence score: {abs(score):.2f} (higher = more confident)")
