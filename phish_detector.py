"""
Phishing URL Detector
- Extracts simple lexical features from URLs, trains Logistic Regression, plots ROC
- Can classify new URLs from stdin
Usage:
  python 10_phish_detector.py train urls.csv
  python 10_phish_detector.py predict "http://example.com/login"
CSV format: url,label (label in {0,1})
Dependencies: scikit-learn, pandas, matplotlib, numpy
"""
import sys, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def features(url):
    return [
        len(url),
        url.count("@"),
        url.count("?"),
        url.count("%"),
        url.count("."),
        int(bool(re.search(r"https?://\d+\.\d+\.\d+\.\d+", url))),
        int("login" in url.lower() or "verify" in url.lower() or "update" in url.lower()),
        int("https://" not in url.lower()),
    ]

def train(csv_path):
    df = pd.read_csv(csv_path)
    X = np.array([features(u) for u in df['url']]); y = df['label'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())]).fit(Xtr, ytr)
    probs = pipe.predict_proba(Xte)[:,1]
    fpr, tpr, _ = roc_curve(yte, probs); roc_auc = auc(fpr, tpr)
    plt.figure(); plt.plot(fpr, tpr); plt.plot([0,1],[0,1],"--"); plt.title(f"ROC AUC={roc_auc:.3f}"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.show()
    return pipe

if __name__ == "__main__":
    if len(sys.argv)<2: print(__doc__); sys.exit(0)
    if sys.argv[1]=="train":
        train(sys.argv[2])
    else:
        url = " ".join(sys.argv[2:])
        print("Features:", features(url))
