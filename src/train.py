"""Kalp hastalığı risk tahmini — model eğitimi ve değerlendirme.

Kullanım:  python src/train.py --data data/heart_2020_cleaned.csv

Doğru değerlendirme protokolü:
  1. Önce katmanlı (stratified) eğitim/test ayrımı yapılır.
  2. Ön işleme ve rastgele çoğaltma (oversampling) YALNIZCA eğitim kümesinde yapılır.
  3. Modeller, gerçek sınıf dağılımını koruyan dokunulmamış test kümesinde ölçülür.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from preprocess import AGE_CATEGORY, GEN_HEALTH, TARGET, Preprocessor, encode_static

SEED = 42
ROOT = Path(__file__).resolve().parents[1]


def random_oversample(X: pd.DataFrame, y: pd.Series, seed: int = SEED):
    """Azınlık sınıfını rastgele tekrar seçerek sınıfları eşitler (imblearn RandomOverSampler eşdeğeri)."""
    rng = np.random.default_rng(seed)
    counts = y.value_counts()
    minority = counts.idxmin()
    idx_min = np.flatnonzero(y.to_numpy() == minority)
    extra = rng.choice(idx_min, size=counts.max() - counts.min(), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def make_models():
    return {
        "Lojistik Regresyon": LogisticRegression(max_iter=2000),
        "Naive Bayes": GaussianNB(),
        "Karar Ağacı": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=SEED),
    }


def scores(y_true, y_pred, y_prob) -> dict:
    # Not: sklearn metrikleri (y_true, y_pred) sırasıyla çağrılır.
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred),
        "f1": metrics.f1_score(y_true, y_pred),
        "roc_auc": metrics.roc_auc_score(y_true, y_prob),
        "pr_auc": metrics.average_precision_score(y_true, y_prob),
    }


def evaluate(models, X_tr, y_tr, X_te, y_te):
    rows, probs = {}, {}
    for name, model in models.items():
        t = time.time()
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        rows[name] = scores(y_te, (p >= 0.5).astype(int), p)
        probs[name] = p
        print(f"  {name:<20} ROC-AUC={rows[name]['roc_auc']:.3f}  recall={rows[name]['recall']:.3f}  "
              f"precision={rows[name]['precision']:.3f}  ({time.time()-t:.0f} sn)")
    return pd.DataFrame(rows).T, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "heart_2020_cleaned.csv"))
    ap.add_argument("--skip-leaky", action="store_true", help="Hatalı protokol karşılaştırmasını atla")
    args = ap.parse_args()

    out = ROOT / "results"; out.mkdir(exist_ok=True)
    df = encode_static(pd.read_csv(args.data))
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    print(f"Veri: {len(df):,} satır, pozitif oran = {y.mean():.3%}")

    # ---------------- Doğru protokol ----------------
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    pre = Preprocessor().fit(X_tr_raw)
    X_tr, X_te = pre.transform(X_tr_raw), pre.transform(X_te_raw)
    X_tr_bal, y_tr_bal = random_oversample(X_tr, y_tr)
    print(f"\nDoğru protokol (test: {len(y_te):,} gerçek hasta kaydı, dokunulmamış):")
    res, probs = evaluate(make_models(), X_tr_bal, y_tr_bal, X_te, y_te)
    res.round(4).to_csv(out / "model_results.csv", encoding="utf-8-sig")

    # ---------------- Karşılaştırma: orijinal (sızıntılı) protokol ----------------
    if not args.skip_leaky:
        Xp = pre.transform(X)  # basitlik için; sızıntının asıl kaynağı aşağıdaki sıralama
        X_ros, y_ros = random_oversample(Xp, y)           # önce çoğalt ...
        a, b, c, d = train_test_split(X_ros, y_ros, test_size=0.3, random_state=SEED)  # ... sonra ayır
        print("\nOrijinal protokol (önce oversampling, sonra ayırma) — yalnızca karşılaştırma için:")
        leaky, _ = evaluate({k: v for k, v in make_models().items() if k in ("Karar Ağacı", "Random Forest")}, a, c, b, d)
        leaky.round(4).to_csv(out / "leaky_protocol_comparison.csv", encoding="utf-8-sig")

    # ---------------- En iyi model: ROC-AUC'ye göre ----------------
    best = res["roc_auc"].idxmax()
    best_model = make_models()[best].fit(X_tr_bal, y_tr_bal)
    p = probs[best]
    # Klinik tarama senaryosu için: recall >= %80 olan en yüksek eşik
    prec, rec, thr = metrics.precision_recall_curve(y_te, p)
    ok = np.flatnonzero(rec[:-1] >= 0.80)
    screen_thr = float(thr[ok[-1]]) if len(ok) else 0.5
    screen = scores(y_te, (p >= screen_thr).astype(int), p)

    # Grafikler
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, pr in probs.items():
        fpr, tpr, _ = metrics.roc_curve(y_te, pr)
        ax[0].plot(fpr, tpr, lw=2 if name == best else 1, label=f"{name} (AUC={res.loc[name,'roc_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[0].set(xlabel="Yanlış pozitif oranı", ylabel="Doğru pozitif oranı (recall)", title="ROC eğrileri — test kümesi")
    ax[0].legend(loc="lower right", fontsize=8); ax[0].grid(alpha=.3)
    cm = metrics.confusion_matrix(y_te, (p >= 0.5).astype(int))
    ax[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[1].text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Sağlıklı", "Hasta"], yticklabels=["Sağlıklı", "Hasta"],
              xlabel="Tahmin", ylabel="Gerçek", title=f"Karışıklık matrisi — {best} (eşik 0.5)")
    plt.tight_layout(); plt.savefig(out / "roc_and_confusion.png", dpi=130); plt.close()

    # Arayüz (app.py) için kayıtlar
    utils = ROOT / "utils"; utils.mkdir(exist_ok=True)
    pickle.dump(best_model, open(utils / "best_model.pkl", "wb"))
    pickle.dump(pre, open(utils / "preprocessor.pkl", "wb"))
    pickle.dump({"gen_health": GEN_HEALTH, "age_category": AGE_CATEGORY}, open(utils / "dicts.pkl", "wb"))
    summary = {
        "best_model": best, "n_rows": int(len(df)), "positive_rate": float(y.mean()),
        "test_size": int(len(y_te)), "threshold_default": 0.5,
        "metrics_default": {k: round(float(v), 4) for k, v in res.loc[best].items()},
        "threshold_screening": round(screen_thr, 4),
        "metrics_screening": {k: round(float(v), 4) for k, v in screen.items()},
    }
    json.dump(summary, open(utils / "model_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nEn iyi model:", best); print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
