"""Kalp hastalığı veri seti için ön işleme.

Kodlamalar orijinal analizle aynıdır (Evet/Hayır → 1/0, yaş ve genel sağlık sıralı
kodlama, Race/Diabetic için LabelEncoder, sürekli değişkenlere MinMaxScaler).
Fark: dönüştürücüler YALNIZCA eğitim kümesine fit edilir, test kümesine sadece uygulanır.
"""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

TARGET = "HeartDisease"
BINARY_COLS = ["Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
               "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer"]
SCALED_COLS = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
GEN_HEALTH = {"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 4}
AGE_CATEGORY = {"18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3, "40-44": 4, "45-49": 5,
                "50-54": 6, "55-59": 7, "60-64": 8, "65-69": 9, "70-74": 10,
                "75-79": 11, "80 or older": 12}


def encode_static(df: pd.DataFrame) -> pd.DataFrame:
    """Veriden öğrenilmeyen, sabit eşlemelerle yapılan kodlamalar."""
    df = df.copy()
    for col in BINARY_COLS + ([TARGET] if TARGET in df else []):
        df[col] = df[col].map({"No": 0, "Yes": 1}).astype(int)
    df["Sex"] = df["Sex"].map({"Male": 0, "Female": 1}).astype(int)
    df["GenHealth"] = df["GenHealth"].map(GEN_HEALTH).astype(int)
    df["AgeCategory"] = df["AgeCategory"].map(AGE_CATEGORY).astype(int)
    return df


class Preprocessor:
    """Eğitim kümesine fit edilip hem eğitime hem teste uygulanan dönüştürücüler."""

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        self.le_race = LabelEncoder().fit(X["Race"])
        self.le_diabetic = LabelEncoder().fit(X["Diabetic"])
        self.scalers = {c: MinMaxScaler().fit(X[[c]]) for c in SCALED_COLS}
        self.feature_names = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["Race"] = self.le_race.transform(X["Race"])
        X["Diabetic"] = self.le_diabetic.transform(X["Diabetic"])
        for c, sc in self.scalers.items():
            X[c] = sc.transform(X[[c]]).ravel()
        return X[self.feature_names]
