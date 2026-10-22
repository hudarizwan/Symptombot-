import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score
import joblib
import os

# === Config ===
SYMPTOM_CSV = "data/symptom_disease.csv"
PRECAUTION_CSV = "data/disease_precaution.csv"  # Fixed file name
MODEL_PATH = "disease_model.pkl"
ENCODER_PATH = "symptom_encoder.pkl"
PRECAUTION_DICT_PATH = "precautions_dict.pkl"

# === Load Datasets ===
print("🔄 Loading datasets...")
symptoms_df = pd.read_csv(SYMPTOM_CSV)
precautions_df = pd.read_csv(PRECAUTION_CSV)

# === Preprocess Symptoms ===
print("🧹 Preprocessing symptoms...")
symptoms_df['Symptom_List'] = symptoms_df.iloc[:, 1:].values.tolist()
symptoms_df['Symptom_List'] = symptoms_df['Symptom_List'].apply(
    lambda x: [symptom.strip() for symptom in x if isinstance(symptom, str) and symptom.strip()]
)
symptoms_df = symptoms_df[symptoms_df['Symptom_List'].map(len) > 0]

# === Encode Symptoms ===
print("🔢 Encoding symptoms...")
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(symptoms_df['Symptom_List'])
y = symptoms_df['Disease']

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Model ===
print("🧠 Training RandomForest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# === Save Artifacts ===
print("💾 Saving model and encoder...")
joblib.dump(clf, MODEL_PATH)
joblib.dump(mlb, ENCODER_PATH)

# === Precaution Mapping ===
print("🩺 Creating precaution dictionary...")
precautions_df.columns = [col.strip() for col in precautions_df.columns]  # Trim headers
precautions_dict = precautions_df.set_index("Disease").T.to_dict()
joblib.dump(precautions_dict, PRECAUTION_DICT_PATH)

print("🎉 Training complete! Files saved:")
print(f" - Model: {MODEL_PATH}")
print(f" - Encoder: {ENCODER_PATH}")
print(f" - Precautions Dict: {PRECAUTION_DICT_PATH}")
