import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Step 1: Load Dataset
df = pd.read_csv(r"ML Model/youtube_offensive_dataset.csv")

# Step 2: Preprocess
df["comment"] = df["comment"].astype(str)
df["offensive"] = df["offensive"].astype(bool)

# For multi-label classification, split 'type' column
df["type_list"] = df["type"].apply(lambda x: x.split(", ") if x != "non_offensive" else [])

# Step 3: Split into Train/Test (80/20)
X_train, X_test, y_train_bin, y_test_bin = train_test_split(
    df["comment"], df["offensive"], test_size=0.2, random_state=42
)

# For multi-label classification
mlb = MultiLabelBinarizer()
y_multi = mlb.fit_transform(df["type_list"])
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    df["comment"], y_multi, test_size=0.2, random_state=42
)

# Step 4: Pipeline for binary classifier
binary_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

binary_clf.fit(X_train, y_train_bin)
y_pred_bin = binary_clf.predict(X_test)

print("=== Binary Classification Report ===")
print(classification_report(y_test_bin, y_pred_bin))

# Step 5: Pipeline for multi-label classifier
multi_label_clf = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000)),
    ('clf', OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
])

multi_label_clf.fit(X_train_multi, y_train_multi)
y_pred_multi = multi_label_clf.predict(X_test_multi)

print("=== Multi-Label Classification Report ===")
print(classification_report(y_test_multi, y_pred_multi, target_names=mlb.classes_))

# Step 6: Save models
joblib.dump(binary_clf, "binary_offensive_model.pkl")
joblib.dump(multi_label_clf, "type_classification_model.pkl")
joblib.dump(mlb, "label_binarizer.pkl")

print("✅ Models saved successfully!")

#terminal : pip install pandas scikit-learn joblib
           #pip install pandas scikit-learn joblib numpy



# Load the saved binary model once
binary_model = joblib.load("binary_offensive_model.pkl")

def is_offensive(comment_text):
    """
    Predict if a comment is offensive using the trained binary classifier.
    Returns True if offensive, else False.
    """
    return bool(binary_model.predict([comment_text])[0])
