import joblib

# Load trained models
binary_model = joblib.load("binary_offensive_model.pkl")
multi_model = joblib.load("type_classification_model.pkl")
mlb = joblib.load("label_binarizer.pkl")

# ✍️ Manual input comment
comment = input("Enter a YouTube comment: ")
comment_list = [comment]

# Predict offensive or not
is_offensive = binary_model.predict(comment_list)[0]
print("\nPrediction:")
print("Offensive:", is_offensive)

# Predict type of offense if offensive
if is_offensive:
    pred_type = mlb.inverse_transform(multi_model.predict(comment_list))[0]
    print("Type:", ", ".join(pred_type) if pred_type else "Uncategorized")
else:
    print("Type: non_offensive")
