YouTube Offensive Comment Moderation System

🔹 Input Dataset:
•	I used a shortened version of Kaggle's Jigsaw Toxic Comment Classification Challenge, trimmed to 500 rows and saved as youtube_offensive_dataset.csv.
🔹 Models Built:
•	Binary Classifier (offensive column): Detects whether a comment is offensive or not.
•	Multi-label Classifier (type column): Predicts offensive categories (if any).
•	Both models are trained with TfidfVectorizer + RandomForestClassifier.
•	Saved to:
	binary_offensive_model.pkl
	type_classification_model.pkl
	label_binarizer.pkl
🔹 YouTube API Integration:
•	Credentials stored in client_secret.json.
•	Uses OAuth 2.0 authentication.
•	Scans your channel’s videos, fetches comments, checks for offensiveness, and automatically deletes the offensive ones.
🔹 Moderation Logic (youtube_cleanup.py):
•	Gets videos from a channel using channel_id .
•	Fetches all comments.
•	If a comment is offensive:
o	Deletes it.
o	Saves it to deleted_comments.csv
o	Tracks offense counts per user into offensive_user_counts.csv
•	Polls every 5 minutes.

