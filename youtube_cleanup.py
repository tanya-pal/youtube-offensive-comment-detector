import os
import csv
import json
import time
import joblib
from collections import defaultdict
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Load your model
binary_model = joblib.load("binary_offensive_model.pkl")

# Scopes for YouTube API
SCOPES = ['https://www.googleapis.com/auth/youtube.force-ssl']

# Authenticate and get API client
def get_youtube_service():
    flow = InstalledAppFlow.from_client_secrets_file(r"D:\project\youtube offensive comments\ML Model\client_secret.json", SCOPES)
    creds = flow.run_local_server(port=0)
    youtube = build('youtube', 'v3', credentials=creds)
    return youtube

# Get all video IDs
def get_video_ids(youtube, channel_id):
    video_ids = []
    next_page_token = None

    while True:
        res = youtube.search().list(
            part="id",
            channelId=channel_id,
            maxResults=50,
            pageToken=next_page_token,
            type="video"
        ).execute()

        for item in res['items']:
            video_ids.append(item['id']['videoId'])

        next_page_token = res.get("nextPageToken")
        if not next_page_token:
            break

    return video_ids

# Get comments for a video
def get_comments(youtube, video_id):
    comments = []
    next_page_token = None

    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            pageToken=next_page_token
        ).execute()

        for item in response["items"]:
            top_comment = item["snippet"]["topLevelComment"]
            comment_text = top_comment["snippet"]["textDisplay"]
            comment_id = top_comment["id"]
            author = top_comment["snippet"].get("authorDisplayName", "Unknown")
            author_channel_id = top_comment["snippet"]["authorChannelId"]["value"]

            comments.append({
                "comment": comment_text,
                "comment_id": comment_id,
                "author": author,
                "author_link": f"https://www.youtube.com/channel/{author_channel_id}",
                "video_id": video_id
            })


        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments

# Delete a comment
def delete_comment(youtube, comment_id):
    youtube.comments().setModerationStatus(
        id=comment_id,
        moderationStatus="rejected"
    ).execute()

# Save seen comment IDs
def save_seen_ids(seen_ids, path="seen_comments.json"):
    with open(path, "w") as f:
        json.dump(list(seen_ids), f)

# Load seen comment IDs
def load_seen_ids(path="seen_comments.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return set(json.load(f))
    return set()

# Main loop
def continuous_moderation():
    youtube = get_youtube_service()
    channel_id = input("Enter your YouTube Channel ID: ")
    seen_ids = load_seen_ids()
    POLL_INTERVAL = 300  # 5 minutes

    while True:
        print("🔁 Checking for new comments...")
        video_ids = get_video_ids(youtube, channel_id)
        deleted_comments = []
        offense_counter = defaultdict(int)

        author_links = {}

        for video_id in video_ids:
            comments = get_comments(youtube, video_id)

            for comment in comments:
                cid = comment["comment_id"]
                if cid in seen_ids:
                    continue

                seen_ids.add(cid)
                pred = binary_model.predict([comment["comment"]])[0]

                if pred:
                    delete_comment(youtube, cid)
                    print(f"❌ Deleted: {comment['comment']} by {comment['author']}")
                    comment["video_link"] = f"https://www.youtube.com/watch?v={video_id}"
                    deleted_comments.append(comment)
                    offense_counter[comment["author"]] += 1
                    author_links[comment["author"]] = comment["author_link"]

        # Save deleted comments
        if deleted_comments:
            with open("deleted_comments.csv", "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                if os.stat("deleted_comments.csv").st_size == 0:
                    writer.writerow(["username", "comment", "video_link"])

                for c in deleted_comments:
                    writer.writerow([c["author"], c["comment"], c["video_link"]])

        # Save offense counts with author links
        if offense_counter:
            with open("offensive_user_counts.csv", "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["username", "offense_count", "author_link"])
                for user, count in offense_counter.items():
                    writer.writerow([user, count, author_links.get(user, "")])

        save_seen_ids(seen_ids)
        print(f"✅ Cycle complete. Sleeping for {POLL_INTERVAL // 60} minutes...\n")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    continuous_moderation()

#pip install google-auth-oauthlib google-api-python-client joblib
# pip install google-auth-oauthlib google-api-python-client joblib pandas scikit-learn