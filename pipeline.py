from src.clustering import add_performance_clusters
from src.database import (
    test_connection,
    insert_videos,
    insert_comments,
    update_engagement_rate,
    update_comment_sentiments,
    update_performance_clusters
)
from src.youtube_extractor import (
    get_video_ids,
    extract_video_details,
    extract_comments,
    clean_comments
)

from src.analytics import (
    add_engagement_rate,
    calculate_sentiment_percentages
)
from src.sentiment import load_sentiment_model, add_sentiment

video_urls = [
    "https://youtu.be/1RqoUViryxs?si=i2ZUS3SGo2aM7cf6",
    "https://youtu.be/4h5FgMOmyng?si=RXOW5ju1qzh3fBDG",
    "https://youtu.be/ysV4--OdBes?si=IDB0ZdtE1m0cf-Xi",
    "https://youtu.be/AAE9CHg4weY?si=gOR1l55uVI2lk3Cp",
    "https://youtu.be/BgQGaI5-0jA?si=Ufjhf44YekfdWt1-"
]


video_ids = get_video_ids(video_urls)

print("Number of videos:", len(video_ids))
print("Video IDs:", video_ids)

video_df = extract_video_details(video_ids)



print("\nVideo Details:")
print(video_df)
print("\nColumns:")
print(video_df.columns.tolist())


comments_df = extract_comments(video_ids)
comments_df = clean_comments(comments_df)

print("\nTotal comments after cleaning:", len(comments_df))

print("\nComments:")
print(comments_df.head())

print("\nComments per video:")
print(comments_df["video_id"].value_counts())

print("\nLoading RoBERTa sentiment model...")
sentiment_model = load_sentiment_model()

print("Analyzing comment sentiments...")
comments_df = add_sentiment(comments_df, sentiment_model)

print("\nSentiment analysis completed!")
print(comments_df[[
    "comment_id",
    "video_id",
    "comment",
    "sentiment"
]].head())

print("\nOverall sentiment counts:")
print(comments_df["sentiment"].value_counts())

sentiment_percentages_df = calculate_sentiment_percentages(
    comments_df
)

print("\nSentiment percentages per video:")
print(sentiment_percentages_df)

if test_connection():
    print("\nMySQL connection successful!")


insert_videos(video_df)
print("\nVideos inserted successfully!")

insert_comments(comments_df)
print("Comments inserted successfully!")

update_comment_sentiments(comments_df)

video_df = add_engagement_rate(video_df)

print("\nVideo data with engagement rate:")
print(video_df[[
    "video_id",
    "views",
    "likes",
    "comment_count",
    "engagement_rate"
]])

update_engagement_rate(video_df)

print("\nEngagement rate updated in MySQL!")
video_df = add_performance_clusters(video_df)

print("\nVideo performance clustering completed!")

print(video_df[[
    "video_id",
    "title",
    "views",
    "engagement_rate",
    "cluster",
    "performance_group"
]])

update_performance_clusters(video_df)

print("\nClustering results updated in MySQL!")