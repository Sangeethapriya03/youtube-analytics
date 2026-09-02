def add_engagement_rate(video_df):

    video_df = video_df.copy()

    video_df["engagement_rate"] = (
        (video_df["likes"] + video_df["comment_count"])
        / video_df["views"]
    ) * 100

    video_df["engagement_rate"] = video_df["engagement_rate"].round(4)

    return video_df

def calculate_sentiment_percentages(comments_df):
    """
    Calculate Positive, Neutral and Negative percentages per video.
    """

    sentiment_counts = (
        comments_df
        .groupby(["video_id", "sentiment"])
        .size()
        .unstack(fill_value=0)
    )

    required_sentiments = [
        "Positive",
        "Neutral",
        "Negative"
    ]

    for sentiment in required_sentiments:
        if sentiment not in sentiment_counts.columns:
            sentiment_counts[sentiment] = 0

    sentiment_counts = sentiment_counts[required_sentiments]

    sentiment_percentages = sentiment_counts.div(
        sentiment_counts.sum(axis=1),
        axis=0
    ) * 100

    sentiment_percentages = (
        sentiment_percentages
        .round(2)
        .reset_index()
    )

    return sentiment_percentages 