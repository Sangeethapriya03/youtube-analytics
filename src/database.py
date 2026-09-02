import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus


load_dotenv()


MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")


def get_engine():

    password = quote_plus(MYSQL_PASSWORD)

    engine = create_engine(
        f"mysql+pymysql://{MYSQL_USER}:{password}"
        f"@{MYSQL_HOST}/{MYSQL_DATABASE}"
    )

    return engine


def test_connection():

    engine = get_engine()

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return True
def insert_videos(video_df):

    engine = get_engine()

    query = text("""
        INSERT INTO videos (
            video_id,
            title,
            upload_date,
            channel_title,
            category,
            duration,
            views,
            likes,
            comment_count
        )
        VALUES (
            :video_id,
            :title,
            :upload_date,
            :channel_title,
            :category,
            :duration,
            :views,
            :likes,
            :comment_count
        )
        ON DUPLICATE KEY UPDATE
            title = VALUES(title),
            upload_date = VALUES(upload_date),
            channel_title = VALUES(channel_title),
            category = VALUES(category),
            duration = VALUES(duration),
            views = VALUES(views),
            likes = VALUES(likes),
            comment_count = VALUES(comment_count)
    """)

    records = video_df.to_dict(orient="records")

    with engine.begin() as connection:
        connection.execute(query, records)


def insert_comments(comments_df):

    engine = get_engine()

    query = text("""
        INSERT IGNORE INTO comments (
            comment_id,
            video_id,
            comment
        )
        VALUES (
            :comment_id,
            :video_id,
            :comment
        )
    """)

    records = comments_df.to_dict(orient="records")

    with engine.begin() as connection:
        connection.execute(query, records)


def update_engagement_rate(video_df):

    engine = get_engine()

    query = text("""
        UPDATE videos
        SET engagement_rate = :engagement_rate
        WHERE video_id = :video_id
    """)

    records = video_df[
        ["video_id", "engagement_rate"]
    ].to_dict(orient="records")

    with engine.begin() as connection:
        connection.execute(query, records)

def update_performance_clusters(video_df):
    """
    Update cluster and performance group in the videos table.
    """

    engine = get_engine()

    query = text("""
        UPDATE videos
        SET
            cluster = :cluster,
            performance_group = :performance_group
        WHERE video_id = :video_id
    """)

    records = video_df[
        ["video_id", "cluster", "performance_group"]
    ].to_dict(orient="records")

    with engine.begin() as connection:
        connection.execute(query, records)

    print(
        f"Performance clusters updated for "
        f"{len(records)} videos!"
    )
def update_comment_sentiments(comments_df):
    """
    Update sentiment values in the comments table.
    """

    engine = get_engine()

    sentiment_data = comments_df[
        ["comment_id", "sentiment"]
    ].dropna().to_dict(orient="records")

    if not sentiment_data:
        print("No sentiment values available to update.")
        return

    query = text("""
        UPDATE comments
        SET sentiment = :sentiment
        WHERE comment_id = :comment_id
    """)

    with engine.begin() as connection:
        connection.execute(query, sentiment_data)

    print(
        f"Sentiment updated successfully for "
        f"{len(sentiment_data)} comments!"
    )