from urllib.parse import urlparse, parse_qs
import re


def validate_urls(video_urls):
    """Validate that the user provides between 5 and 20 URLs."""

    if not 5 <= len(video_urls) <= 20:
        raise ValueError(
            f"Please provide between 5 and 20 YouTube URLs. "
            f"You provided {len(video_urls)}."
        )


def extract_video_id(url):
    """Extract the video ID from different types of YouTube URLs."""

    parsed_url = urlparse(url.strip())

    # Normal YouTube URL
    # https://www.youtube.com/watch?v=VIDEO_ID
    if parsed_url.hostname in [
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com"
    ]:

        if parsed_url.path == "/watch":
            return parse_qs(parsed_url.query).get("v", [None])[0]

        # YouTube Shorts
        # https://youtube.com/shorts/VIDEO_ID
        if parsed_url.path.startswith("/shorts/"):
            return parsed_url.path.split("/shorts/")[1].split("/")[0]

        # Embedded YouTube URL
        # https://youtube.com/embed/VIDEO_ID
        if parsed_url.path.startswith("/embed/"):
            return parsed_url.path.split("/embed/")[1].split("/")[0]

    # Shortened YouTube URL
    # https://youtu.be/VIDEO_ID
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.strip("/").split("/")[0]

    return None


def get_video_ids(video_urls):
    """Validate URLs and return the extracted video IDs."""

    validate_urls(video_urls)

    video_ids = []

    for url in video_urls:
        video_id = extract_video_id(url)

        if video_id is None:
            raise ValueError(f"Invalid YouTube URL: {url}")

        video_ids.append(video_id)

    return video_ids
import requests
import pandas as pd
import os

from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("YOUTUBE_API_KEY")


def extract_video_details(video_ids):

    video_api_url = "https://www.googleapis.com/youtube/v3/videos"

    params = {
        "part": "snippet,statistics,contentDetails",
        "id": ",".join(video_ids),
        "key": API_KEY
    }

    response = requests.get(video_api_url, params=params)

    if response.status_code != 200:
        raise Exception(
            f"YouTube API Error {response.status_code}: {response.text}"
        )

    data = response.json()

    video_data = []

    for video in data.get("items", []):

        snippet = video["snippet"]
        statistics = video["statistics"]
        content_details = video["contentDetails"]

        video_data.append({
            "video_id": video["id"],
            "title": snippet.get("title"),
            "upload_date": snippet.get("publishedAt"),
            "channel_title": snippet.get("channelTitle"),
            "category_id": snippet.get("categoryId"),
            "duration": content_details.get("duration"),
            "views": statistics.get("viewCount", 0),
            "likes": statistics.get("likeCount", 0),
            "comment_count": statistics.get("commentCount", 0)
        })

    video_df = pd.DataFrame(video_data)

    # Convert published timestamp to date
    video_df["upload_date"] = pd.to_datetime(
        video_df["upload_date"]
    ).dt.date

    # Convert duration from PT4M20S to 4:20
    video_df["duration"] = video_df["duration"].apply(format_duration)

    # Convert numeric columns
    video_df["views"] = pd.to_numeric(video_df["views"])
    video_df["likes"] = pd.to_numeric(video_df["likes"])
    video_df["comment_count"] = pd.to_numeric(video_df["comment_count"])

    # Get actual YouTube category names
    category_api_url = (
        "https://www.googleapis.com/youtube/v3/videoCategories"
    )

    category_params = {
        "part": "snippet",
        "id": ",".join(video_df["category_id"].astype(str)),
        "key": API_KEY
    }

    category_response = requests.get(
        category_api_url,
        params=category_params
    )

    if category_response.status_code != 200:
        raise Exception(
            f"Category API Error "
            f"{category_response.status_code}: "
            f"{category_response.text}"
        )

    category_data = category_response.json()

    category_mapping = {}

    for item in category_data.get("items", []):
        category_mapping[item["id"]] = item["snippet"]["title"]

    video_df["category"] = (
        video_df["category_id"]
        .astype(str)
        .map(category_mapping)
    )

    # Remove category_id because we only need the category name
    video_df.drop(columns=["category_id"], inplace=True)

    return video_df

def format_duration(duration):
    hours = re.search(r"(\d+)H", duration)
    minutes = re.search(r"(\d+)M", duration)
    seconds = re.search(r"(\d+)S", duration)

    h = int(hours.group(1)) if hours else 0
    m = int(minutes.group(1)) if minutes else 0
    s = int(seconds.group(1)) if seconds else 0

    return f"{h * 60 + m}:{s:02d}"

def extract_comments(video_ids, max_comments_per_video=100):

    comments_data = []

    comments_url = "https://www.googleapis.com/youtube/v3/commentThreads"

    for video_id in video_ids:

        next_page_token = None
        extracted_for_video = 0

        while extracted_for_video < max_comments_per_video:

            params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(100, max_comments_per_video - extracted_for_video),
                "textFormat": "plainText",
                "key": API_KEY
            }

            if next_page_token:
                params["pageToken"] = next_page_token

            response = requests.get(
                comments_url,
                params=params
            )

            # Some videos may have comments disabled
            if response.status_code != 200:
                print(
                    f"Could not extract comments for video: {video_id}"
                )
                break

            data = response.json()

            for item in data.get("items", []):

                comment = (
                    item["snippet"]
                    ["topLevelComment"]
                    ["snippet"]
                )

                comments_data.append({
                    "video_id": video_id,
                    "comment_id": (
                        item["snippet"]
                        ["topLevelComment"]
                        ["id"]
                    ),
                    "comment": comment.get("textDisplay")
                })
                extracted_for_video += 1

                if extracted_for_video >= max_comments_per_video:
                    break

            next_page_token = data.get("nextPageToken")

            if not next_page_token or extracted_for_video >= max_comments_per_video:
                break

    comments_df = pd.DataFrame(comments_data)

    return comments_df

def clean_comments(comments_df):

    # Remove rows where comment is missing
    comments_df = comments_df.dropna(subset=["comment"])

    # Remove empty or whitespace-only comments
    comments_df = comments_df[
        comments_df["comment"].str.strip() != ""
    ]

    # Remove duplicate comments using comment_id
    comments_df = comments_df.drop_duplicates(
        subset=["comment_id"]
    )

    # Reset index
    comments_df = comments_df.reset_index(drop=True)

    return comments_df
