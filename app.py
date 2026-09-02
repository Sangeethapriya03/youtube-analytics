import pandas as pd
import plotly.express as px
import streamlit as st

from src.youtube_extractor import get_video_ids, extract_video_details, extract_comments, clean_comments
from src.sentiment import load_sentiment_model, add_sentiment
from src.analytics import add_engagement_rate, calculate_sentiment_percentages
from src.clustering import add_performance_clusters

MAX_COMMENTS_PER_VIDEO = 100

st.set_page_config(page_title="YouTube Analytics", layout="wide")

@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    return load_sentiment_model()

def require_data():
    if "video_df" not in st.session_state:
        st.info("Paste 5 to 20 URLs in Overview and click Analyze Videos.")
        return False
    return True

def short_titles(frame):
    result = frame.copy()
    result["short_title"] = result["title"].str.slice(0, 45)
    return result

st.title("YouTube Analytics")
st.caption("Video performance, category trends, clustering and comment sentiment")

overview_tab, performance_tab, category_tab, clustering_tab, sentiment_tab = st.tabs([
    "Overview", "Video Performance", "Category & Time", "Video Clustering", "Comment Sentiment"
])

with overview_tab:
    st.subheader("YouTube Analytics Overview")
    urls_text = st.text_area("Paste 5 to 20 YouTube video URLs (one URL per line)", height=180)

    if st.button("Analyze Videos", type="primary"):
        video_urls = [url.strip() for url in urls_text.splitlines() if url.strip()]
        try:
            video_ids = get_video_ids(video_urls)
            if len(video_ids) != len(set(video_ids)):
                raise ValueError("Duplicate videos detected. Use every video only once.")

            with st.status("Starting YouTube analysis...", expanded=True) as status:
                st.write("URLs validated successfully.")
                st.write("Extracting video details...")
                video_df = extract_video_details(video_ids)
                returned_ids = set(video_df["video_id"])
                unavailable = [url for url, video_id in zip(video_urls, video_ids) if video_id not in returned_ids]
                if unavailable:
                    raise ValueError("These videos are private, deleted or unavailable:\n" + "\n".join(unavailable))

                st.write("Extracting a maximum of 100 comments per video...")
                comments_df = clean_comments(extract_comments(video_ids, MAX_COMMENTS_PER_VIDEO))
                if comments_df.empty:
                    raise ValueError("No public comments were found for the selected videos.")

                st.write("Loading the RoBERTa sentiment model...")
                model = get_sentiment_model()
                st.write(f"Analysing sentiment for {len(comments_df):,} comments...")
                comments_df = add_sentiment(comments_df, model)
                video_df = add_performance_clusters(add_engagement_rate(video_df))

                st.session_state.video_df = video_df
                st.session_state.comments_df = comments_df
                st.session_state.sentiment_df = calculate_sentiment_percentages(comments_df)
                status.update(label="Analysis completed!", state="complete", expanded=False)

            st.success(f"Analysed {len(video_df)} videos and {len(comments_df):,} comments successfully!")
            st.rerun()
        except Exception as error:
            st.error(str(error))

    if require_data():
        videos = st.session_state.video_df
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Videos", f"{len(videos):,}")
        c2.metric("Total Views", f"{videos['views'].sum():,}")
        c3.metric("Total Likes", f"{videos['likes'].sum():,}")
        c4.metric("Total Comments", f"{videos['comment_count'].sum():,}")
        c5.metric("Avg Engagement", f"{videos['engagement_rate'].mean():.2f}%")
        best = videos.loc[videos["engagement_rate"].idxmax()]
        st.success(f"Top-performing video: {best['title']} — {best['engagement_rate']:.2f}% engagement")
        st.dataframe(videos[["title", "channel_title", "category", "views", "likes", "comment_count", "engagement_rate"]], use_container_width=True, hide_index=True)

with performance_tab:
    st.subheader("Video Performance")
    if require_data():
        videos = short_titles(st.session_state.video_df)
        top10 = videos.nlargest(10, "views").sort_values("views")
        st.plotly_chart(px.bar(top10, x="views", y="short_title", orientation="h", color="views", title="Top 10 Videos by Views", labels={"short_title": "Video"}), use_container_width=True)
        left, right = st.columns(2)
        left.plotly_chart(px.bar(videos, x="short_title", y="likes", color="likes", title="Likes by Video", labels={"short_title": "Video"}), use_container_width=True)
        right.plotly_chart(px.bar(videos, x="short_title", y="comment_count", color="comment_count", title="Comments by Video", labels={"short_title": "Video", "comment_count": "Comments"}), use_container_width=True)
        st.plotly_chart(px.bar(videos, x="short_title", y="engagement_rate", color="engagement_rate", title="Engagement Rate by Video", labels={"short_title": "Video", "engagement_rate": "Engagement Rate (%)"}), use_container_width=True)

with category_tab:
    st.subheader("Category and Time Analysis")
    if require_data():
        videos = st.session_state.video_df.copy()
        category = videos.groupby("category", as_index=False).agg(views=("views", "sum"), likes=("likes", "sum"), comments=("comment_count", "sum"), engagement_rate=("engagement_rate", "mean"))
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.bar(category, x="category", y="views", color="views", title="Views by Category"), use_container_width=True)
        c2.plotly_chart(px.bar(category, x="category", y="likes", color="likes", title="Likes by Category"), use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(px.bar(category, x="category", y="comments", color="comments", title="Comments by Category"), use_container_width=True)
        c4.plotly_chart(px.bar(category, x="category", y="engagement_rate", color="engagement_rate", title="Category vs Engagement Rate"), use_container_width=True)

        duration_parts = videos["duration"].str.split(":", expand=True).astype(int)
        videos["duration_minutes"] = duration_parts[0] + duration_parts[1] / 60
        videos["duration_group"] = pd.cut(videos["duration_minutes"], bins=[-1, 5, 15, 30, float("inf")], labels=["Short (≤5 min)", "Medium (6–15 min)", "Long (16–30 min)", "Very Long (>30 min)"])
        duration = videos.groupby("duration_group", observed=True, as_index=False)["views"].sum()
        left, right = st.columns(2)
        left.plotly_chart(px.bar(duration, x="duration_group", y="views", color="views", title="Views by Duration"), use_container_width=True)
        right.plotly_chart(px.line(videos.sort_values("upload_date"), x="upload_date", y="views", markers=True, hover_name="title", title="Video Uploads Over Time"), use_container_width=True)

with clustering_tab:
    st.subheader("Video Clustering")
    if require_data():
        videos = st.session_state.video_df
        colors = {"Best Performing": "#00CC96", "Popular but Less Interactive": "#636EFA", "Promising": "#FECB52", "Needs Improvement": "#EF553B"}
        figure = px.scatter(videos, x="views", y="engagement_rate", color="performance_group", size="likes", hover_name="title", log_x=True, color_discrete_map=colors, title="Video Performance Clusters", labels={"views": "Views (log scale)", "engagement_rate": "Engagement Rate (%)", "performance_group": "Performance Group"})
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(videos[["title", "views", "engagement_rate", "performance_group"]], use_container_width=True, hide_index=True)

with sentiment_tab:
    st.subheader("Comment Sentiment Analysis")
    if require_data():
        videos = st.session_state.video_df
        comments = st.session_state.comments_df
        title = st.selectbox("Select a video", videos["title"].tolist())
        video_id = videos.loc[videos["title"] == title, "video_id"].iloc[0]
        selected = comments[comments["video_id"] == video_id]
        counts = selected["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0).reset_index()
        counts.columns = ["Sentiment", "Comments"]
        counts["Percentage"] = (counts["Comments"] / counts["Comments"].sum() * 100).round(2)
        c1, c2, c3 = st.columns(3)
        for column, name in zip([c1, c2, c3], ["Positive", "Neutral", "Negative"]):
            column.metric(name, f"{counts.loc[counts['Sentiment'] == name, 'Percentage'].iloc[0]:.2f}%")
        left, right = st.columns(2)
        colors = {"Positive": "#00CC96", "Neutral": "#FECB52", "Negative": "#EF553B"}
        left.plotly_chart(px.pie(counts, names="Sentiment", values="Comments", hole=0.45, title="Sentiment Distribution", color="Sentiment", color_discrete_map=colors), use_container_width=True)
        right.plotly_chart(px.bar(counts, x="Sentiment", y="Percentage", color="Sentiment", text="Percentage", title="Sentiment Percentage", color_discrete_map=colors), use_container_width=True)
        with st.expander("View analysed comments"):
            st.dataframe(selected[["comment", "sentiment"]], use_container_width=True, hide_index=True)
