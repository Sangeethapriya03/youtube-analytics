import html
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.youtube_extractor import get_video_ids, extract_video_details, extract_comments, clean_comments
from src.sentiment import load_sentiment_model, add_sentiment
from src.analytics import add_engagement_rate, calculate_sentiment_percentages
from src.clustering import add_performance_clusters

MAX_COMMENTS_PER_VIDEO = 100

# ---- Palette ----
RED, RED_DIM, BLUE, GREEN, AMBER, GRAY = "#FF0000", "#B00020", "#3EA6FF", "#2BA640", "#FF9D42", "#909090"
BG, PANEL, LINE, TEXT, MUTED, DIM = "#0F0F0F", "#181818", "#2D2D2D", "#F1F1F1", "#AAAAAA", "#717171"

st.set_page_config(page_title="YouTube Analytics", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap" rel="stylesheet">
<style>
:root{{--red:{RED};--blue:{BLUE};--green:{GREEN};--amber:{AMBER};--gray:{GRAY};--bg:{BG};--panel:{PANEL};--line:{LINE};--text:{TEXT};--muted:{MUTED};--dim:{DIM}}}
html,body,[class*="css"],.stApp{{font-family:"Roboto","Segoe UI",Arial,sans-serif!important;color:var(--text)!important}}
.stApp{{background:var(--bg)}}
.block-container{{max-width:1400px;padding:1.4rem 2.4rem 4rem}}
#MainMenu,footer,header{{visibility:hidden}}
/* ---- Top brand bar ---- */
.brandbar{{display:flex;align-items:center;gap:.7rem;padding:0 0 1.1rem;margin-bottom:1.3rem;border-bottom:1px solid var(--line)}}
.brandmark{{width:34px;height:34px;border-radius:8px;background:var(--red);display:flex;align-items:center;justify-content:center;flex-shrink:0}}
.brandmark svg{{width:15px;height:15px}}
.brandname{{font-size:1.18rem;font-weight:700;letter-spacing:-.01em;color:var(--text)}}
.brandsub{{font-size:.8rem;color:var(--dim);margin-left:.15rem}}
/* ---- Sidebar ---- */
section[data-testid="stSidebar"]{{background:var(--panel);border-right:1px solid var(--line)}}
section[data-testid="stSidebar"] .block-container{{padding:1.4rem 1rem}}
.sidebrand{{display:flex;align-items:center;gap:.55rem;padding:0 .4rem 1.1rem;margin-bottom:.9rem;border-bottom:1px solid var(--line)}}
.sidbrandmark{{width:28px;height:28px;border-radius:6px;background:var(--red);display:flex;align-items:center;justify-content:center;flex-shrink:0}}
.sidbrandmark svg{{width:12px;height:12px}}
.sidbrandtext{{font-size:1rem;font-weight:700;color:var(--text)}}
section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"]{{gap:.2rem!important}}
section[data-testid="stSidebar"] .stButton{{margin:0!important}}
section[data-testid="stSidebar"] .stButton>button{{background:transparent!important;color:var(--muted)!important;border:0!important;border-radius:8px!important;padding:.6rem .8rem!important;font-weight:500!important;font-size:.92rem!important;text-align:left!important;justify-content:flex-start!important;width:100%!important;box-shadow:none!important}}
section[data-testid="stSidebar"] .stButton>button:hover{{background:#272727!important;color:var(--text)!important}}
.navitem-active{{padding:.6rem .8rem;border-radius:8px;background:#272727;color:var(--text)!important;font-weight:700;font-size:.92rem;margin-bottom:.2rem}}
.sidefoot{{margin-top:1.4rem;padding:.9rem .8rem 0;border-top:1px solid var(--line);font-size:.76rem;color:var(--dim);line-height:1.5}}
/* ---- Headings ---- */
.pagetitle{{font-size:1.65rem;font-weight:700;letter-spacing:-.02em;color:var(--text);margin:0 0 .15rem}}
.pagesub{{color:var(--muted);font-size:.94rem;margin-bottom:1.35rem}}
/* ---- Search / URL input ---- */
div[data-testid="stTextArea"] label{{color:var(--text)!important;font-weight:500!important;font-size:.9rem!important}}
div[data-testid="stTextArea"] textarea{{border-radius:10px!important;border:1px solid var(--line)!important;background:var(--panel)!important;color:var(--text)!important;padding:.9rem 1rem!important;caret-color:var(--text)!important}}
div[data-testid="stTextArea"] textarea::placeholder{{color:var(--dim)!important}}
div[data-testid="stTextArea"] textarea:focus{{border-color:var(--red)!important;box-shadow:0 0 0 1px var(--red)!important}}
div[data-testid="stSelectbox"] label{{color:var(--text)!important;font-weight:500!important}}
div[data-testid="stSelectbox"] > div{{background:var(--panel)!important;border-color:var(--line)!important;border-radius:8px!important}}
.stButton>button{{border:0!important;border-radius:20px!important;padding:.7rem 1.6rem!important;font-weight:700!important;font-size:.92rem!important;color:#fff!important;background:var(--red)!important;box-shadow:none!important;transition:background .15s}}
.stButton>button:hover{{color:#fff!important;background:{RED_DIM}!important}}
/* ---- Charts / tables / expanders sit flush, no boxy cards ---- */
div[data-testid="stPlotlyChart"]{{background:transparent}}
div[data-testid="stDataFrame"]{{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
div[data-testid="stExpander"]{{background:var(--panel);border:1px solid var(--line);border-radius:10px}}
div[data-testid="stExpander"] summary{{color:var(--text)!important}}
div[data-testid="stStatusWidget"], div[data-testid="stStatus"]{{background:var(--panel)!important;border:1px solid var(--line)!important;border-radius:10px!important}}
/* ---- Metric strip: big number over label, hairline dividers ---- */
.metricstrip{{display:flex;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin-bottom:1.15rem}}
.metric{{flex:1;padding:1.05rem 1.3rem;border-right:1px solid var(--line)}}
.metric:last-child{{border-right:none}}
.metric-label{{font-size:.82rem;color:var(--muted);margin-bottom:.4rem}}
.metric-value{{font-size:1.65rem;font-weight:700;color:var(--text);letter-spacing:-.01em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
/* ---- Winner spotlight ---- */
.spotlight{{display:flex;align-items:center;gap:1rem;padding:1rem 1.3rem;margin:0 0 1.3rem;border:1px solid var(--line);border-left:3px solid var(--red);border-radius:10px;background:var(--panel)}}
.spotlight-eyebrow{{font-size:.78rem;color:var(--red);font-weight:700}}
.spotlight-title{{margin-top:.2rem;color:var(--text);font-weight:500}}
.spotlight-rate{{margin-left:auto;font-size:1.35rem;font-weight:700;color:var(--red);white-space:nowrap}}
/* ---- Cluster summary chips ---- */
.cluster-card{{padding:1rem 1.1rem;border:1px solid var(--line);border-left:3px solid var(--cluster-color);border-radius:10px;background:var(--panel);min-height:88px}}
.cluster-name{{font-size:.84rem;color:var(--muted)}}
.cluster-count{{margin-top:.4rem;font-size:1.5rem;font-weight:700;color:var(--text)}}
/* ---- Sentiment chips ---- */
.sentiment-card{{padding:1rem 1.1rem;border:1px solid var(--line);border-radius:10px;background:var(--panel)}}
.sentiment-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--sentiment-color);margin-right:.45rem}}
.sentiment-label{{font-size:.84rem;color:var(--muted)}}
.sentiment-value{{margin-top:.45rem;font-size:1.55rem;font-weight:700;color:var(--text)}}
.empty-note{{padding:2.6rem 1.5rem;text-align:center;border:1px dashed var(--line);border-radius:10px;background:var(--panel);color:var(--muted)}}
.empty-note b{{color:var(--text)}}
/* ---- About panel ---- */
.about{{padding:1.1rem 1.3rem;margin-bottom:1.3rem;border:1px solid var(--line);border-radius:10px;background:var(--panel)}}
.about p{{color:var(--muted);font-size:.92rem;line-height:1.6;margin:0 0 .6rem}}
.about p:last-child{{margin-bottom:0}}
.about strong{{color:var(--text)}}
@media(max-width:800px){{.block-container{{padding:1rem}}.metricstrip{{flex-wrap:wrap}}.metric{{flex:1 1 45%;border-bottom:1px solid var(--line)}}.spotlight-rate{{display:none}}}}
</style>""", unsafe_allow_html=True)

PLAY_SVG = '<svg viewBox="0 0 24 24" fill="white"><path d="M8 5v14l11-7z"/></svg>'


@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    return load_sentiment_model()


def page_header(title, copy):
    st.markdown(f'<div class="pagetitle">{title}</div><div class="pagesub">{copy}</div>', unsafe_allow_html=True)


def require_data():
    if "video_df" not in st.session_state:
        st.markdown(
            '<div class="empty-note"><b>No videos analysed yet.</b><br>'
            'Head to Overview, paste 5–20 links and select Analyze videos.</div>',
            unsafe_allow_html=True,
        )
        return False
    return True


def short_titles(frame):
    result = frame.copy()
    result["short_title"] = result["title"].apply(lambda x: x if len(x) <= 42 else x[:42] + "…")
    return result


def style_chart(fig, height=420):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Roboto, Segoe UI, Arial", color=TEXT, size=13),
        title=dict(font=dict(family="Roboto, Segoe UI, Arial", size=17, color=TEXT), x=0),
        margin=dict(l=34, r=24, t=64, b=45),
        hoverlabel=dict(bgcolor=PANEL, font_color=TEXT, bordercolor=LINE),
        legend=dict(orientation="h", y=1.09, x=0, xanchor="left", font=dict(color=MUTED, size=12)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LINE, tickfont_color=MUTED, title_font_color=MUTED)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, linecolor=LINE, tickfont_color=MUTED, title_font_color=MUTED)
    return fig


def red_bar(frame, x, y, title, orientation="v", labels=None, text_auto=".2s"):
    fig = px.bar(
        frame, x=x, y=y, orientation=orientation, title=title, labels=labels, text_auto=text_auto,
        color=y if orientation == "v" else x,
        color_continuous_scale=["#5c0000", RED, "#ff6a5e"],
    )
    fig.update_coloraxes(showscale=False)
    fig.update_traces(marker_cornerradius=4, marker_line_width=0, textposition="outside",
                       textfont_color=MUTED, cliponaxis=False)
    return style_chart(fig)


def duration_minutes(value):
    parts = [int(x) for x in str(value).split(":")]
    return parts[0] * 60 + parts[1] + parts[2] / 60 if len(parts) == 3 else \
        parts[0] + parts[1] / 60 if len(parts) == 2 else parts[0] / 60


def metric_strip(items):
    cells = "".join(
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" title="{value}">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="metricstrip">{cells}</div>', unsafe_allow_html=True)


PAGES = ["Overview", "Video performance", "Category & time", "Video clustering", "Comment sentiment"]
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(
        f'<div class="sidebrand"><div class="sidbrandmark">{PLAY_SVG}</div>'
        f'<div class="sidbrandtext">YouTube Analytics</div></div>',
        unsafe_allow_html=True,
    )
    for nav_page in PAGES:
        if nav_page == st.session_state.page:
            st.markdown(f'<div class="navitem-active">{nav_page}</div>', unsafe_allow_html=True)
        else:
            if st.button(nav_page, key=f"nav_{nav_page}", use_container_width=True):
                st.session_state.page = nav_page
                st.rerun()
    if "video_df" in st.session_state:
        st.markdown(
            f'<div class="sidefoot">{len(st.session_state.video_df)} videos analysed<br>'
            f'{len(st.session_state.comments_df):,} comments read</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="sidefoot">Waiting for your first batch of videos.</div>', unsafe_allow_html=True)

page = st.session_state.page

st.markdown(
    f'<div class="brandbar"><div class="brandmark">{PLAY_SVG}</div>'
    f'<div class="brandname">YouTube Analytics</div><div class="brandsub">Video performance & audience insight</div></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- overview
if page == "Overview":
    page_header("Overview", "Add 5–20 public YouTube links to extract, compare, cluster and analyse them.")
    st.markdown(
        '<div class="about">'
        '<p><strong>What this does:</strong> paste a batch of public YouTube video links below and this '
        'dashboard pulls each video\'s statistics and comments through the YouTube Data API, works out an '
        'engagement rate for every video, and groups the videos into four performance clusters with K-Means '
        'based on views and engagement.</p>'
        '<p>It also reads up to 100 comments per video through a pretrained RoBERTa model to score audience '
        'sentiment as positive, neutral or negative, then lays all of it out across the pages on the left: '
        'overview stats, video performance, category and upload-time patterns, clustering, and comment sentiment.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    urls_text = st.text_area("YouTube video URLs", placeholder="Paste one URL per line…", height=150,
                              label_visibility="collapsed", key="url_input")
    if st.button("Analyze videos", type="primary"):
        video_urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        try:
            video_ids = get_video_ids(video_urls)
            if len(video_ids) != len(set(video_ids)):
                raise ValueError("Duplicate videos detected. Use every video only once.")
            with st.status("Preparing your analytics dashboard…", expanded=True) as status:
                st.write("URLs validated")
                st.write("Fetching video performance details…")
                video_df = extract_video_details(video_ids)
                unavailable = [u for u, v in zip(video_urls, video_ids) if v not in set(video_df["video_id"])]
                if unavailable:
                    raise ValueError("These videos are private, deleted or unavailable:\n" + "\n".join(unavailable))
                st.write(f"Fetching up to {MAX_COMMENTS_PER_VIDEO} comments per video…")
                comments_df = clean_comments(extract_comments(video_ids, MAX_COMMENTS_PER_VIDEO))
                if comments_df.empty:
                    raise ValueError("No public comments were found for the selected videos.")
                st.write("Loading the RoBERTa language model…")
                model = get_sentiment_model()
                st.write(f"Reading audience sentiment across {len(comments_df):,} comments…")
                comments_df = add_sentiment(comments_df, model)
                video_df = add_performance_clusters(add_engagement_rate(video_df))
                st.session_state.video_df, st.session_state.comments_df = video_df, comments_df
                st.session_state.sentiment_df = calculate_sentiment_percentages(comments_df)
                st.session_state.submitted_urls = video_urls
                status.update(label="Dashboard ready", state="complete", expanded=False)
            st.success(f"Analysed {len(video_df)} videos and {len(comments_df):,} comments successfully.")
            st.rerun()
        except Exception as error:
            st.error(str(error))

    if require_data():
        videos = st.session_state.video_df
        metric_strip([
            ("Videos", f"{len(videos):,}"),
            ("Total views", f"{videos.views.sum():,}"),
            ("Total likes", f"{videos.likes.sum():,}"),
            ("Comments", f"{videos.comment_count.sum():,}"),
            ("Avg. engagement", f"{videos.engagement_rate.mean():.2f}%"),
        ])
        best = videos.loc[videos.engagement_rate.idxmax()]
        st.markdown(
            f'<div class="spotlight"><div><div class="spotlight-eyebrow">Top performer</div>'
            f'<div class="spotlight-title">{html.escape(str(best["title"]))}</div></div>'
            f'<div class="spotlight-rate">{best["engagement_rate"]:.2f}%</div></div>',
            unsafe_allow_html=True,
        )
        table = videos.copy()
        table["video_link"] = "https://www.youtube.com/watch?v=" + table["video_id"]
        st.dataframe(
            table[["title", "channel_title", "category", "views", "likes", "comment_count",
                   "engagement_rate", "video_link"]],
            use_container_width=True, hide_index=True,
            column_config={
                "engagement_rate": st.column_config.ProgressColumn(
                    "Engagement", format="%.2f%%", min_value=0, max_value=max(10., float(videos.engagement_rate.max()))),
                "video_link": st.column_config.LinkColumn("Video link", display_text="Open on YouTube"),
            },
        )

        if "submitted_urls" in st.session_state:
            with st.expander("URLs you submitted"):
                for submitted_url in st.session_state.submitted_urls:
                    st.markdown(f"- {submitted_url}")

# ---------------------------------------------------------------- performance
elif page == "Video performance":
    page_header("Video performance", "See which uploads earn attention and which ones turn viewers into an active audience.")
    if require_data():
        videos = short_titles(st.session_state.video_df)
        top10 = videos.nlargest(10, "views").sort_values("views")
        st.plotly_chart(red_bar(top10, "views", "short_title", "Top 10 videos by views", "h", {"short_title": "Video"}),
                         use_container_width=True)
        left, right = st.columns(2)
        left.plotly_chart(red_bar(videos, "short_title", "likes", "Likes by video"), use_container_width=True)
        right.plotly_chart(red_bar(videos, "short_title", "comment_count", "Comments by video"), use_container_width=True)
        st.plotly_chart(red_bar(videos, "short_title", "engagement_rate", "Engagement rate by video", text_auto=".2f"),
                         use_container_width=True)

# ---------------------------------------------------------------- category & time
elif page == "Category & time":
    page_header("Category & time", "Compare categories, video length and upload timing to find repeatable content opportunities.")
    if require_data():
        videos = st.session_state.video_df.copy()
        category = videos.groupby("category", as_index=False).agg(
            views=("views", "sum"), likes=("likes", "sum"),
            comments=("comment_count", "sum"), engagement_rate=("engagement_rate", "mean"),
        )
        a, b = st.columns(2)
        a.plotly_chart(red_bar(category, "category", "views", "Views by category"), use_container_width=True)
        b.plotly_chart(red_bar(category, "category", "likes", "Likes by category"), use_container_width=True)
        a, b = st.columns(2)
        a.plotly_chart(red_bar(category, "category", "comments", "Comments by category"), use_container_width=True)
        b.plotly_chart(red_bar(category, "category", "engagement_rate", "Engagement rate by category", text_auto=".2f"),
                        use_container_width=True)

        videos["duration_minutes"] = videos.duration.apply(duration_minutes)
        videos["duration_group"] = pd.cut(
            videos.duration_minutes, [-1, 5, 15, 30, float("inf")],
            labels=["Short · ≤5 min", "Medium · 6–15 min", "Long · 16–30 min", "Very long · >30 min"],
        )
        duration = videos.groupby("duration_group", observed=True, as_index=False).views.sum()
        left, right = st.columns(2)
        left.plotly_chart(red_bar(duration, "duration_group", "views", "Views by video length"), use_container_width=True)

        timeline = px.line(videos.sort_values("upload_date"), x="upload_date", y="views", markers=True,
                            hover_name="title", title="Performance over upload time")
        timeline.update_traces(line=dict(color=RED, width=3), marker=dict(size=8, color=BG, line=dict(width=2, color=RED)),
                                fill="tozeroy", fillcolor="rgba(255,0,0,.08)")
        right.plotly_chart(style_chart(timeline), use_container_width=True)

# ---------------------------------------------------------------- clustering
elif page == "Video clustering":
    page_header("Video clustering", "Videos are grouped using views and engagement so strengths and growth opportunities are easy to spot.")
    if require_data():
        videos = st.session_state.video_df.copy()
        colors = {"Best Performing": RED, "Popular but Less Interactive": AMBER,
                  "Promising": BLUE, "Needs Improvement": GRAY}
        cluster_order = ["Best Performing", "Popular but Less Interactive", "Promising", "Needs Improvement"]
        cluster_counts = videos["performance_group"].value_counts()
        summary_cols = st.columns(4)
        for col, name in zip(summary_cols, cluster_order):
            count = int(cluster_counts.get(name, 0))
            col.markdown(
                f'<div class="cluster-card" style="--cluster-color:{colors[name]}">'
                f'<div class="cluster-name">{name}</div>'
                f'<div class="cluster-count">{count} video{"s" if count != 1 else ""}</div></div>',
                unsafe_allow_html=True,
            )
        fig = px.scatter(
            videos, x="views", y="engagement_rate", color="performance_group", size="likes", hover_name="title",
            log_x=True, color_discrete_map=colors, title="Views × engagement map",
            labels={"views": "Views · logarithmic scale", "engagement_rate": "Engagement rate (%)",
                    "performance_group": "Performance group"},
            size_max=40,
        )
        fig.update_traces(marker=dict(opacity=.9, line=dict(width=1.5, color=BG), sizemin=12))
        st.plotly_chart(style_chart(fig, 500), use_container_width=True)
        st.dataframe(videos[["title", "views", "engagement_rate", "performance_group"]],
                     use_container_width=True, hide_index=True)

# ---------------------------------------------------------------- sentiment
elif page == "Comment sentiment":
    page_header("Comment sentiment", "Select a video to understand the emotional balance behind its comments.")
    if require_data():
        videos, comments = st.session_state.video_df, st.session_state.comments_df
        title = st.selectbox("Choose a video", videos.title.tolist())
        video_id = videos.loc[videos.title == title, "video_id"].iloc[0]
        selected = comments[comments.video_id == video_id]
        counts = selected.sentiment.value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0).reset_index()
        counts.columns = ["Sentiment", "Comments"]
        counts["Percentage"] = (counts.Comments / max(1, counts.Comments.sum()) * 100).round(2)
        colors = {"Positive": GREEN, "Neutral": AMBER, "Negative": RED}

        chip_cols = st.columns(3)
        for col, name in zip(chip_cols, ["Positive", "Neutral", "Negative"]):
            value = f"{counts.loc[counts.Sentiment == name, 'Percentage'].iloc[0]:.2f}%"
            col.markdown(
                f'<div class="sentiment-card"><div class="sentiment-label">'
                f'<span class="sentiment-dot" style="--sentiment-color:{colors[name]}"></span>{name}</div>'
                f'<div class="sentiment-value">{value}</div></div>',
                unsafe_allow_html=True,
            )

        left, right = st.columns(2)
        pie = px.pie(counts, names="Sentiment", values="Comments", hole=.66, title="Sentiment share",
                     color="Sentiment", color_discrete_map=colors)
        pie.update_traces(textposition="outside", textinfo="percent+label",
                           marker=dict(line=dict(color=BG, width=3)), pull=[.03, 0, .03])
        left.plotly_chart(style_chart(pie), use_container_width=True)

        bar = px.bar(counts, x="Sentiment", y="Percentage", color="Sentiment", text="Percentage",
                     title="Sentiment percentage", color_discrete_map=colors)
        bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_cornerradius=4,
                           textfont_color=MUTED)
        right.plotly_chart(style_chart(bar), use_container_width=True)

        with st.expander("Explore analysed comments"):
            st.dataframe(selected[["comment", "sentiment"]], use_container_width=True, hide_index=True)
