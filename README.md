# youtube-analytics
# YouTube Analytics Dashboard

An interactive data analytics application that extracts information from YouTube videos and presents video performance, engagement, category analysis, clustering, and comment sentiment through Streamlit dashboards.

## Project Objective

The objective of this project is to analyse multiple YouTube videos using their URLs. The application retrieves video statistics and comments through the YouTube Data API, calculates engagement metrics, groups videos based on their performance, and analyses comment sentiment using a pretrained RoBERTa model.

Users can submit between 5 and 20 YouTube video URLs for analysis.

## Main Features

* Accepts 5 to 20 YouTube video URLs
* Validates different YouTube URL formats
* Extracts video details using the YouTube Data API
* Extracts a maximum of 100 comments per video
* Calculates the engagement rate of each video
* Groups videos into four performance clusters
* Performs comment sentiment analysis
* Displays interactive dashboards and charts
* Allows users to select a video for individual sentiment analysis

## Dashboard Sections

### 1. Overview Dashboard

The Overview dashboard displays:

* Total number of videos
* Total views
* Total likes
* Total comments
* Average engagement rate
* Top-performing video
* Video details table

### 2. Video Performance Dashboard

The Video Performance dashboard contains:

* Top 10 videos by views
* Likes by video
* Comments by video
* Engagement rate by video

### 3. Category and Time Dashboard

The Category and Time dashboard contains:

* Views by category
* Likes by category
* Comments by category
* Category versus engagement rate
* Views by video duration
* Video uploads over time

### 4. Video Clustering Dashboard

K-Means clustering is used to group videos based on:

* Number of views
* Engagement rate

The videos are divided into four performance groups:

* Best Performing
* Popular but Less Interactive
* Promising
* Needs Improvement

### 5. Comment Sentiment Dashboard

The Comment Sentiment dashboard allows the user to select a video and view:

* Positive comment percentage
* Neutral comment percentage
* Negative comment percentage
* Sentiment distribution pie chart
* Sentiment percentage bar chart
* Analysed comments and their predicted sentiment

## Technologies Used

* Python
* Streamlit
* Pandas
* Plotly
* YouTube Data API v3
* Hugging Face Transformers
* RoBERTa
* Scikit-learn
* K-Means Clustering
* MySQL
* SQLAlchemy
* PyMySQL
* Git and GitHub

## Machine Learning Techniques

### Sentiment Analysis

The project uses the pretrained model:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

The model classifies YouTube comments into:

* Positive
* Neutral
* Negative

A maximum of 100 comments is extracted from each video to reduce processing time.

### Video Clustering

K-Means clustering groups videos using:

* Views
* Engagement rate

Before clustering:

* Log transformation is applied to views.
* StandardScaler is used to scale the features.
* Four clusters are created using K-Means.

## Engagement Rate Formula

```text
Engagement Rate = ((Likes + Comments) / Views) × 100
```

## Project Structure

```text
youtube-analytics/
│
├── app.py
├── pipeline.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
└── src/
    ├── __init__.py
    ├── youtube_extractor.py
    ├── sentiment.py
    ├── analytics.py
    ├── clustering.py
    └── database.py
```

## Environment Variables

Create a `.env` file inside the main project folder and add your YouTube API key:

```env
YOUTUBE_API_KEY=your_youtube_api_key
```

If MySQL storage is used, add the following details:

```env
MYSQL_HOST=localhost
MYSQL_USER=your_mysql_username
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=youtube_analytics
```

Do not upload the `.env` file to GitHub because it contains private credentials.

## Installation

Create and activate a Python virtual environment:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
```

Install the required libraries:

```powershell
pip install -r requirements.txt
```

## How to Run the Application

Activate the virtual environment:

```powershell
.\venv\Scripts\activate
```

Start the Streamlit application:

```powershell
streamlit run app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

Paste 5 to 20 public YouTube video URLs and click **Analyze Videos**.

## Supported YouTube URL Formats

```text
https://www.youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
https://youtube.com/shorts/VIDEO_ID
https://youtube.com/embed/VIDEO_ID
```

## Important Notes

* The application requires an active internet connection.
* A valid YouTube Data API key is required.
* Private, deleted, restricted, or unavailable videos cannot be analysed.
* Videos with disabled comments may not produce sentiment results.
* The first execution may take additional time because the RoBERTa model must be downloaded.
* A maximum of 100 comments is processed per video to improve speed.

## Author

**Sangeethapriya Sivakumar**

B.Sc. Computer Science Graduate
Aspiring Data Science Professional

## Repository

[YouTube Analytics GitHub Repository](https://github.com/Sangeethapriya03/youtube-analytics)
