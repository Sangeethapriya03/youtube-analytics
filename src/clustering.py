import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def add_performance_clusters(video_df):
    """
    Group videos using views and engagement rate.
    """

    if len(video_df) < 4:
        raise ValueError(
            "At least 4 videos are required for clustering."
        )

    video_df = video_df.copy()

    # Select clustering features
    features = video_df[
        ["views", "engagement_rate"]
    ].copy()

    # Reduce the effect of extremely large view counts
    features["views"] = np.log1p(features["views"])

    # Scale both features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Create four clusters
    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    video_df["cluster"] = model.fit_predict(
        scaled_features
    )

    # Convert cluster centres back to original values
    cluster_centres = scaler.inverse_transform(
        model.cluster_centers_
    )

    cluster_centres[:, 0] = np.expm1(
        cluster_centres[:, 0]
    )

    median_views = video_df["views"].median()
    median_engagement = video_df[
        "engagement_rate"
    ].median()

    performance_mapping = {}

    for cluster_number, centre in enumerate(cluster_centres):

        centre_views = centre[0]
        centre_engagement = centre[1]

        if (
            centre_views >= median_views
            and centre_engagement >= median_engagement
        ):
            label = "Best Performing"

        elif (
            centre_views >= median_views
            and centre_engagement < median_engagement
        ):
            label = "Popular but Less Interactive"

        elif (
            centre_views < median_views
            and centre_engagement >= median_engagement
        ):
            label = "Promising"

        else:
            label = "Needs Improvement"

        performance_mapping[cluster_number] = label

    video_df["performance_group"] = (
        video_df["cluster"].map(performance_mapping)
    )

    return video_df