import pytest
import numpy as np
import pandas as pd
from backend.app.pipelines.st_dbscan import STDBSCAN


@pytest.fixture
def model():
    """Returns standard STDBSCAN instance."""
    return STDBSCAN(eps1=10.0, eps2=2.0, min_samples=3)


@pytest.fixture
def base_timestamp():
    return pd.Timestamp("2026-08-01 12:00:00")


# ============================================================================
# Edge Cases & Boundary Conditions
# ============================================================================

def test_empty_dataframe(model):
    """Empty dataframe should return an empty numpy array without throwing errors."""
    df = pd.DataFrame(columns=['latitude', 'longitude', 'timestamp'])
    labels = model.fit_predict(df)
    assert isinstance(labels, np.ndarray)
    assert len(labels) == 0


def test_single_point(model):
    """A single point cannot satisfy min_samples=3 and must be marked as noise (-1)."""
    df = pd.DataFrame([{
        'latitude': 37.7749,
        'longitude': -122.4194,
        'timestamp': '2026-08-01 12:00:00'
    }])
    labels = model.fit_predict(df)
    assert len(labels) == 1
    assert labels[0] == -1


def test_fewer_points_than_min_samples(model):
    """2 points close together with min_samples=3 should both be marked as noise (-1)."""
    df = pd.DataFrame([
        {'latitude': 37.7749, 'longitude': -122.4194, 'timestamp': '2026-08-01 12:00:00'},
        {'latitude': 37.7750, 'longitude': -122.4195, 'timestamp': '2026-08-01 13:00:00'},
    ])
    labels = model.fit_predict(df)
    assert np.all(labels == -1)


# ============================================================================
# Core Spatio-Temporal Logic
# ============================================================================

def test_spatial_proximity_temporal_gap(model, base_timestamp):
    """Points close in space (~0.1km) but separated by time > eps2 (10 days) must NOT cluster."""
    df = pd.DataFrame([
        {'latitude': 37.7749, 'longitude': -122.4194, 'timestamp': base_timestamp},
        {'latitude': 37.7750, 'longitude': -122.4195, 'timestamp': base_timestamp + pd.Timedelta(days=10)},
        {'latitude': 37.7751, 'longitude': -122.4196, 'timestamp': base_timestamp + pd.Timedelta(days=20)},
    ])
    labels = model.fit_predict(df)
    assert np.all(labels == -1)


def test_temporal_proximity_spatial_gap(model, base_timestamp):
    """Points at identical timestamps but spatially > eps1 (50km apart) must NOT cluster."""
    df = pd.DataFrame([
        {'latitude': 37.7749, 'longitude': -122.4194, 'timestamp': base_timestamp},  # San Francisco
        {'latitude': 37.3382, 'longitude': -121.8863, 'timestamp': base_timestamp},  # San Jose (~70km away)
        {'latitude': 38.5815, 'longitude': -121.4944, 'timestamp': base_timestamp},  # Sacramento (~140km away)
    ])
    labels = model.fit_predict(df)
    assert np.all(labels == -1)


def test_valid_single_cluster(model, base_timestamp):
    """4 points satisfying both eps1 (10km) and eps2 (2 days) with min_samples=3 must form cluster 0."""
    df = pd.DataFrame([
        {'latitude': 37.7749, 'longitude': -122.4194, 'timestamp': base_timestamp},
        {'latitude': 37.7752, 'longitude': -122.4198, 'timestamp': base_timestamp + pd.Timedelta(hours=4)},
        {'latitude': 37.7755, 'longitude': -122.4201, 'timestamp': base_timestamp + pd.Timedelta(hours=12)},
        {'latitude': 37.7748, 'longitude': -122.4190, 'timestamp': base_timestamp + pd.Timedelta(days=1)},
    ])
    labels = model.fit_predict(df)
    assert np.all(labels == 0)


def test_two_distinct_clusters_and_noise(model, base_timestamp):
    """Tests isolation between two valid clusters and an isolated noise point."""
    cluster1 = [
        {'latitude': 37.7749, 'longitude': -122.4194, 'timestamp': base_timestamp},
        {'latitude': 37.7750, 'longitude': -122.4195, 'timestamp': base_timestamp + pd.Timedelta(hours=2)},
        {'latitude': 37.7751, 'longitude': -122.4196, 'timestamp': base_timestamp + pd.Timedelta(hours=4)},
    ]
    
    # Spatially far away (New York ~4000km)
    cluster2 = [
        {'latitude': 40.7128, 'longitude': -74.0060, 'timestamp': base_timestamp},
        {'latitude': 40.7129, 'longitude': -74.0061, 'timestamp': base_timestamp + pd.Timedelta(hours=1)},
        {'latitude': 40.7130, 'longitude': -74.0062, 'timestamp': base_timestamp + pd.Timedelta(hours=3)},
    ]
    
    # Isolated point
    noise = [{'latitude': 0.0, 'longitude': 0.0, 'timestamp': base_timestamp + pd.Timedelta(days=50)}]

    df = pd.DataFrame(cluster1 + cluster2 + noise)
    labels = model.fit_predict(df)

    assert labels[0] == labels[1] == labels[2] == 0
    assert labels[3] == labels[4] == labels[5] == 1
    assert labels[6] == -1


# ============================================================================
# Schema & Interface Tests
# ============================================================================

def test_custom_column_names(model, base_timestamp):
    """Ensures algorithm works when given custom dataframe column names."""
    df = pd.DataFrame([
        {'lat': 37.7749, 'lng': -122.4194, 'date': base_timestamp},
        {'lat': 37.7750, 'lng': -122.4195, 'date': base_timestamp + pd.Timedelta(hours=2)},
        {'lat': 37.7751, 'lng': -122.4196, 'date': base_timestamp + pd.Timedelta(hours=4)},
    ])
    
    labels = model.fit_predict(df, lat_col='lat', lon_col='lng', time_col='date')
    assert np.all(labels == 0)


def test_border_point_relabeling(base_timestamp):
    """
    Ensures points evaluated early as noise get relabeled to cluster ID 
    when encountered later as a border point of a core point.
    """
    model = STDBSCAN(eps1=10.0, eps2=2.0, min_samples=3)
    
    # Point 0 is initially evaluated with < 3 neighbors if processed first, 
    # but connects to core points 1, 2, 3
    df = pd.DataFrame([
        {'latitude': 37.7740, 'longitude': -122.4190, 'timestamp': base_timestamp},                  # Border
        {'latitude': 37.7749, 'longitude': -122.4194, 'timestamp': base_timestamp},                  # Core
        {'latitude': 37.7750, 'longitude': -122.4195, 'timestamp': base_timestamp + pd.Timedelta(hours=1)}, # Core
        {'latitude': 37.7751, 'longitude': -122.4196, 'timestamp': base_timestamp + pd.Timedelta(hours=2)}, # Core
    ])
    
    labels = model.fit_predict(df)
    assert np.all(labels == 0)


# ============================================================================
# Cluster Summary Tests
# ============================================================================

def test_generate_cluster_summary_not_fitted(model, base_timestamp):
    """Should raise error if summary is requested before fit_predict."""
    df = pd.DataFrame([{
        'latitude': 37.7749,
        'longitude': -122.4194,
        'timestamp': base_timestamp
    }])
    with pytest.raises(ValueError, match="Model must be fitted"):
        model.generate_cluster_summary(df)


def test_generate_cluster_summary_valid(model, base_timestamp):
    """Should correctly summarize a valid cluster and ignore noise."""
    # 3 points forming cluster 0, and 1 isolated point
    df = pd.DataFrame([
        {'latitude': 37.7749, 'longitude': -122.4194, 'timestamp': base_timestamp},
        {'latitude': 37.7750, 'longitude': -122.4195, 'timestamp': base_timestamp + pd.Timedelta(hours=2)},
        {'latitude': 37.7751, 'longitude': -122.4196, 'timestamp': base_timestamp + pd.Timedelta(hours=4)},
        {'latitude': 0.0, 'longitude': 0.0, 'timestamp': base_timestamp + pd.Timedelta(days=50)} # Noise
    ])
    
    model.fit_predict(df)
    summary_df = model.generate_cluster_summary(df)
    
    assert not summary_df.empty
    assert len(summary_df) == 1
    assert summary_df.iloc[0]['cluster_id'] == 0
    assert summary_df.iloc[0]['count'] == 3
    assert np.isclose(summary_df.iloc[0]['mean_latitude'], 37.7750)
    assert np.isclose(summary_df.iloc[0]['mean_longitude'], -122.4195)
    assert summary_df.iloc[0]['duration_days'] == 4.0 / 24.0 # 4 hours


def test_generate_cluster_summary_empty(model):
    """Should return empty dataframe if no clusters are formed (only noise or empty input)."""
    df = pd.DataFrame([
        {'latitude': 37.7749, 'longitude': -122.4194, 'timestamp': '2026-08-01 12:00:00'},
        {'latitude': 37.7750, 'longitude': -122.4195, 'timestamp': '2026-08-01 13:00:00'},
    ])
    model.fit_predict(df)
    summary_df = model.generate_cluster_summary(df)
    
    assert summary_df.empty


def test_generate_cluster_summary_custom_columns(model, base_timestamp):
    """Ensures summary generation works with custom column names."""
    df = pd.DataFrame([
        {'lat': 37.7749, 'lng': -122.4194, 'date': base_timestamp},
        {'lat': 37.7750, 'lng': -122.4195, 'date': base_timestamp + pd.Timedelta(hours=2)},
        {'lat': 37.7751, 'lng': -122.4196, 'date': base_timestamp + pd.Timedelta(hours=4)},
    ])
    
    model.fit_predict(df, lat_col='lat', lon_col='lng', time_col='date')
    summary_df = model.generate_cluster_summary(df, lat_col='lat', lon_col='lng', time_col='date')
    
    assert not summary_df.empty
    assert summary_df.iloc[0]['count'] == 3