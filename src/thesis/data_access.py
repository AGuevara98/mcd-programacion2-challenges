from __future__ import annotations

import os
import subprocess
from pathlib import Path
from urllib.parse import quote_plus

import duckdb
import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine, text

# Import credentials from local config file (not committed to git)
from src.thesis.credentials import PG_USER, PG_PASS, PG_HOST, PG_PORT, PG_DB

POSTGIS_CONN_URL = (
    f"postgresql://{quote_plus(PG_USER)}:{quote_plus(PG_PASS)}@{PG_HOST}:{PG_PORT}/{PG_DB}"
)
DEFAULT_AGEB_QUERY = "SELECT * FROM features.station_nprv_audit"
DEFAULT_AGEB_FALLBACK_QUERY = "SELECT * FROM features.master_suitability"
DEFAULT_SYNTH_QUERY = """
SELECT
    b.cvegeo AS ageb_id,
    b.geom,
    COALESCE(rs.route_km_within_800m, 0.0) AS node_connectivity,
    COALESCE(ea.denue_units_total, em.total_establishments, 0.0) AS place_population,
    COALESCE(em.employment_proxy, ea.jobs_proxy_sum, 0.0) AS place_employment,
    COALESCE(acc.min_stop_dist_m, 0.0) AS real_estate_proxy,
    COALESCE(ea.denue_retail, 0.0) AS vitality_retail,
    COALESCE(ea.denue_education, 0.0) AS vitality_food,
    COALESCE(ea.denue_health, 0.0) + COALESCE(ea.denue_government, 0.0) AS vitality_social
FROM base.ageb b
LEFT JOIN features.ageb_accessibility acc ON acc.ageb_id = b.cvegeo
LEFT JOIN features.ageb_employment em ON em.ageb_id = b.cvegeo
LEFT JOIN features.ageb_route_supply rs ON rs.ageb_id = b.cvegeo
LEFT JOIN features.ageb_economic_activity ea ON ea.ageb_id = b.cvegeo
"""


def _build_conn_url(host: str) -> str:
    return f"postgresql://{quote_plus(PG_USER)}:{quote_plus(PG_PASS)}@{host}:{PG_PORT}/{PG_DB}"


def _get_ubuntu_wsl_ip() -> str | None:
    try:
        proc = subprocess.run(
            ["wsl", "-d", "Ubuntu", "-e", "sh", "-lc", "hostname -I | awk '{print $1}'"],
            check=True,
            capture_output=True,
            text=True,
        )
        ip = proc.stdout.strip()
        return ip or None
    except Exception:
        return None


def get_postgis_engine(conn_url: str | None = None):
    base_url = conn_url or os.getenv("THESIS_POSTGIS_URL") or POSTGIS_CONN_URL
    candidate_urls = []

    if "@localhost:" in base_url or "@127.0.0.1:" in base_url:
        candidate_urls.append(base_url)
        wsl_ip = _get_ubuntu_wsl_ip()
        if wsl_ip:
            candidate_urls.append(_build_conn_url(wsl_ip))
    else:
        candidate_urls.append(base_url)

    last_exc = None
    for url in candidate_urls:
        try:
            engine = create_engine(url, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return engine
        except Exception as exc:
            last_exc = exc

    if last_exc:
        raise last_exc
    raise RuntimeError("Unable to initialize PostGIS engine")


def _try_read_geodataframe(query: str, engine):
    try:
        return gpd.read_postgis(query, engine, geom_col="geom")
    except Exception:
        return None


def load_thesis_input(data_path: str | None, engine=None) -> pd.DataFrame:
    """Load thesis input from CSV, SQL query, or default PostGIS table."""
    if engine is None:
        engine = get_postgis_engine()

    # CSV path mode
    if data_path and Path(data_path).exists():
        return pd.read_csv(data_path)

    # SQL mode: pass as sql:<query>
    if data_path and data_path.startswith("sql:"):
        query = data_path[len("sql:") :].strip()
        gdf = _try_read_geodataframe(query, engine)
        if gdf is not None:
            return gdf
        return pd.read_sql_query(text(query), engine)

    # Table mode: pass as table:schema.table
    if data_path and data_path.startswith("table:"):
        table_name = data_path[len("table:") :].strip()
        query = f"SELECT * FROM {table_name}"
        gdf = _try_read_geodataframe(query, engine)
        if gdf is not None:
            return gdf
        return pd.read_sql_query(text(query), engine)

    # Default mode: PostGIS station audit table, then fallback.
    for query in (DEFAULT_AGEB_QUERY, DEFAULT_AGEB_FALLBACK_QUERY):
        try:
            gdf = _try_read_geodataframe(query, engine)
            if gdf is not None and not gdf.empty:
                return gdf
        except Exception:
            pass

        try:
            df = pd.read_sql_query(text(query), engine)
            if not df.empty:
                return df
        except Exception:
            continue

    # Synthesized fallback from feature component tables.
    synth_gdf = _try_read_geodataframe(DEFAULT_SYNTH_QUERY, engine)
    if synth_gdf is not None and not synth_gdf.empty:
        return synth_gdf

    try:
        synth_df = pd.read_sql_query(text(DEFAULT_SYNTH_QUERY), engine)
        if not synth_df.empty:
            return synth_df
    except Exception:
        pass

    raise ValueError(
        "No thesis input found. Provide CSV path, sql:<query>, table:<schema.table>, or populate features.station_nprv_audit/master_suitability."
    )


def save_dataframe_to_features(df: pd.DataFrame, table_name: str, engine=None) -> None:
    """Save DataFrame or GeoDataFrame to features schema in PostGIS."""
    if engine is None:
        engine = get_postgis_engine()

    def _write(target_engine):
        if isinstance(df, gpd.GeoDataFrame):
            df.to_postgis(table_name, target_engine, schema="features", if_exists="replace", index=False)
            return

        # Try geopandas writer if a geometry column exists and can be promoted to GeoDataFrame.
        if "geom" in df.columns and "geometry" not in df.columns:
            try:
                gdf = gpd.GeoDataFrame(df, geometry="geom", crs="EPSG:6372")
                gdf.to_postgis(table_name, target_engine, schema="features", if_exists="replace", index=False)
                return
            except Exception:
                pass

        df.to_sql(table_name, target_engine, schema="features", if_exists="replace", index=False)

    try:
        _write(engine)
    except Exception:
        # Reconnect once in case WSL networking or pooled connection dropped.
        fresh_engine = get_postgis_engine()
        _write(fresh_engine)


def get_duckdb_connection(db_file: str = "zmg_thesis.db") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_file)
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(
        """
        CREATE OR REPLACE SECRET (
            TYPE S3,
            KEY_ID '',
            SECRET '',
            REGION 'us-west-2'
        );
        """
    )
    return con


def ingest_overture_pois_from_s3(
    s3_uri: str,
    category_filter: tuple[str, ...] = ("retail", "food_and_drink", "social"),
    db_file: str = "zmg_thesis.db",
) -> gpd.GeoDataFrame:
    """Load Overture POIs from S3 via DuckDB and return as GeoDataFrame.

    Uses ST_AsWKB for safe geometry handoff to GeoPandas.
    """
    con = get_duckdb_connection(db_file=db_file)
    categories = ", ".join([f"'{c}'" for c in category_filter])

    query = f"""
    SELECT
        id,
        names.primary AS name,
        categories.primary AS category,
        ST_AsWKB(geometry) AS geom
    FROM read_parquet('{s3_uri}')
    WHERE categories.primary IN ({categories})
      AND geometry IS NOT NULL
    """

    res = con.execute(query).fetch_df()
    if res.empty:
        return gpd.GeoDataFrame(columns=["id", "name", "category", "geometry"], geometry="geometry", crs="EPSG:4326")

    gdf = gpd.GeoDataFrame(
        res.drop(columns=["geom"]),
        geometry=gpd.GeoSeries.from_wkb(res["geom"]),
        crs="EPSG:4326",
    )
    return gdf
