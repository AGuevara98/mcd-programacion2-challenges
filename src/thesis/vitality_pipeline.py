from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src.thesis.data_access import (
    get_postgis_engine,
    ingest_overture_pois_from_s3,
    save_dataframe_to_features,
)

DEFAULT_OVERTURE_URI = "s3://overturemaps-us-west-2/release/2026-02-18.0/theme=places/type=place/*"


def build_ageb_vitality_features(
    overture_s3_uri: str = DEFAULT_OVERTURE_URI,
    ageb_table: str = "base.ageb",
    output_table: str = "ageb_vitality_overture",
):
    """Aggregate Overture POI vitality features at AGEB level and save to features schema."""
    engine = get_postgis_engine()

    ageb = gpd.read_postgis(f"SELECT * FROM {ageb_table}", engine, geom_col="geom")
    if ageb.empty:
        raise ValueError(f"No AGEB data found in {ageb_table}")

    pois = ingest_overture_pois_from_s3(overture_s3_uri).to_crs("EPSG:6372")
    if pois.empty:
        out = pd.DataFrame(columns=["ageb_id", "vitality_retail", "vitality_food", "vitality_social"])
        save_dataframe_to_features(out, output_table, engine=engine)
        return out

    ageb = ageb.to_crs("EPSG:6372")
    if "ageb_id" not in ageb.columns:
        if "cvegeo" in ageb.columns:
            ageb = ageb.rename(columns={"cvegeo": "ageb_id"})
        elif "CVEGEO" in ageb.columns:
            ageb = ageb.rename(columns={"CVEGEO": "ageb_id"})
        else:
            raise ValueError("AGEB table must contain ageb_id or cvegeo/CVEGEO")

    joined = gpd.sjoin(
        pois[["category", "geometry"]],
        ageb[["ageb_id", "geom"]].rename(columns={"geom": "geometry"}),
        how="inner",
        predicate="within",
    )

    if joined.empty:
        out = pd.DataFrame(columns=["ageb_id", "vitality_retail", "vitality_food", "vitality_social"])
        save_dataframe_to_features(out, output_table, engine=engine)
        return out

    pivot = (
        joined.groupby(["ageb_id", "category"]).size().unstack(fill_value=0).reset_index()
    )

    out = pd.DataFrame()
    out["ageb_id"] = pivot["ageb_id"]
    out["vitality_retail"] = pivot.get("retail", 0)
    out["vitality_food"] = pivot.get("food_and_drink", 0)
    out["vitality_social"] = pivot.get("social", 0)

    save_dataframe_to_features(out, output_table, engine=engine)
    return out
