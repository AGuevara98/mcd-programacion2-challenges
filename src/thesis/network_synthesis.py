from __future__ import annotations

from typing import Iterable, List
from datetime import datetime

import geopandas as gpd
import networkx as nx
import osmnx as ox
from networkx.algorithms.approximation import steiner_tree

from src.thesis.data_access import get_postgis_engine, save_dataframe_to_features


EPSILON = 1e-6


def _progress(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [steiner] {msg}", flush=True)


def _progress_every(total: int, chunks: int) -> int:
    return max(1, total // max(1, chunks))


def build_drive_graph_from_polygon(polygon_gdf: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    """Build routable street graph from a study-area polygon."""
    if polygon_gdf.empty:
        raise ValueError("polygon_gdf is empty")

    _progress("Building study-area polygon in EPSG:4326")
    poly = polygon_gdf.to_crs(4326).geometry.unary_union
    _progress("Downloading/building OSMnx drive graph (this can take several minutes)")
    graph = ox.graph_from_polygon(poly, network_type="drive", simplify=True)
    _progress("Projecting graph to EPSG:6372")
    projected = ox.project_graph(graph, to_crs="EPSG:6372")
    _progress(f"Graph ready: {projected.number_of_nodes()} nodes, {projected.number_of_edges()} edges")
    return projected


def add_suitability_edge_cost(
    graph: nx.MultiDiGraph,
    suitability_gdf: gpd.GeoDataFrame,
    suitability_col: str = "suitability_score",
    node_id_col: str = "nearest_node",
    progress_chunks: int = 10,
) -> nx.MultiDiGraph:
    """Set edge cost = distance * (1 / suitability) based on origin-node suitability."""
    if suitability_col not in suitability_gdf.columns:
        raise ValueError(f"Missing suitability column: {suitability_col}")

    _progress("Preparing suitability GeoDataFrame in EPSG:6372")
    gdf = suitability_gdf.to_crs("EPSG:6372").copy()
    _progress("Extracting graph nodes GeoDataFrame")
    nodes_gdf, _ = ox.graph_to_gdfs(graph)

    # Attach nearest graph node to each AGEB centroid.
    _progress("Computing AGEB centroids")
    gdf["centroid"] = gdf.geometry.centroid
    _progress(f"Mapping {len(gdf)} centroids to nearest graph nodes")
    nearest_nodes = []
    progress_every = _progress_every(len(gdf), progress_chunks)
    for idx, row in enumerate(gdf.itertuples(index=False), start=1):
        nearest_nodes.append(ox.distance.nearest_nodes(graph, row.centroid.x, row.centroid.y))
        if idx % progress_every == 0 or idx == len(gdf):
            _progress(f"Nearest-node mapping progress: {idx}/{len(gdf)}")
    gdf[node_id_col] = nearest_nodes

    _progress("Aggregating suitability by nearest node")
    suit_by_node = (
        gdf.groupby(node_id_col)[suitability_col]
        .mean()
        .clip(lower=EPSILON)
        .to_dict()
    )

    _progress("Applying suitability-based edge cost")
    g2 = graph.copy()
    total_edges = g2.number_of_edges()
    progress_every = _progress_every(total_edges, progress_chunks)
    for idx, (u, v, k, data) in enumerate(g2.edges(keys=True, data=True), start=1):
        length = float(data.get("length", 1.0))
        su = float(suit_by_node.get(u, EPSILON))
        data["cost"] = length * (1.0 / su)
        if idx % progress_every == 0 or idx == total_edges:
            _progress(f"Edge cost progress: {idx}/{total_edges}")

    # Keep a light reference for downstream terminal selection.
    g2.graph["suitability_gdf"] = gdf.drop(columns=["centroid"]) if "centroid" in gdf.columns else gdf
    g2.graph["nodes_gdf"] = nodes_gdf
    _progress("Suitability edge weighting complete")
    return g2


def select_terminals(
    suitability_gdf: gpd.GeoDataFrame,
    top_k: int = 12,
    suitability_col: str = "suitability_score",
) -> List[int]:
    gdf = suitability_gdf.copy()
    if "nearest_node" not in gdf.columns:
        raise ValueError("Expected nearest_node column. Run add_suitability_edge_cost first.")
    ranked = gdf.sort_values(suitability_col, ascending=False).head(top_k)
    return ranked["nearest_node"].astype(int).tolist()


def steiner_connect_terminals(
    graph: nx.MultiDiGraph,
    terminals: Iterable[int],
    weight: str = "cost",
    progress_chunks: int = 10,
) -> nx.Graph:
    terminals = list(dict.fromkeys(terminals))
    if len(terminals) < 2:
        raise ValueError("At least 2 terminals are required for Steiner tree synthesis")

    _progress(f"Building undirected graph for Steiner approximation with {len(terminals)} terminals")
    # Steiner approximation runs on undirected graph.
    g_und = nx.Graph()
    total_edges = graph.number_of_edges()
    progress_every = _progress_every(total_edges, progress_chunks)
    for idx, (u, v, data) in enumerate(graph.edges(data=True), start=1):
        w = float(data.get(weight, data.get("length", 1.0)))
        if g_und.has_edge(u, v):
            if w < g_und[u][v][weight]:
                g_und[u][v][weight] = w
        else:
            g_und.add_edge(u, v, **{weight: w})
        if idx % progress_every == 0 or idx == total_edges:
            _progress(f"Undirected build progress: {idx}/{total_edges}")

    valid_terminals = [node for node in terminals if node in g_und]
    missing_terminals = [node for node in terminals if node not in g_und]
    if missing_terminals:
        _progress(
            f"Skipping {len(missing_terminals)} terminals not present in graph: {missing_terminals[:5]}"
        )
    if len(valid_terminals) < 2:
        raise ValueError("At least 2 valid terminals are required for Steiner tree synthesis")

    _progress("Running Steiner tree approximation")
    result = steiner_tree(g_und, valid_terminals, weight=weight)
    _progress(f"Steiner result: {result.number_of_nodes()} nodes, {result.number_of_edges()} edges")
    return result


def steiner_edges_to_gdf(graph: nx.MultiDiGraph, steiner_g: nx.Graph) -> gpd.GeoDataFrame:
    _progress("Extracting Steiner edges to GeoDataFrame")
    _, edges = ox.graph_to_gdfs(graph)
    pairs = {(u, v) for u, v in steiner_g.edges()} | {(v, u) for u, v in steiner_g.edges()}

    kept = edges.loc[[idx for idx in edges.index if (idx[0], idx[1]) in pairs]].copy()
    if kept.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:6372")
    _progress(f"Steiner corridor GeoDataFrame contains {len(kept)} edges")
    return kept.reset_index()


def run_steiner_from_postgis(
    top_k: int = 12,
    suitability_table: str = "features.thesis_ageb_scored",
    boundary_table: str = "base.ageb",
    output_table: str = "thesis_steiner_corridors",
    progress_chunks: int = 10,
):
    """Run full phase-3 network synthesis using PostGIS inputs and persist result in features schema."""
    _progress("Initializing Steiner run from PostGIS")
    engine = get_postgis_engine()

    boundary_query = f"SELECT ST_Union(geom) AS geom FROM {boundary_table}"
    _progress(f"Loading boundary geometry from {boundary_table}")
    boundary = gpd.read_postgis(boundary_query, engine, geom_col="geom")
    if boundary.empty:
        raise ValueError(f"No boundary geometry found in {boundary_table}")

    suitability_query = f"SELECT * FROM {suitability_table}"
    _progress(f"Loading suitability rows from {suitability_table}")
    suitability = gpd.read_postgis(suitability_query, engine, geom_col="geom")
    if suitability.empty:
        raise ValueError(f"No suitability rows found in {suitability_table}")
    _progress(f"Loaded {len(suitability)} suitability rows")

    graph = build_drive_graph_from_polygon(boundary)
    weighted_graph = add_suitability_edge_cost(
        graph,
        suitability,
        progress_chunks=progress_chunks,
    )
    _progress(f"Selecting top {top_k} terminals")
    terminals = select_terminals(weighted_graph.graph["suitability_gdf"], top_k=top_k)
    steiner_g = steiner_connect_terminals(
        weighted_graph,
        terminals,
        progress_chunks=progress_chunks,
    )
    corridor_gdf = steiner_edges_to_gdf(weighted_graph, steiner_g)

    if corridor_gdf.empty:
        raise ValueError("Steiner corridor output is empty")

    save_dataframe_to_features(corridor_gdf, output_table, engine=engine)
    _progress(f"Saved corridor output to features.{output_table}")
    return corridor_gdf
