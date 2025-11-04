# paris_bnb_map.py
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional imports for richer map (folium)
try:
    import folium
    from streamlit_folium import st_folium, folium_static
    from folium.plugins import MarkerCluster
    HAS_FOLIUM = True
except Exception:
    HAS_FOLIUM = False
    st_folium = None
    folium_static = None

# Pydeck fallback
try:
    import pydeck as pdk
    HAS_PYDECK = True
except Exception:
    HAS_PYDECK = False

# DB import for fallback load-from-db & fetching single listing on click
try:
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except Exception:
    HAS_SQLALCHEMY = False

st.set_page_config(page_title="Paris Airbnb Map", layout="wide")
st.title("Paris — Airbnb listings map")

# ---------------- Configuration ----------------
# Provide a full dataset file (CSV/Parquet) if you want full-row downloads on click.
DATA_FILE = os.getenv("DATA_FILE", "data/listings.csv")
DB_URL = os.getenv("AIRBNB_DB_URL", "postgresql://airbnb:airbnb@localhost:5432/airbnb")
TABLE_SCHEMA = os.getenv("LISTINGS_SCHEMA", "clean")
TABLE_NAME = os.getenv("LISTINGS_TABLE", "listings_features")

# Minimal columns we need up front
MINIMAL_COLUMNS = ["id", "latitude", "longitude", "room_type", "property_type", "title"]

# Candidate column names for each semantic field (common variants)
CANDIDATES = {
    "id": ["id", "listing_id", "listingId"],
    "latitude": ["latitude", "lat", "lat_wgs84", "y"],
    "longitude": ["longitude", "lon", "lng", "lon_wgs84", "x"],
    "room_type": ["room_type", "roomType", "room_type_clean"],
    "property_type": ["property_type_slim", "property_type", "propertyType", "property_type_clean"],
    "title": ["name", "title", "listing_title", "headline"],
}


def _detect_column_map_from_file_columns(columns: List[str]) -> dict:
    """Return a mapping logical -> actual column name for columns present in file."""
    mapping = {}
    for logical, candidates in CANDIDATES.items():
        for cand in candidates:
            if cand in columns:
                mapping[logical] = cand
                break
    return mapping


def _normalize_min_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with normalized minimal columns: id, latitude, longitude, room_type, property_type, title."""
    colmap: Dict[str, str] = {}
    for logical, candidates in CANDIDATES.items():
        for c in candidates:
            if c in df.columns:
                colmap[logical] = c
                break

    out = pd.DataFrame()
    # id: if absent, use index
    if colmap.get("id") and colmap["id"] in df.columns:
        out["id"] = df[colmap["id"]]
    else:
        out["id"] = df.index.astype(str)

    out["latitude"] = df[colmap["latitude"]] if colmap.get("latitude") and colmap["latitude"] in df.columns else None
    out["longitude"] = df[colmap["longitude"]] if colmap.get("longitude") and colmap["longitude"] in df.columns else None

    out["room_type"] = df[colmap["room_type"]] if colmap.get("room_type") and colmap["room_type"] in df.columns else None
    out["property_type"] = df[colmap["property_type"]] if colmap.get("property_type") and colmap["property_type"] in df.columns else None
    out["title"] = df[colmap["title"]] if colmap.get("title") and colmap["title"] in df.columns else None

    # coerce numeric lat/lon and drop invalids
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    for c in ("room_type", "property_type", "title"):
        out[c] = out[c].astype(object).where(out[c].notnull(), None)

    return out


@st.cache_data(ttl=600)
def load_minimal_from_file() -> Optional[pd.DataFrame]:
    """
    Load only minimal columns from DATA_FILE (CSV or Parquet).
    This avoids loading the entire dataset when the file contains many columns.
    Returns None if file missing or load failed.
    """
    p = Path(DATA_FILE)
    if not p.exists():
        return None
    try:
        if p.suffix.lower() in (".parquet", ".parq"):
            # parquet supports column projection
            # try to map candidate columns present in file -> select them
            # read metadata columns first
            import pyarrow.parquet as pq  # optional dependency but usually available with parquet files
            table = pq.ParquetFile(str(p))
            file_cols = table.schema.names
            colmap = _detect_column_map_from_file_columns(file_cols)
            selected = [colmap[l] for l in colmap.keys() if l in colmap]
            # ensure we always select at least lat & lon variants if present
            if selected:
                df = pd.read_parquet(p, columns=selected)
            else:
                df = pd.read_parquet(p, columns=None)  # fallback to all
        else:
            # For CSV use usecols to limit columns read
            # Peek header
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                header = fh.readline()
            # simple header parse
            cols = [c.strip() for c in header.strip().split(",")]
            colmap = _detect_column_map_from_file_columns(cols)
            usecols = [colmap[l] for l in colmap.keys() if l in colmap]
            if usecols:
                df = pd.read_csv(p, usecols=usecols, low_memory=False)
            else:
                # fallback: read only first N rows to avoid full read (safe fallback)
                df = pd.read_csv(p, nrows=5000, low_memory=False)
        return df
    except Exception as exc:
        st.warning(f"Failed to read minimal columns from DATA_FILE {DATA_FILE}: {exc}")
        return None


def _quote_identifier(name: str) -> str:
    return f'"{name}"'


def _build_min_select_sql(available_cols: List[str]) -> Tuple[str, Optional[str]]:
    """
    Build a SQL string selecting minimal columns using available columns.
    Returns (sql, picked_id_colname_or_None)
    """
    def pick_candidates_for_logical(logical: str) -> List[str]:
        return [c for c in CANDIDATES[logical] if c in available_cols]

    def col_expr(cname: str) -> str:
        return f"l.{cname}"

    select_parts: List[str] = []

    id_candidates = pick_candidates_for_logical("id")
    if id_candidates:
        select_parts.append(f"{col_expr(id_candidates[0])} AS id")
        picked_id = id_candidates[0]
    else:
        select_parts.append("NULL AS id")
        picked_id = None

    lat_candidates = pick_candidates_for_logical("latitude")
    if lat_candidates:
        coalesced = ", ".join(col_expr(c) for c in lat_candidates)
        select_parts.append(f"COALESCE({coalesced}) AS latitude")
    else:
        select_parts.append("NULL AS latitude")

    lon_candidates = pick_candidates_for_logical("longitude")
    if lon_candidates:
        coalesced = ", ".join(col_expr(c) for c in lon_candidates)
        select_parts.append(f"COALESCE({coalesced}) AS longitude")
    else:
        select_parts.append("NULL AS longitude")

    rt_candidates = pick_candidates_for_logical("room_type")
    if rt_candidates:
        select_parts.append(f"{col_expr(rt_candidates[0])} AS room_type")
    else:
        select_parts.append("NULL AS room_type")

    pt_candidates = pick_candidates_for_logical("property_type")
    if pt_candidates:
        select_parts.append(f"{col_expr(pt_candidates[0])} AS property_type")
    else:
        select_parts.append("NULL AS property_type")

    t_candidates = pick_candidates_for_logical("title")
    if t_candidates:
        select_parts.append(f"{col_expr(t_candidates[0])} AS title")
    else:
        select_parts.append("NULL AS title")

    select_clause = ",\n       ".join(select_parts)
    sql = f"SELECT {select_clause}\nFROM {_quote_identifier(TABLE_SCHEMA)}.{_quote_identifier(TABLE_NAME)} l\n"
    return sql, picked_id


@st.cache_data(ttl=600)
def load_from_db_min() -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Query the DB for the minimal columns (only those present) and return (normalized_df, id_column_name_or_None).
    """
    if not (HAS_SQLALCHEMY and DB_URL):
        return pd.DataFrame(columns=MINIMAL_COLUMNS), None
    try:
        engine = create_engine(DB_URL)
        info_sql = text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = :schema AND table_name = :table"
        )
        with engine.connect() as conn:
            res = conn.execute(info_sql, {"schema": TABLE_SCHEMA, "table": TABLE_NAME})
            available = [row[0] for row in res.fetchall()]

        if not available:
            st.warning("No columns found for table — check LISTINGS_SCHEMA / LISTINGS_TABLE env vars.")
            return pd.DataFrame(columns=MINIMAL_COLUMNS), None

        sql, id_col = _build_min_select_sql(available)
        df = pd.read_sql(sql, engine)
        return _normalize_min_columns(df), id_col
    except Exception as exc:
        st.warning(f"Failed to read from DB: {exc}")
        return pd.DataFrame(columns=MINIMAL_COLUMNS), None


def fetch_full_row_from_db_by_id(id_value: Any, id_col: str) -> Optional[Dict[str, Any]]:
    """
    Fetch SELECT * FROM schema.table WHERE {id_col} = :id LIMIT 1
    Returns dict or None on failure.
    """
    if not (HAS_SQLALCHEMY and DB_URL and id_col):
        return None
    try:
        engine = create_engine(DB_URL)
        q = text(f"SELECT * FROM {_quote_identifier(TABLE_SCHEMA)}.{_quote_identifier(TABLE_NAME)} WHERE {_quote_identifier(id_col)} = :id LIMIT 1")
        with engine.connect() as conn:
            result = conn.execute(q, {"id": id_value}).fetchone()
            if result is None:
                return None
            return dict(result)
    except Exception as exc:
        st.warning(f"Failed to fetch full row from DB for id {id_value}: {exc}")
        return None


# ---------------- Load minimal data (file projected or DB) ----------------
# First attempt to read only minimal columns from the file (cheaper than reading full dataset)
projected = load_minimal_from_file()
if projected is not None:
    # We keep `full_df` for the click-to-download only if user explicitly wants it later.
    # For initial load we only keep minimal projected frame.
    df_all = _normalize_min_columns(projected)
    full_df = None  # not loaded to avoid heavy memory usage
    data_source = "file_minimal"
    db_id_colname = None
else:
    df_all, db_id_colname = load_from_db_min()
    full_df = None
    data_source = "db" if not df_all.empty else "none"

if df_all.empty:
    st.warning("No listings found from DATA_FILE or DB. Please set DATA_FILE or ensure DB is reachable.")
    st.stop()


# ---------------- UI Controls ----------------
with st.sidebar:
    st.header("Map controls")
    view_mode = st.radio("View", ("Filtered", "All"))  # default to Filtered first to avoid heavy visual load
    center_choice = st.radio("Center map on:", ("Auto (data mean)", "Paris center"))
    st.markdown("---")
    st.markdown("**Select listing by:**")
    select_by = st.radio("Pick listing via:", ("Map click (recommended)", "Select from list"))

# Dependent filter UI: default to example small subset
# Build list of available room types and property types (full names, not collapsed)
all_room_types = sorted([r for r in pd.unique(df_all["room_type"]) if r is not None])
all_prop_types = sorted([p for p in pd.unique(df_all["property_type"]) if p is not None])

# Choose sensible defaults — prefer the example "Hotel room" + "Room in hotel"
default_room = "Hotel room" if "Hotel room" in all_room_types else (all_room_types[0] if all_room_types else "All")
room_type_options = ["All"] + all_room_types
# default index chooses the default_room in the options
default_index = room_type_options.index(default_room) if default_room in room_type_options else 0
chosen_room_type = st.sidebar.selectbox("Room type (filter)", room_type_options, index=default_index)

# Property-type options depend on chosen_room_type
if chosen_room_type == "All":
    property_candidates = all_prop_types
else:
    property_candidates = sorted([p for p in pd.unique(df_all.loc[df_all["room_type"] == chosen_room_type, "property_type"]) if p is not None])

# Choose default selected property types — prefer "Room in hotel" if present
default_prop = "Room in hotel"
default_selection = [default_prop] if default_prop in property_candidates else property_candidates.copy()
selected_property_types = st.sidebar.multiselect("Property type (filter)", property_candidates, default=default_selection)

# Apply filters to build df
df = df_all.copy()
if view_mode == "Filtered":
    if chosen_room_type != "All":
        df = df[df["room_type"] == chosen_room_type]
    if selected_property_types:
        df = df[df["property_type"].isin(selected_property_types)]

st.sidebar.markdown(f"**Listings visible:** {len(df):,} (after filters)")
st.subheader(f"Map — showing {len(df):,} listings")

# ---------------- Map center ----------------
if center_choice == "Paris center" or df.empty:
    center_lat, center_lon = 48.8566, 2.3522
else:
    center_lat = float(df["latitude"].mean())
    center_lon = float(df["longitude"].mean())

# ---------------- Map rendering + interaction ----------------
last_clicked_listing_id: Optional[Any] = None
clicked_coords: Optional[Tuple[float, float]] = None

if HAS_FOLIUM:
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, control_scale=True)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in df.iterrows():
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        listing_id = row["id"]
        room = row["room_type"] or "Unknown"
        prop = row["property_type"] or "Unknown"
        title = row["title"] or ""
        popup_html = folium.IFrame(
            html=(
                f"<b>{title}</b><br>"
                f"<b>ID</b>: {listing_id}<br>"
                f"<b>Room type</b>: {room}<br>"
                f"<b>Property type</b>: {prop}<br>"
                f"(click marker or use sidebar 'Select from list')"
            ),
            width=320,
            height=140,
        )
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html),
            tooltip=str(title) if title else f"ID: {listing_id}",
        ).add_to(marker_cluster)

    if st_folium is not None and select_by == "Map click (recommended)":
        map_data = st_folium(m, width=1100, height=700)
        last_click = None
        if isinstance(map_data, dict):
            last_click = map_data.get("last_clicked") or map_data.get("last_object_clicked") or None
        if last_click:
            lat = last_click.get("lat") or last_click.get("latitude")
            lng = last_click.get("lng") or last_click.get("longitude")
            if lat is not None and lng is not None:
                clicked_coords = (float(lat), float(lng))
    else:
        folium_static(m, width=1100, height=700)
        clicked_coords = None

elif HAS_PYDECK:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        pickable=True,
        get_position=["longitude", "latitude"],
        get_radius=25,
        auto_highlight=True,
    )
    tooltip = {
        "html": "<b>{title}</b><br>ID: {id}<br>Room: {room_type}<br>Property: {property_type}",
        "style": {"backgroundColor": "steelblue", "color": "white"},
    }
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(r)
    clicked_coords = None
else:
    st.info("For clickable markers install `folium` + `streamlit-folium`. Falling back to Streamlit map.")
    map_df = df[["latitude", "longitude"]].rename(columns={"latitude": "lat", "longitude": "lon"})
    st.map(map_df.assign(lat=map_df["lat"], lon=map_df["lon"]))
    clicked_coords = None

# ---------------- Selection from list fallback ----------------
selected_listing_choice = None
if select_by == "Select from list":
    options = [f"{rid} — { (str(title)[:60] + '...') if title and len(str(title))>60 else (title or '') }" for rid, title in zip(df["id"], df["title"])]
    options_map = dict(zip(options, df["id"]))
    if options:
        sel = st.sidebar.selectbox("Select listing", options)
        selected_listing_choice = options_map.get(sel)
    else:
        st.sidebar.info("No listings available to select.")

# If clicked on map, find nearest listing id
def _find_listing_id_by_coords(df_frame: pd.DataFrame, coords: Tuple[float, float], tol_meters: float = 5.0) -> Optional[Any]:
    lat_click, lon_click = coords
    if df_frame.empty:
        return None
    df_frame = df_frame.copy()
    df_frame["_dist2"] = (df_frame["latitude"] - lat_click) ** 2 + (df_frame["longitude"] - lon_click) ** 2
    best = df_frame.nsmallest(1, "_dist2").iloc[0]
    return best["id"]

if clicked_coords:
    clicked_id = _find_listing_id_by_coords(df, clicked_coords)
    last_clicked_listing_id = clicked_id

if selected_listing_choice:
    last_clicked_listing_id = selected_listing_choice

# ---------------- Show full details for selected listing (click or select) ----------------
if last_clicked_listing_id is not None:
    st.markdown("---")
    st.subheader("Listing details")

    # Try to load full row from file (only if user provided a full DATA_FILE and we didn't avoid loading it)
    full_row = None
    if full_df is not None:
        matches = full_df[full_df.apply(lambda r: str(r.get("id", r.get("listing_id", ""))) == str(last_clicked_listing_id), axis=1)]
        if not matches.empty:
            full_row = matches.iloc[0].to_dict()

    # otherwise fetch from DB if possible
    if full_row is None and data_source.startswith("db") and db_id_colname:
        fetched = fetch_full_row_from_db_by_id(last_clicked_listing_id, db_id_colname)
        if fetched:
            full_row = fetched

    # fallback to minimal
    if full_row is None:
        minimal = df[df["id"].astype(str) == str(last_clicked_listing_id)]
        if not minimal.empty:
            full_row = minimal.iloc[0].to_dict()

    if full_row:
        rows_display = []
        for k in sorted(full_row.keys()):
            v = full_row[k]
            if isinstance(v, float):
                if math.isfinite(v):
                    disp = f"{v:.6f}" if ("lat" in k.lower() or "lon" in k.lower()) else f"{v}"
                else:
                    disp = ""
            else:
                disp = str(v)
            rows_display.append((k, disp))
        left = [k for k, _ in rows_display]
        right = [v for _, v in rows_display]
        details_df = pd.DataFrame({"field": left, "value": right})
        st.dataframe(details_df, use_container_width=True)
        st.download_button("Download listing JSON", data=pd.Series(full_row).to_json(orient="index", force_ascii=False), file_name=f"listing_{last_clicked_listing_id}.json", mime="application/json")
    else:
        st.info("Unable to retrieve full details for that listing.")

# ---------------- Data table and download for visible listings ----------------
with st.expander("Show visible listings table and download"):
    st.dataframe(df.reset_index(drop=True)[["id", "title", "room_type", "property_type", "latitude", "longitude"]], use_container_width=True)
    csv = df.to_csv(index=False)
    st.download_button("Download visible listings (CSV)", data=csv, file_name="paris_listings_filtered.csv", mime="text/csv")

st.caption("Click a marker (Folium) or select from the list to view full listing details. Use the 'Room type' filter to restrict property types shown.")
