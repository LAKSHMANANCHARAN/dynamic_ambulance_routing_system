import streamlit as st # type: ignore
from streamlit_folium import folium_static # type: ignore
import folium # type: ignore
import osmnx as ox # type: ignore
import networkx as nx  # type: ignore
import random
import time
import pandas as pd
import numpy as np
import os
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(layout="wide", page_title="🚑 Ambulance Routing Dashboard", page_icon="🚑")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid rgba(255,71,87,0.35);
        padding: 1.6rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(255,71,87,0.18);
    }
    .main-header h1 { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.7; font-size: 0.95rem; }

    .section-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #ff6b81;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin: 1.4rem 0 0.6rem;
        padding-bottom: 5px;
        border-bottom: 1px solid rgba(255,107,129,0.2);
    }

    .fleet-card {
        background: linear-gradient(135deg, #1e1e2e, #252535);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.5rem;
    }
    .fleet-card:hover { border-color: rgba(255,71,87,0.4); }
    .fleet-card .fc-id   { font-size: 1.05rem; font-weight: 700; color: #fff; }
    .fleet-card .fc-enroute  { color: #ffa502; font-weight: 600; font-size: 0.85rem; }
    .fleet-card .fc-arrived  { color: #2ed573; font-weight: 600; font-size: 0.85rem; }
    .fleet-card .fc-idle     { color: #747d8c; font-weight: 600; font-size: 0.85rem; }
    .fleet-card .fc-pill {
        background: rgba(255,71,87,0.12);
        border: 1px solid rgba(255,71,87,0.25);
        border-radius: 20px;
        padding: 2px 9px;
        font-size: 0.75rem;
        color: #ff6b81;
        font-weight: 600;
    }
    .fc-bar-bg  { width:100%; background:rgba(255,255,255,0.07); border-radius:4px; height:4px; margin-top:6px; }
    .fc-bar-fg  { height:4px; border-radius:4px; background:linear-gradient(90deg,#ff4757,#ff6b81); }

    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #252535);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        height: 100%;
    }
    .metric-card .mc-label { font-size: 0.7rem; color: #a4b0be; text-transform: uppercase; letter-spacing: 0.6px; }
    .metric-card .mc-value { font-size: 1.55rem; font-weight: 700; color: #fff; margin-top: 4px; }
    .metric-card .mc-sub   { font-size: 0.75rem; color: #636e72; margin-top: 3px; }

    .event-card {
        background: rgba(255,71,87,0.06);
        border-left: 3px solid #ff4757;
        border-radius: 0 8px 8px 0;
        padding: 0.65rem 1rem;
        margin-bottom: 0.45rem;
        font-size: 0.85rem;
        color: #dfe6e9;
    }
    .event-card.pickup { border-left-color: #2ed573; background: rgba(46,213,115,0.06); }

    .legend-row { display:flex; flex-wrap:wrap; gap:14px 24px; margin-top:4px; }
    .legend-item { display:flex; align-items:center; gap:7px; font-size:0.83rem; color:#b2bec3; }
    .legend-dot  { width:11px; height:11px; border-radius:50%; flex-shrink:0; }
    .legend-line { width:22px; height:3px; border-radius:2px; flex-shrink:0; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg,#1a1a2e 0%,#16213e 100%);
        border-right: 1px solid rgba(255,71,87,0.18);
    }
    .stProgress > div > div { background: linear-gradient(90deg,#ff4757,#ff6b81) !important; border-radius:4px; }
    .map-wrap { border:1px solid rgba(255,71,87,0.22); border-radius:12px; overflow:hidden;
                box-shadow:0 4px 24px rgba(0,0,0,0.35); margin-top:0.4rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>🚑 Smart Ambulance Routing Dashboard</h1>
  <p>Real-Time Emergency Navigation &mdash; Anna Nagar, Chennai</p>
</div>
""", unsafe_allow_html=True)

# ---------------- CONGESTION MODEL (replaces TensorFlow/Keras LSTM) ----------------
# Uses scikit-learn RandomForestRegressor — fully compatible with Python 3.14.
# Mimics the same interface: predict() accepts (n_samples, 24) feature arrays
# and returns congestion scores in [0.0, 1.0].

@st.cache_resource
def load_congestion_model():
    """
    Load a pre-trained sklearn model from disk (congestion_model.pkl) if available.
    If not, build and return a lightweight in-memory model trained on synthetic data.
    This replaces the TensorFlow LSTM while preserving the same congestion-factor logic.
    """
    model_path = "congestion_model.pkl"
    try:
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            return model
    except Exception:
        pass

    # Build a lightweight fallback model on synthetic traffic data
    try:
        from sklearn.ensemble import RandomForestRegressor  # type: ignore

        rng = np.random.default_rng(42)
        # 24 time-step features per sample (hour-of-day traffic pattern)
        X_train = rng.random((2000, 24)).astype("float32")
        # Ground-truth: busy hours (8-10, 17-20) → higher congestion
        hour_weights = np.array([
            0.2, 0.2, 0.15, 0.15, 0.2, 0.3,   # 0-5
            0.45, 0.65, 0.85, 0.8, 0.55, 0.5,  # 6-11
            0.55, 0.5, 0.45, 0.5, 0.7, 0.9,    # 12-17
            0.85, 0.75, 0.6, 0.45, 0.35, 0.25  # 18-23
        ], dtype="float32")
        y_train = np.clip(X_train.dot(hour_weights) / hour_weights.sum() + rng.normal(0, 0.05, 2000), 0, 1).astype("float32")

        model = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model
    except Exception as e:
        st.warning(f"Could not build congestion model: {e}. Using random fallback.")
        return None


def predict_congestion(model, num_edges: int) -> np.ndarray:
    """
    Run congestion inference for `num_edges` road segments.
    Input shape: (num_edges, 24) — 24 time-step traffic features per edge.
    Returns a 1-D array of floats in [0.0, 1.0].
    """
    if model is None:
        return np.random.choice([0.3, 0.5, 0.8], size=num_edges).astype("float32")
    try:
        batch_data = np.random.rand(num_edges, 24).astype("float32") * 0.8
        predictions = model.predict(batch_data).flatten()
        return np.clip(predictions, 0.0, 1.0).astype("float32")
    except Exception:
        return np.random.choice([0.3, 0.5, 0.8], size=num_edges).astype("float32")


# ML: Converts model output (0.0 to 1.0) into a congestion multiplier — low / medium / high
def get_congestion_factor(prediction: float) -> float:
    if prediction < 0.4:
        return 1.0
    elif prediction < 0.7:
        return 1.5
    else:
        return 2.0


congestion_model = load_congestion_model()

# ---------------- GRAPH LOADING ----------------
@st.cache_resource
def load_graph():
    try:
        return ox.graph_from_place("Anna Nagar, Chennai, India", network_type="drive").to_undirected()
    except Exception as e:
        st.error(f"Failed to load map data: {e}")
        return None

def get_min_edge_length(graph, u, v):
    edge_data = graph.get_edge_data(u, v, default={}) or {}
    if not edge_data:
        return 0.0
    if isinstance(edge_data, dict) and "length" in edge_data:
        try:
            return float(edge_data["length"])
        except (TypeError, ValueError):
            return 0.0
    lengths = []
    if isinstance(edge_data, dict):
        for attrs in edge_data.values():
            if isinstance(attrs, dict) and "length" in attrs:
                try:
                    lengths.append(float(attrs["length"]))
                except (TypeError, ValueError):
                    continue
    return min(lengths) if lengths else 0.0

def build_alternate_routes(graph, primary_route, start_node, destination, max_alternates=3):
    """Build alternate routes by temporarily removing edges from the primary route."""
    alternates = []
    for i in range(1, min(len(primary_route) - 1, 6)):
        if graph.has_edge(primary_route[i], primary_route[i + 1]):
            edge_data = graph.get_edge_data(primary_route[i], primary_route[i + 1])
            graph.remove_edge(primary_route[i], primary_route[i + 1])
            try:
                alt_route = nx.shortest_path(graph, start_node, destination, weight="dynamic_weight")
                if alt_route not in alternates and alt_route != primary_route:
                    alternates.append(alt_route)
            except nx.NetworkXNoPath:
                pass
            graph.add_edge(primary_route[i], primary_route[i + 1], **edge_data)
            if len(alternates) >= max_alternates:
                break
    return alternates

# ---------------- ROUTE COMPUTATION ----------------
def calculate_routes(seed, _congestion_model=None):
    """Load graph, apply congestion model predictions, fetch hospitals."""
    G = load_graph()
    if G is None:
        return None

    num_edges = G.number_of_edges()

    # ML: Run congestion inference — predictions array shape (num_edges,) in [0, 1]
    predictions = predict_congestion(_congestion_model, num_edges)

    # ML: Apply predictions to graph edges — each edge gets a congestion-adjusted weight
    for idx, (u, v, data) in enumerate(G.edges(data=True)):
        if 'length' in data:
            prediction = float(predictions[idx])
            congestion_factor = get_congestion_factor(prediction)  # ML: map prediction -> multiplier
            data['dynamic_weight'] = data['length'] * congestion_factor  # ML: congestion-aware edge cost
            data['congestion_factor'] = congestion_factor
        else:
            data['dynamic_weight'] = 100
            data['congestion_factor'] = 1.0

    hospital_nodes = []
    hospital_coords = []
    try:
        tags = {"amenity": "hospital"}
        hospital_gdf = ox.features_from_place("Anna Nagar, Chennai, India", tags)

        for geom in hospital_gdf.geometry:
            try:
                lon, lat = geom.centroid.x, geom.centroid.y
                nearest_node = ox.distance.nearest_nodes(G, lon, lat)
                if nearest_node not in hospital_nodes:
                    hospital_nodes.append(nearest_node)
                    hospital_coords.append((lat, lon))
            except (AttributeError, ValueError):
                continue
    except Exception as e:
        st.warning(f"Could not fetch hospitals: {e}")

    if not hospital_nodes:
        nodes = list(G.nodes)
        hospital_nodes = random.sample(nodes, min(5, len(nodes)))
        hospital_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in hospital_nodes]

    return {
        "G": G,
        "hospital_nodes": hospital_nodes,
        "hospital_coords": hospital_coords
    }

def compute_route_for_ambulance(scenario_data, ambulance_node):
    """Select nearest hospital and compute route from ambulance's current node."""
    if scenario_data is None:
        return None

    G = scenario_data["G"]
    hospital_nodes = scenario_data["hospital_nodes"]

    # DP: Dijkstra's algorithm — finds shortest path to each hospital using ML-weighted edges
    distances = {}
    for h in hospital_nodes:
        try:
            distances[h] = nx.shortest_path_length(G, ambulance_node, h, weight="dynamic_weight")
        except nx.NetworkXNoPath:
            continue

    if not distances:
        return None

    destination = min(distances, key=distances.get)  # DP: greedy selection of nearest hospital

    try:
        optimal_route = nx.shortest_path(G, ambulance_node, destination, weight="dynamic_weight")
    except nx.NetworkXNoPath:
        return None

    lats = [G.nodes[n]['y'] for n in optimal_route]
    lons = [G.nodes[n]['x'] for n in optimal_route]
    center = [(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2]

    # DP: Alternate route generation — temporarily remove edges and re-run Dijkstra to find detours
    alternate_routes = []
    for i in range(1, min(len(optimal_route) - 1, 6)):
        if G.has_edge(optimal_route[i], optimal_route[i+1]):
            edge_data = G.get_edge_data(optimal_route[i], optimal_route[i+1])
            G.remove_edge(optimal_route[i], optimal_route[i+1])

            try:
                alt_route = nx.shortest_path(G, ambulance_node, destination, weight="dynamic_weight")
                if alt_route not in alternate_routes and len(alt_route) != len(optimal_route):
                    alternate_routes.append(alt_route)
            except nx.NetworkXNoPath:
                pass

            G.add_edge(optimal_route[i], optimal_route[i+1], **edge_data)

            if len(alternate_routes) >= 3:
                break

    return {
        "G": G,
        "start": ambulance_node,
        "destination": destination,
        "hospital_nodes": scenario_data["hospital_nodes"],
        "hospital_coords": scenario_data["hospital_coords"],
        "distances": distances,
        "optimal_route": optimal_route,
        "alternate_routes": alternate_routes,
        "center": center
    }

def sync_ambulance_state(amb):
    """Single source of truth for node/step/status transitions."""
    if amb["routes_data"] is None:
        amb["status"] = "Idle"
        amb["auto_drive"] = False
        return

    route = amb["routes_data"].get("optimal_route", [])
    if not route:
        amb["status"] = "Idle"
        amb["auto_drive"] = False
        amb["routes_data"] = None
        return

    amb["step"] = min(max(int(amb["step"]), 0), len(route) - 1)
    amb["node"] = route[amb["step"]]
    amb["routes_data"]["start"] = amb["node"]

    if amb["step"] >= len(route) - 1:
        amb["step"] = len(route) - 1
        amb["node"] = route[-1]
        amb["routes_data"]["start"] = amb["node"]
        amb["status"] = "Arrived"
        amb["auto_drive"] = False
    else:
        amb["status"] = "En Route"

def assign_emergency_to_ambulance(ambulance_id, scenario_data, pickup_node):
    """Assign emergency: Phase 1 = ambulance → pickup."""
    amb = st.session_state.fleet[ambulance_id]
    ambulance_node = amb["node"]
    G = scenario_data["G"]

    try:
        route_to_pickup = nx.shortest_path(G, ambulance_node, pickup_node, weight="dynamic_weight")
    except nx.NetworkXNoPath:
        return False

    amb["routes_data"] = {
        "G": G,
        "start": ambulance_node,
        "destination": pickup_node,
        "hospital_nodes": scenario_data["hospital_nodes"],
        "hospital_coords": scenario_data["hospital_coords"],
        "optimal_route": route_to_pickup,
        "alternate_routes": [],
        "center": [(G.nodes[ambulance_node]['y'] + G.nodes[pickup_node]['y'])/2,
                   (G.nodes[ambulance_node]['x'] + G.nodes[pickup_node]['x'])/2]
    }
    amb["step"] = 0
    amb["auto_drive"] = False
    amb["status"] = "En Route"
    amb["phase"] = "ToPickup"
    amb["pickup_node"] = pickup_node
    amb["destination_hospital"] = None
    amb["reroute_count"] = 0
    amb["event_log"] = []
    amb["slider_override"] = True
    return True

# ---------------- FLEET INITIALIZATION ----------------
if "fleet" not in st.session_state:
    G = load_graph()
    if G:
        nodes = list(G.nodes)
        st.session_state.fleet = {}
        for i in range(1, 6):
            st.session_state.fleet[f"A{i}"] = {
                "routes_data": None,
                "node": random.choice(nodes),
                "step": 0,
                "auto_drive": False,
                "status": "Idle",
                "reroute_count": 0,
                "event_log": [],
                "slider_override": False,
                "phase": None,
                "pickup_node": None,
                "destination_hospital": None
            }
    else:
        st.session_state.fleet = {}

if "last_move_ts" not in st.session_state:
    st.session_state.last_move_ts = 0.0

if "current_emergency" not in st.session_state:
    st.session_state.current_emergency = None

# ---------------- SIDEBAR MODE SELECTION ----------------
with st.sidebar:
    st.header("🎛️ Control Panel")
    mode = st.radio("Select View Mode", ["👨‍💼 Admin Dashboard", "🚑 Ambulance Panel"])

    if mode == "🚑 Ambulance Panel":
        selected_ambulance = st.selectbox("Select Ambulance", list(st.session_state.fleet.keys()))

    if st.button("🔄 Generate New Emergency Scenario", type="primary", use_container_width=True):
        scenario_seed = int(time.time())
        scenario_data = calculate_routes(scenario_seed, congestion_model)
        if scenario_data:
            G = scenario_data["G"]
            all_nodes = list(G.nodes)
            emergency_node = random.choice(all_nodes)
            st.session_state.current_emergency = emergency_node

            if mode == "🚑 Ambulance Panel":
                if assign_emergency_to_ambulance(selected_ambulance, scenario_data, emergency_node):
                    st.success(f"✅ {selected_ambulance} dispatched to pickup location")
                else:
                    st.error(f"❌ No route available for {selected_ambulance}")
            else:
                idle_ambulances = [aid for aid, amb in st.session_state.fleet.items() if amb["status"] == "Idle"]
                if idle_ambulances:
                    min_dist = float("inf")
                    nearest_amb = None

                    for aid in idle_ambulances:
                        amb_node = st.session_state.fleet[aid]["node"]
                        try:
                            dist = nx.shortest_path_length(G, amb_node, emergency_node, weight="dynamic_weight")
                            if dist < min_dist:
                                min_dist = dist
                                nearest_amb = aid
                        except Exception:
                            continue

                    if nearest_amb:
                        if assign_emergency_to_ambulance(nearest_amb, scenario_data, emergency_node):
                            st.success(f"✅ Nearest ambulance {nearest_amb} dispatched to pickup location")
                        else:
                            st.error(f"❌ No route available for {nearest_amb}")
                else:
                    st.warning("⚠️ All ambulances are busy!")
        st.rerun()

# ============================================================
# GLOBAL AUTO-MOVEMENT ENGINE
# ============================================================
for amb_id, amb in st.session_state.fleet.items():
    sync_ambulance_state(amb)
    if amb["status"] == "Arrived":
        amb["auto_drive"] = False

MOVE_INTERVAL_SEC = 1.2

any_auto_drive = any(
    amb["auto_drive"] and amb["routes_data"] and amb["status"] != "Arrived"
    for amb in st.session_state.fleet.values()
)

# ============================================================
# ADMIN DASHBOARD MODE
# ============================================================
if mode == "👨‍💼 Admin Dashboard":
    st.subheader("👨‍💼 Fleet Overview")

    fleet_data = []
    for amb_id, amb in st.session_state.fleet.items():
        if amb["routes_data"] and amb["status"] == "En Route":
            route_len = len(amb["routes_data"]["optimal_route"])
            progress = int((amb["step"] / max(route_len - 1, 1)) * 100)
        else:
            progress = 0

        fleet_data.append({
            "🚑 ID": amb_id,
            "📊 Status": amb["status"],
            "📈 Progress": f"{progress}%",
            "🔄 Reroutes": amb["reroute_count"]
        })

    st.dataframe(pd.DataFrame(fleet_data), use_container_width=True, hide_index=True)

    G = load_graph()
    if G:
        all_lats, all_lons = [], []
        for amb in st.session_state.fleet.values():
            if amb["routes_data"]:
                route = amb["routes_data"]["optimal_route"]
                all_lats.extend([G.nodes[n]['y'] for n in route])
                all_lons.extend([G.nodes[n]['x'] for n in route])

        if all_lats:
            center = [(min(all_lats) + max(all_lats)) / 2, (min(all_lons) + max(all_lons)) / 2]
        else:
            center = [13.0850, 80.2101]

        m = folium.Map(location=center, zoom_start=16, control_scale=True)

        if st.session_state.current_emergency:
            e_lat = G.nodes[st.session_state.current_emergency]['y']
            e_lon = G.nodes[st.session_state.current_emergency]['x']
            folium.Marker(
                location=(e_lat, e_lon),
                popup="🚨 Emergency Location",
                icon=folium.DivIcon(
                    html='<div style="font-size: 36px; animation: blink 1s infinite;">🚨</div>'
                         '<style>@keyframes blink {0%, 100% {opacity: 1;} 50% {opacity: 0.3;}}</style>'
                )
            ).add_to(m)

        for amb_id, amb in st.session_state.fleet.items():
            if amb["routes_data"] and amb["status"] in ["En Route", "Arrived"]:
                data = amb["routes_data"]
                route = data["optimal_route"]

                if "old_route" in data:
                    old_pts = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in data["old_route"]]
                    folium.PolyLine(old_pts, color="gray", weight=2, opacity=0.4).add_to(m)

                route_color = "orange" if amb["phase"] == "ToPickup" else "red"
                pts = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route]
                phase_label = "→ Pickup" if amb["phase"] == "ToPickup" else "→ Hospital"
                folium.PolyLine(pts, color=route_color, weight=4, opacity=0.7,
                                tooltip=f"{amb_id} {phase_label}").add_to(m)

                current_node = route[amb["step"]]
                lat, lon = G.nodes[current_node]['y'], G.nodes[current_node]['x']
                icon_html = "🏁" if amb["step"] == len(route) - 1 else "🚑"

                folium.Marker(
                    location=(lat, lon),
                    popup=f"{amb_id} - Step {amb['step']+1}/{len(route)}",
                    icon=folium.DivIcon(html=f'<div style="font-size: 32px;">{icon_html}</div>')
                ).add_to(m)
            elif amb["status"] == "Idle" and amb["node"]:
                lat, lon = G.nodes[amb["node"]]['y'], G.nodes[amb["node"]]['x']
                folium.Marker(
                    location=(lat, lon),
                    popup=f"{amb_id} - Idle",
                    icon=folium.DivIcon(html='<div style="font-size: 30px;">🚑</div>')
                ).add_to(m)

        folium_static(m, height=600, width=None)

# ============================================================
# AMBULANCE PANEL MODE
# ============================================================
else:
    amb = st.session_state.fleet[selected_ambulance]

    if amb["routes_data"] is None:
        st.info(f"🚑 {selected_ambulance} is currently Idle. Generate an emergency scenario to assign a route.")
        st.stop()

    data = amb["routes_data"]
    G = data["G"]
    max_idx = max(len(amb["routes_data"]["optimal_route"]) - 1, 0)

    amb["step"] = min(max(amb["step"], 0), max_idx)
    route_steps = max(len(amb["routes_data"]["optimal_route"]) - 1, 1)
    progress = int((amb["step"] / route_steps) * 100)

    # ---------------- ANALYTICS PANEL ----------------
    st.subheader(f"📊 {selected_ambulance} - Emergency Response Analytics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Progress", f"{progress}%")
    with col2:
        st.metric("🚑 Status", amb["status"])
    with col3:
        remaining = len(amb["routes_data"]["optimal_route"]) - amb["step"] - 1
        st.metric("⏱️ Steps Left", remaining if remaining > 0 else "0")
    with col4:
        # ML: Average congestion factor across all route edges — derived from model predictions
        if len(amb["routes_data"]["optimal_route"]) > 1:
            route_opt = amb["routes_data"]["optimal_route"]
            avg_congestion = np.mean([
                G[route_opt[i]][route_opt[i+1]][0].get('congestion_factor', 1.0)
                for i in range(len(route_opt) - 1)
            ])
        else:
            avg_congestion = 1.0
        congestion_status = "🟢 Low" if avg_congestion < 1.3 else "🟡 Medium" if avg_congestion < 1.8 else "🔴 High"
        st.metric("🚦 Congestion", congestion_status)

    # ---------------- TABS ----------------
    tab1, tab2 = st.tabs(["🏥 Hospital Analysis", "🗺️ Route Comparison"])

    with tab1:
        if "distances" in data and data["distances"]:
            hospital_df = pd.DataFrame([
                {"🏆 Rank": i+1, "🏥 Hospital ID": h, "📏 Distance (km)": round(d/1000, 2),
                 "📍 Status": "🎯 Selected" if h == data["destination"] else "⚪ Available"}
                for i, (h, d) in enumerate(sorted(data["distances"].items(), key=lambda x: x[1]))
            ])
            st.dataframe(hospital_df, use_container_width=True, hide_index=True)
        else:
            st.info("Hospital analysis will be available after patient pickup.")

    with tab2:
        route_data = []
        current_start = amb["routes_data"]["optimal_route"][0] if "old_route" in data else data["start"]

        try:
            optimal_dist = nx.shortest_path_length(G, current_start, data["destination"], weight="dynamic_weight")
        except nx.NetworkXNoPath:
            optimal_dist = sum(
                G[amb["routes_data"]["optimal_route"][j]][amb["routes_data"]["optimal_route"][j+1]][0].get('dynamic_weight', 0)
                for j in range(len(amb["routes_data"]["optimal_route"]) - 1)
            )

        route_data.append({
            "🛣️ Route": "🎯 Optimal",
            "📏 Distance (km)": round(optimal_dist/1000, 2),
            "🔗 Nodes": len(amb["routes_data"]["optimal_route"]),
            "💡 Weight Type": "RF Congestion Model",
            "📊 Status": "🟢 Active"
        })

        for i, route in enumerate(data["alternate_routes"]):
            try:
                alt_dist = sum(
                    G[route[j]][route[j+1]][0].get('dynamic_weight', get_min_edge_length(G, route[j], route[j+1]))
                    for j in range(len(route)-1)
                )
                route_data.append({
                    "🛣️ Route": f"🔄 Alternate {i+1}",
                    "📏 Distance (km)": round(alt_dist/1000, 2),
                    "🔗 Nodes": len(route),
                    "💡 Weight Type": "RF Congestion Model",
                    "📊 Status": "🟡 Backup"
                })
            except (KeyError, IndexError, TypeError):
                continue

        st.dataframe(pd.DataFrame(route_data), use_container_width=True, hide_index=True)

    # ---------------- MOVEMENT CONTROL ----------------
    st.subheader("🚑 Ambulance Movement Control")

    if st.button("⚠️ Simulate Accident", type="secondary", use_container_width=True):
        if amb["step"] > 0 and amb["step"] < len(amb["routes_data"]["optimal_route"]) - 1:
            current_node = amb["routes_data"]["optimal_route"][amb["step"]]
            old_route = data["optimal_route"].copy()

            if len(old_route) > 1:
                old_congestion = np.mean([G[old_route[i]][old_route[i+1]][0].get('congestion_factor', 1.0)
                                          for i in range(len(old_route)-1)])
                old_time = sum(G[old_route[i]][old_route[i+1]][0].get('dynamic_weight', 0)
                               for i in range(amb["step"], len(old_route)-1))
            else:
                old_congestion = 1.0
                old_time = 0

            for i in range(amb["step"], len(amb["routes_data"]["optimal_route"]) - 1):
                u, v = amb["routes_data"]["optimal_route"][i], amb["routes_data"]["optimal_route"][i+1]
                if G.has_edge(u, v):
                    edge_data = G[u][v][0]
                    # ML: Spike congestion to 3x to simulate accident — forces reroute
                    edge_data['dynamic_weight'] = edge_data.get('length', 100) * 3.0
                    edge_data['congestion_factor'] = 3.0

            try:
                new_route = nx.shortest_path(G, current_node, data["destination"], weight="dynamic_weight")

                if new_route[0] == current_node:
                    if len(new_route) > 1:
                        new_congestion = np.mean([G[new_route[i]][new_route[i+1]][0].get('congestion_factor', 1.0)
                                                  for i in range(len(new_route)-1)])
                        new_time = sum(G[new_route[i]][new_route[i+1]][0].get('dynamic_weight', 0)
                                       for i in range(len(new_route)-1))
                    else:
                        new_congestion = 1.0
                        new_time = 0
                    time_saved = old_time - new_time

                    data["old_route"] = old_route
                    data["optimal_route"] = new_route
                    data["alternate_routes"] = build_alternate_routes(
                        G, new_route, current_node, data["destination"], max_alternates=3
                    )
                    data["start"] = current_node
                    new_lats = [G.nodes[n]['y'] for n in new_route]
                    new_lons = [G.nodes[n]['x'] for n in new_route]
                    data["center"] = [(min(new_lats) + max(new_lats)) / 2, (min(new_lons) + max(new_lons)) / 2]
                    data["old_congestion"] = old_congestion
                    data["new_congestion"] = new_congestion
                    data["time_saved"] = time_saved

                    amb["step"] = 0
                    amb["auto_drive"] = False
                    sync_ambulance_state(amb)
                    amb["slider_override"] = True
                    amb["reroute_count"] += 1
                    amb["event_log"].append({
                        "event": "Accident Detected",
                        "node": current_node,
                        "old_congestion": old_congestion,
                        "new_congestion": new_congestion,
                        "time_saved": time_saved
                    })
                    st.success("✅ Route Adjusted Successfully")
                    st.rerun()
            except nx.NetworkXNoPath:
                st.error("No alternate path available")

    can_generate_hospital_path = (
        amb.get("phase") == "ToPickup"
        and amb["step"] == len(amb["routes_data"]["optimal_route"]) - 1
        and amb["node"] == amb.get("pickup_node")
    )
    if st.button("🏥 Generate Hospital Path from Pickup", use_container_width=True, disabled=not can_generate_hospital_path):
        G = data["G"]
        hospital_nodes = data["hospital_nodes"]
        pickup_node = amb["pickup_node"]

        distances = {}
        for h in hospital_nodes:
            try:
                distances[h] = nx.shortest_path_length(G, pickup_node, h, weight="dynamic_weight")
            except nx.NetworkXNoPath:
                continue

        if distances:
            nearest_hospital = min(distances, key=distances.get)
            try:
                route_to_hospital = nx.shortest_path(G, pickup_node, nearest_hospital, weight="dynamic_weight")

                alternate_routes = build_alternate_routes(
                    G, route_to_hospital, pickup_node, nearest_hospital, max_alternates=3
                )

                data["optimal_route"] = route_to_hospital
                data["start"] = pickup_node
                data["destination"] = nearest_hospital
                data["distances"] = distances
                data["alternate_routes"] = alternate_routes
                data["center"] = [(G.nodes[pickup_node]['y'] + G.nodes[nearest_hospital]['y'])/2,
                                  (G.nodes[pickup_node]['x'] + G.nodes[nearest_hospital]['x'])/2]
                amb["step"] = 0
                amb["phase"] = "ToHospital"
                amb["destination_hospital"] = nearest_hospital
                amb["event_log"].append({"event": "Patient Picked Up", "node": pickup_node})
                amb["auto_drive"] = False
                sync_ambulance_state(amb)
                st.success("✅ Hospital route generated. Press Start to begin auto movement.")
                st.rerun()
            except nx.NetworkXNoPath:
                st.error("No path to hospital from pickup node.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_disabled = amb["status"] == "Arrived" and amb["phase"] == "ToHospital"
        if st.button("⏮️ Start", use_container_width=True, disabled=start_disabled):
            amb["auto_drive"] = True
            st.rerun()

    with col2:
        if st.button("⬅️ Previous", disabled=amb["step"] == 0, use_container_width=True):
            amb["auto_drive"] = False
            amb["step"] -= 1
            sync_ambulance_state(amb)
            st.rerun()

    with col3:
        if st.button("➡️ Next", disabled=amb["step"] == len(amb["routes_data"]["optimal_route"]) - 1, use_container_width=True):
            amb["auto_drive"] = False
            amb["step"] += 1
            sync_ambulance_state(amb)
            st.rerun()

    with col4:
        end_disabled = amb["step"] == len(amb["routes_data"]["optimal_route"]) - 1
        if st.button("⏭️ End", disabled=end_disabled, use_container_width=True):
            amb["step"] = len(amb["routes_data"]["optimal_route"]) - 1
            sync_ambulance_state(amb)
            st.rerun()

    if "slider_override" not in amb:
        amb["slider_override"] = False

    _slider_override_active = amb["slider_override"]
    if amb["slider_override"]:
        amb["slider_override"] = False

    _route_len = len(amb["routes_data"]["optimal_route"])
    _slider_max = max(_route_len - 1, 1)
    _safe_step = max(0, min(amb["step"], _slider_max))
    _slider_key = (
        f"slider_{selected_ambulance}_arrived_{_safe_step}"
        if amb["status"] == "Arrived"
        else f"slider_{selected_ambulance}_{_route_len}"
    )
    slider_step = st.slider("🚑 Ambulance Position", 0, _slider_max, _safe_step, key=_slider_key)
    if (
        not amb["auto_drive"]
        and not _slider_override_active
        and amb["status"] != "Arrived"
        and slider_step != amb["step"]
    ):
        amb["step"] = slider_step
        sync_ambulance_state(amb)

    # ---------------- REROUTE LOG ----------------
    if amb["reroute_count"] > 0 or amb["event_log"]:
        st.subheader("📜 Reroute Event Log")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔄 Total Reroutes", amb["reroute_count"])
        with col2:
            if "old_congestion" in data and "new_congestion" in data:
                congestion_change = ((data["new_congestion"] - data["old_congestion"]) / data["old_congestion"]) * 100
                st.metric("🚦 Congestion Change", f"{congestion_change:+.1f}%")
        with col3:
            if "time_saved" in data:
                st.metric("⏱️ Time Impact", f"{abs(data['time_saved']):.0f}s")

        if amb["event_log"]:
            with st.expander("📝 View Event Details", expanded=True):
                for idx, event in enumerate(reversed(amb["event_log"])):
                    if event["event"] == "Patient Picked Up":
                        st.markdown(f"""
                        **Event #{len(amb['event_log']) - idx}:**
                        - 📍 {event['event']} at Node `{event['node']}`
                        ---
                        """)
                    else:
                        st.markdown(f"""
                        **Event #{len(amb['event_log']) - idx}:**
                        - ⚠️ {event['event']} at Node `{event['node']}`
                        - 🚦 Congestion: {event.get('old_congestion', 0):.2f} → {event.get('new_congestion', 0):.2f}
                        - ⏱️ Time: {abs(event.get('time_saved', 0)):.0f}s {'saved' if event.get('time_saved', 0) > 0 else 'added'}
                        ---
                        """)

    # ---------------- MAP ----------------
    route_coords = [
        [G.nodes[n]['y'], G.nodes[n]['x']]
        for n in amb["routes_data"]["optimal_route"]
    ]
    current_pos = route_coords[amb["step"]]

    m = folium.Map(location=data["center"], zoom_start=15, control_scale=True)

    route_colors = ["blue", "green", "orange", "purple"]
    for idx, alt_route in enumerate(data["alternate_routes"]):
        if idx < len(route_colors):
            pts = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in alt_route]
            folium.PolyLine(pts, color=route_colors[idx], weight=4, opacity=0.6,
                            tooltip=f"Alternate Route {idx+1}").add_to(m)

    if "old_route" in data:
        old_pts = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in data["old_route"]]
        folium.PolyLine(old_pts, color="gray", weight=4, opacity=0.5, tooltip="Previous Route").add_to(m)

    folium.PolyLine(route_coords, color="red", weight=5, opacity=0.85, tooltip="🚑 Optimal Route").add_to(m)

    start_node = data["start"]
    folium.Marker(
        location=(G.nodes[start_node]['y'], G.nodes[start_node]['x']),
        popup="🏁 Start", icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    if amb["phase"] == "ToPickup" and amb["pickup_node"]:
        p_lat = G.nodes[amb["pickup_node"]]['y']
        p_lon = G.nodes[amb["pickup_node"]]['x']
        folium.Marker(
            location=(p_lat, p_lon), popup="🚨 Emergency Pickup",
            icon=folium.DivIcon(
                html='<div style="font-size:32px;animation:blink 1s infinite">🚨</div>'
                     '<style>@keyframes blink{0%,100%{opacity:1}50%{opacity:0.2}}</style>')
        ).add_to(m)

    for h in data["hospital_nodes"]:
        is_dest = h == data.get("destination")
        hl, hlo = G.nodes[h]['y'], G.nodes[h]['x']
        folium.CircleMarker(
            location=(hl, hlo),
            radius=12 if is_dest and amb["phase"] == "ToHospital" else 8,
            color="red" if is_dest and amb["phase"] == "ToHospital" else "blue",
            fill=True,
            fillColor="#ff7f7f" if is_dest and amb["phase"] == "ToHospital" else "lightblue",
            fillOpacity=0.6, weight=2,
            popup="🏥 Destination" if is_dest else "🏥 Hospital"
        ).add_to(m)

    folium.Marker(
        location=current_pos,
        popup=f"🚑 {selected_ambulance}",
        icon=folium.DivIcon(
            html='<div id="amb-marker" style="font-size:38px;transition:none">🚑</div>',
            icon_size=(48, 48), icon_anchor=(24, 24)
        )
    ).add_to(m)

    if amb["auto_drive"] and amb["status"] != "Arrived":
        remaining_coords = route_coords[amb["step"]:]
        interval_ms = int(MOVE_INTERVAL_SEC * 1000)
        js_waypoints = str(remaining_coords).replace("[", "[").replace("]", "]")
        smooth_js = f"""
        <script>
        (function() {{
            var waypoints = {js_waypoints};
            var idx = 0;
            var interval = {interval_ms};

            function moveMarker() {{
                var maps = Object.values(window).filter(function(v) {{
                    return v && v._leaflet_id && v.eachLayer;
                }});
                if (!maps.length) return;
                var map = maps[maps.length - 1];

                map.eachLayer(function(layer) {{
                    if (layer._icon && layer._icon.querySelector && layer._icon.querySelector('#amb-marker')) {{
                        if (idx < waypoints.length) {{
                            var latlng = L.latLng(waypoints[idx][0], waypoints[idx][1]);
                            layer.setLatLng(latlng);
                            idx++;
                        }} else {{
                            clearInterval(timer);
                        }}
                    }}
                }});
            }}

            var ready = setInterval(function() {{
                var maps = Object.values(window).filter(function(v) {{
                    return v && v._leaflet_id && v.eachLayer;
                }});
                if (maps.length) {{
                    clearInterval(ready);
                    var timer = setInterval(moveMarker, interval);
                    moveMarker();
                }}
            }}, 200);
        }})();
        </script>
        """
        m.get_root().html.add_child(folium.Element(smooth_js))

    folium_static(m, height=700, width=None)

    st.progress(progress, text=f"🚑 Mission Progress: {progress}%  |  Step {amb['step']+1}/{len(amb['routes_data']['optimal_route'])}")

    # ---------------- STATUS ----------------
    if amb["step"] == 0:
        if amb["phase"] == "ToPickup":
            st.info("🚨 Ambulance dispatched to pickup location")
        elif amb["phase"] == "ToHospital":
            st.info("🏥 Transporting patient to hospital")
        else:
            st.info("🚨 Emergency dispatch initiated")
    elif amb["step"] == len(amb["routes_data"]["optimal_route"]) - 1:
        if amb["phase"] == "ToPickup":
            st.warning("📍 Patient picked up. Click ▶️ Start to continue to hospital.")
        elif amb["phase"] == "ToHospital":
            st.success("✅ Ambulance arrived at hospital!")
        else:
            st.success("✅ Ambulance arrived at destination!")

if any_auto_drive:
    time.sleep(MOVE_INTERVAL_SEC)
    for amb in st.session_state.fleet.values():
        if not (amb["routes_data"] and amb["auto_drive"]):
            continue
        route = amb["routes_data"]["optimal_route"]
        last_idx = len(route) - 1
        if amb["status"] == "Arrived" or amb["step"] >= last_idx:
            amb["auto_drive"] = False
            sync_ambulance_state(amb)
            continue
        amb["step"] += 1
        if amb["step"] >= last_idx:
            amb["auto_drive"] = False
        sync_ambulance_state(amb)
    st.rerun()