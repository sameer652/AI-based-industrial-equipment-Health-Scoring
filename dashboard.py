"""
Streamlit dashboard: fleet overview, machine detail charts, and alerts.
Refreshes itself every few seconds.
"""
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = "http://127.0.0.1:8000"
EMOJI = {"HEALTHY": "🟢", "WARNING": "🟡", "CRITICAL": "🔴"}
SENSORS = ["temperature", "vibration", "pressure", "rpm", "current"]

st.set_page_config(page_title="Equipment Health Monitor", layout="wide")
st.title("🏭 AI-Based Industrial Equipment Health Monitor")


def api_get(path: str, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Cannot reach the API at {API}. Is it running? ({e})")
        st.stop()


# ---------- Sidebar: model quality ----------
with st.sidebar:
    st.header("Model performance")
    info = api_get("/model/metrics")
    for name, value in info["metrics"].items():
        st.metric(name.capitalize(), f"{value:.1%}")
    st.caption("Evaluated on machines unseen during training (simulated data).")
    st.subheader("Most important sensors")
    st.bar_chart(pd.Series(info["feature_importance"]))


@st.fragment(run_every=3)
def live_view():
    equipment = pd.DataFrame(api_get("/equipment"))
    scored = equipment.dropna(subset=["score"])

    if scored.empty:
        st.info("No readings yet. Start the simulator: `python simulator.py`")
        return

    # ---------- KPI row ----------
    counts = scored["status"].value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Machines monitored", len(scored))
    c2.metric("🟢 Healthy", int(counts.get("HEALTHY", 0)))
    c3.metric("🟡 Warning", int(counts.get("WARNING", 0)))
    c4.metric("🔴 Critical", int(counts.get("CRITICAL", 0)))
    c5.metric("Fleet average health", f"{scored['score'].mean():.1f}")

    # ---------- Fleet table ----------
    st.subheader("Fleet overview")
    table = scored[["name", "type", "location", "score", "status"]].copy()
    table["status"] = table["status"].map(lambda s: f"{EMOJI[s]} {s}")
    table = table.sort_values("score")  # worst machines first
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "name": "Machine",
            "type": "Type",
            "location": "Location",
            "score": st.column_config.ProgressColumn(
                "Health score", min_value=0, max_value=100, format="%.1f"
            ),
            "status": "Status",
        },
    )

    left, right = st.columns([2, 1])

    # ---------- Machine detail ----------
    with left:
        st.subheader("Machine detail")
        names = scored.sort_values("id")["name"].tolist()
        choice = st.selectbox("Select machine", names, key="machine_select")
        machine_id = int(scored[scored["name"] == choice]["id"].iloc[0])

        history = pd.DataFrame(api_get(f"/equipment/{machine_id}/history", limit=200))
        if history.empty:
            st.info("No history for this machine yet.")
        else:
            fig = px.line(history, x="timestamp", y="score", title=f"{choice}: health score over time")
            fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Healthy")
            fig.add_hline(y=50, line_dash="dash", line_color="red", annotation_text="Critical")
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

            cols = st.columns(len(SENSORS))
            for col, sensor in zip(cols, SENSORS):
                col.caption(sensor.capitalize())
                col.line_chart(history[sensor], height=120)

    # ---------- Alerts ----------
    with right:
        st.subheader("🚨 Open alerts")
        alerts = api_get("/alerts", only_open=True)
        if not alerts:
            st.success("No open alerts")
        for alert in alerts[:10]:
            with st.container(border=True):
                st.write(f"{EMOJI[alert['severity']]} **{alert['equipment_name']}**")
                st.caption(f"{alert['message']}  \n{alert['created_at']}")
                if st.button("Acknowledge", key=f"ack_{alert['id']}"):
                    requests.post(f"{API}/alerts/{alert['id']}/acknowledge", timeout=5)
                    st.rerun(scope="fragment")


live_view()