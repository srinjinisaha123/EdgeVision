import streamlit as st
import sqlite3
import pandas as pd
import time

# 1. Page Configuration
st.set_page_config(
    page_title="EdgeVision Telematics Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Style Header
st.title("🛡️ EdgeVision: Real-Time Telematics & ADAS Analytics")
st.markdown("---")

# 2. Database Loader Function
def load_data():
    try:
        conn = sqlite3.connect("edgevision_telematics.db")
        df = pd.read_sql_query("SELECT * FROM safety_logs ORDER BY id DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame(columns=["id", "timestamp", "event_type", "details"])

# 3. Sidebar Controls
st.sidebar.header("🕹️ Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Enable Live Auto-Refresh (2s)", value=True)
if st.sidebar.button("🗑️ Clear Telematics Log"):
    try:
        conn = sqlite3.connect("edgevision_telematics.db")
        c = conn.cursor()
        c.execute("DELETE FROM safety_logs")
        conn.commit()
        conn.close()
        st.sidebar.success("Database cleared successfully!")
    except Exception:
        pass

# 4. Fetch Latest Data
df = load_data()

# 5. Top Level KPI Metrics
col1, col2, col3 = st.columns(3)
total_alerts = len(df)
drowsy_count = len(df[df['event_type'] == 'DRIVER_DROWSINESS']) if not df.empty else 0
collision_count = len(df[df['event_type'] == 'FORWARD_COLLISION_HAZARD']) if not df.empty else 0

with col1:
    st.metric(label="🚨 Total Safety Violations", value=total_alerts)
with col2:
    st.metric(label="😴 Driver Fatigue Events", value=drowsy_count, delta="High Priority" if drowsy_count > 3 else "Normal", delta_color="inverse")
with col3:
    st.metric(label="🚗 Forward Collision Warnings", value=collision_count, delta="Critical" if collision_count > 5 else "Normal", delta_color="inverse")

st.markdown("---")

# 6. Analytics & Logs Section
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📊 Violations by Type")
    if not df.empty:
        type_counts = df['event_type'].value_counts()
        st.bar_chart(type_counts, color="#FF4B4B")
    else:
        st.info("No safety violations recorded yet. Drive safely!")

with col_right:
    st.subheader("📋 Live Telematics Event Log")
    if not df.empty:
        # Display clean dataframe
        st.dataframe(
            df[['timestamp', 'event_type', 'details']], 
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("Awaiting live data from EdgeVision engine...")

# Auto-refresh logic for real-time edge monitoring
if auto_refresh:
    time.sleep(2)
    st.rerun()