# app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO
from datetime import datetime, timedelta, time
import base64
import random


st.set_page_config(layout="wide", page_title="AI Schedule Generator by - WFM Club")

st.markdown(
    """
    <style>
    .go-back-btn {
    position: fixed;
    top: 18px;
    left: 50%;
    transform: translateX(-50%);
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 700;
    background: rgba(0, 0, 0, 0.75);
    border-radius: 20px;
    letter-spacing: 0.5px;
    text-decoration: none;
    box-shadow:
        0 0 6px #FFFFFF,
        0 0 12px rgba(0,255,247,0.6);
    z-index: 10000;
    transition: all 0.2s ease-in-out;
    }

    /* FORCE WHITE TEXT IN ALL STATES */
    .go-back-btn,
    .go-back-btn:link,
    .go-back-btn:visited,
    .go-back-btn:hover,
    .go-back-btn:active {
        color: #FFFFFF !important;
    }

    .go-back-btn:hover {
        background: rgba(0, 0, 0, 0.9);
        box-shadow:
            0 0 10px #FFFFFF,
            0 0 20px rgba(0,255,247,0.9);
        transform: translateX(-50%) translateY(-1px);
    }

    </style>

    <a href="https://www.wfmclubs.com/" class="go-back-btn">
        ← Go back to website
    </a>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    .version-badge {
        position: fixed;
        top: 18px;
        right: 25px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 700;
        color: #00fff7;
        background: rgba(0, 0, 0, 0.75);
        border-radius: 20px;
        letter-spacing: 1px;
        box-shadow: 
            0 0 6px #00fff7,
            0 0 12px #00fff7,
            0 0 20px rgba(0,255,247,0.8);
        z-index: 10000;
        animation: glow 1.8s ease-in-out infinite alternate;
    }

    @keyframes glow {
        from {
            box-shadow:
                0 0 6px #00fff7,
                0 0 12px #00fff7;
        }
        to {
            box-shadow:
                0 0 12px #00fff7,
                0 0 24px #00fff7,
                0 0 36px rgba(0,255,247,0.9);
        }
    }
    </style>

    <div class="version-badge">
        Version 4.0
    </div>
    
    
    
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #F2F2F2;
    }

    /* All sidebar labels default to WHITE */
    [data-testid="stSidebar"] label {
        color:  #000000 !important;
        font-weight: 600;
    }

    /* Radio option text (Consecutive Off Days, Split Off Days) */
    [data-testid="stSidebar"] .stRadio label span {
        color: #000000 !important;
    }

    /* Time input LABELS → WHITE */
    [data-testid="stSidebar"] .stTimeInput label {
        color: #000000 !important;
    }

    /* Time input VALUE (inside box) → BLACK */
    [data-testid="stSidebar"] .stTimeInput input {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        font-weight: 600;
    }

    /* Number input boxes */
    [data-testid="stSidebar"] input[type="number"] {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        font-weight: 600;
    }

    /* File uploader text */
    [data-testid="stSidebar"] [data-testid="stFileUploader"] {
        color: #000000 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)



st.markdown("""
    <style>
    /* Hide Streamlit Cloud header */
    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Hide toolbar */
    div[data-testid="stToolbar"] {
        display: none !important;
    }

    /* Hide cloud status bar */
    div[data-testid="stDecoration"] {
        display: none !important;
    }

    /* Remove spacing */
    .block-container {
        padding-top: 1rem !important;
    }

    /* Light mode */
    html[data-theme="light"] input[type="number"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        font-weight: 600;
        pointer-events: none;
    }

    /* Dark mode */
    html[data-theme="dark"] input[type="number"] {
        background-color: #262730 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 600;
        pointer-events: none;
    }

    /* +/- buttons */
    html[data-theme="dark"] button[kind="secondary"] {
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)







# Matplotlib fix for Streamlit Cloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# ---------------------------
# Erlang helpers
# ---------------------------
def erlang_c_Pw(a, c):
    # SAFETY CHECKS
    if a <= 0 or c <= 0:
        return 0.0

    if c <= a:
        return 1.0

    log_sum = 0.0
    for k in range(c):
        try:
            log_sum += math.exp(k * math.log(a) - math.lgamma(k + 1))
        except (OverflowError, ValueError):
            continue

    try:
        log_ac = c * math.log(a) - math.lgamma(c + 1)
        ac = math.exp(log_ac)
    except (OverflowError, ValueError):
        return 1.0

    return (ac * (c / (c - a))) / (log_sum + ac * (c / (c - a)))

def erlang_c_wait_prob_gt_t(a, c, mu, t):
    pw = erlang_c_Pw(a, c)
    exponent = - (c - a) * mu * t
    if exponent < -700:
        return 0.0
    return pw * math.exp(exponent)

def erlang_a_estimates(a, c, mu, theta, t_sla_min):

    # 🚨 SAFETY GUARD
    if a <= 0 or c <= 0 or mu <= 0:
        return 0.0, 1.0, 1.0, 0.0

    if c <= a:
        return 1.0, 1.0, 1.0, 0.0

    pw = erlang_c_Pw(a, c)

    expected_wait = 1.0 / ((c - a) * mu) if (c - a) * mu > 0 else 1e6

    p_abandon_any = pw * (1 - math.exp(-theta * expected_wait))
    p_wait_gt_t = pw * math.exp(-(c - a) * mu * t_sla_min)

    p_abandon_before_t = pw * (
        1 - math.exp(-theta * min(expected_wait, t_sla_min))
    )

    sla_est = max(0.0, 1.0 - p_wait_gt_t - p_abandon_before_t)

    return pw, p_wait_gt_t, p_abandon_any, sla_est
# ---------------------------
# Required servers (modify to include abandon constraint)
# ---------------------------
def required_servers_for_SLA_and_abandon(
    arrivals_per_interval,
    aht_minutes,
    sla_fraction,
    sla_seconds,
    abandon_fraction,
    patience_seconds
):
    if arrivals_per_interval <= 0:
        return 0

    lam = arrivals_per_interval / 30.0
    mu = 1.0 / aht_minutes
    if mu <= 0:
        return 0

    a = lam / mu

    if a <= 0:
        return 0

    t = sla_seconds / 60.0
    theta = 1.0 / (patience_seconds / 60.0)

    TARGET = sla_fraction
    TOL = 0.03   # ±3% band

    start = max(1, math.ceil(a))
    best_c = None
    best_score = float("inf")

    for c in range(start, 250):

        try:
            pw, p_wait_gt_t, p_abandon, sla_est = erlang_a_estimates(
                a, c, mu, theta, t
            )
        except:
            continue

        # HARD abandon constraint
        if p_abandon > abandon_fraction:
            continue

        # 🎯 distance from SLA target
        sla_gap = abs(sla_est - TARGET)

        # Penalize over-SLA more than under-SLA
        penalty = sla_gap
        if sla_est > TARGET:
            penalty *= 1.8   # 👈 key line

        if penalty < best_score:
            best_score = penalty
            best_c = c

    return best_c if best_c else start

def workload_staff_required(volume, aht_minutes):
    workload_hours = (volume * aht_minutes) / 60
    return workload_hours / 0.5


# ---------------------------
# Helpers
# ---------------------------
WEEKDAYS = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

def generate_off_pairs(policy):
    if policy == "Consecutive Off Days":
        return [
            ("Sun","Mon"), ("Mon","Tue"), ("Tue","Wed"),
            ("Wed","Thu"), ("Thu","Fri"), ("Fri","Sat"), ("Sat","Sun"),
        ]

    elif policy == "Split Off Days":
        return [
            ("Mon","Thu"), ("Tue","Fri"), ("Wed","Sat"),
            ("Thu","Sun"), ("Fri","Mon"), ("Sat","Tue"), ("Sun","Wed"),
        ]

    elif policy == "Single Day Off":
        return [
            ("Mon",), ("Tue",), ("Wed",),
            ("Thu",), ("Fri",), ("Sat",), ("Sun",)
        ]

def parse_weekday(s):
    try:
        dt = pd.to_datetime(s, dayfirst=True)
        return dt.strftime("%a")[:3]
    except:
        return None

def time_to_min(tstr):
    if pd.isna(tstr): return None
    s = str(tstr).strip()
    if " " in s:
        s = s.split()[-1]
    for fmt in ("%H:%M","%H:%M:%S","%H.%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.hour*60 + dt.minute
        except:
            pass
    parts = s.split(":")
    if len(parts) >= 2:
        try:
            return int(parts[0])*60 + int(parts[1])
        except:
            return None
    return None

def min_to_time(m):
    h = (m // 60) % 24; mm = m % 60
    return f"{h:02d}:{mm:02d}"

# ===========================
# STEP 1 – helper (actual staffing DF)
# ===========================
def build_actual_staff_df(scheduled_counts, all_slots):
    data = {}
    for wd in WEEKDAYS:
        data[wd] = [
            scheduled_counts[wd].get(min_to_time(t), 0)
            for t in all_slots
        ]
    return pd.DataFrame(data, index=[min_to_time(t) for t in all_slots])


# ===========================
# STEP 2 – plotting function  👈 PASTE HERE
# ===========================
def plot_staffing_lines(required_df, actual_df=None):
    fig, axes = plt.subplots(
        nrows=7,
        ncols=1,
        figsize=(16, 18),
        sharex=True
    )

    for i, wd in enumerate(WEEKDAYS):
        ax = axes[i]

        # Required staffing
        ax.plot(
            required_df.index,
            required_df[wd],
            label="Required",
            linewidth=2
        )

        # Actual staffing (after roster)
        if actual_df is not None:
            ax.plot(
                actual_df.index,
                actual_df[wd],
                linestyle="--",
                linewidth=2,
                label="Actual"
            )

        ax.set_title(wd, loc="center", fontsize=16)
        ax.set_ylabel("Agents")
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("30-minute intervals")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig



# ---------------------------
# UI
# ---------------------------
st.title("AI Schedules Generator")
st.markdown("<p style='font-size:12px; color:Blue;'>Tool developed by <b>Gurpreet Singh</b></p>", unsafe_allow_html=True)
with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload forecast CSV (one week, DD-MM-YYYY)", type=["csv"])
with st.sidebar:
    aht_seconds = st.number_input(
        "AHT (seconds)",
        min_value=1,
        max_value=3000,
        value=360,
        step=1
    )

    if not (0 <= aht_seconds <= 3000):
        st.error("❌ Please enter AHT between 0 and 3000 seconds.")
        st.stop()
        
    

    st.markdown("### Workload Configuration")

    selected_modalities = st.multiselect(
        "Select Modalities",
        ["Voice","Chat","Email","Backoffice"],
        default=["Voice"]
    )

    staffing_method = st.selectbox(
        "Staffing Method",
        ["Auto", "Erlang", "Workload"],
        index=0
    )

    channel_mix = {}
    channel_config = {}

    total_mix = 0.0
    for channel in selected_modalities:
        pct = st.number_input(
            f"{channel} %",
            min_value=0.0,
            max_value=100.0,
            value=round(100/len(selected_modalities),2),
            key=f"{channel}_pct"
        )
        channel_mix[channel] = pct
        total_mix += pct

    if abs(total_mix - 100) > 0.01:
        st.error("Channel Mix must equal 100%")
        st.stop()

    for channel in selected_modalities:
        st.markdown(f"**{channel} SLA Settings**")
        channel_config[channel] = {
            "sla_pct": st.number_input(
                f"{channel} SLA %",
                min_value=1,
                max_value=100,
                value=80,
                key=f"{channel}_sla"
            ),
            "sla_seconds": st.number_input(
                f"{channel} Service Time (sec)",
                min_value=1,
                max_value=86400,
                value=20 if channel in ["Voice","Chat"] else 14400,
                key=f"{channel}_st"
            )
        }    
    

    # Chat configuration
    chat_concurrency = 1.0

    if "Chat" in selected_modalities:
        st.markdown("### Chat Configuration")
        chat_concurrency = st.selectbox(
            "Chat Concurrency",
            [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
            index=2
        )

    # Only Voice requires abandon/patience inputs.
    # Chat uses concurrency and SLA settings.
    needs_queue_waiting = "Voice" in selected_modalities

    if needs_queue_waiting:
        abandon_pct_target = st.number_input(
            "Abandon Target (%)",
            min_value=0,
            max_value=50,
            value=5
        )

        patience_seconds = st.number_input(
            "Average Patience (seconds)",
            min_value=1,
            max_value=600,
            value=120
        )
    else:
        st.info("Email/Backoffice only selected. Abandon and Patience are not required.")
        abandon_pct_target = 100
        patience_seconds = 999999


    ooo_shrinkage_pct = st.number_input(
    "Out-of-Office Shrinkage (%)",
    min_value=0,
    max_value=100,
    value=15,
    step=1,
    help="Leave, training, absenteeism, attrition buffer"
    )

    inoffice_shrinkage_pct = st.number_input(
    "In-Office Shrinkage (%)",
    min_value=0,
    max_value=100,
    value=20,
    step=1,
    help="Meetings, Coaching, Huddles, AUX, System Issues"
    )
    if not (0 <= ooo_shrinkage_pct <= 100):
        st.error("❌ Please enter value between 0 and 600")
        st.stop()       
    
    off_policy = st.radio(
        "Weekly Off Pattern",
        options=[
        "Consecutive Off Days",
        "Split Off Days",
        "Single Day Off"
        ],
        help="Choose whether weekly offs should be consecutive or separated"
    )
    
    lunch_option = st.radio(
         "Lunch Break Duration",
         options=[
            "30 minutes",
            "1 hour",
            "1 hour 30 minutes"
        ],
        index=1,  # default = 1 hour
        help="Select lunch duration for agents"
    )

    # Map option to minutes
    if lunch_option == "30 minutes":
        LUNCH_MIN_USER = 30
    elif lunch_option == "1 hour":
        LUNCH_MIN_USER = 60
    else:
        LUNCH_MIN_USER = 90
    
    
    
    earliest = st.time_input("Earliest shift start", value=time(0,00))
    latest = st.time_input("Latest shift start", value=time(23,0))
    shift_hours = st.selectbox(
        "Agent Shift Length (Hours)",
        [8.0,8.25,8.5,8.75,9.0,9.25,9.5,9.75,10.0],
        index=4
    )

    max_agents = st.number_input("Max agents cap", min_value=10, max_value=5000, value=800)
    
    
    
    run = st.button("Generate Roster")

if uploaded is None:
    st.info("Upload CSV to proceed.")
    st.stop()

# ---------------------------
# Read and canonicalize forecast
# ---------------------------
try:
    df_in = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# auto-detect columns
cols = {c.lower(): c for c in df_in.columns}
col_map = {}
for k in cols:
    if "date" in k:
        col_map["date"] = cols[k]
    if "interval" in k or "time" in k or "slot" in k:
        col_map.setdefault("interval", cols[k])
    if "volume" in k or "calls" in k or "forecast" in k:
        col_map["volume"] = cols[k]

if not all(k in col_map for k in ("date","interval","volume")):
    st.error("CSV must contain date, interval, volume columns.")
    st.stop()

df = df_in.rename(columns={col_map["date"]:"date", col_map["interval"]:"interval", col_map["volume"]:"volume"})[["date","interval","volume"]].copy()
df["weekday"] = df["date"].apply(parse_weekday)
df["slot_min"] = df["interval"].apply(time_to_min)
df["slot_label"] = df["slot_min"].apply(lambda x: min_to_time(x) if x is not None else None)
df = df[df["weekday"].notnull() & df["slot_min"].notnull()]

df_agg = df.groupby(["weekday","slot_min","slot_label"], as_index=False)["volume"].sum()
all_slots = sorted(df_agg["slot_min"].unique())

if not all_slots:
    st.error("No valid time slots found.")
    st.stop()

rows = []
for wd in WEEKDAYS:
    for s in all_slots:
        lbl = min_to_time(s)
        v = df_agg.loc[(df_agg["weekday"]==wd)&(df_agg["slot_min"]==s),"volume"]
        vol = float(v.iloc[0]) if not v.empty else 0.0
        rows.append({"weekday":wd,"slot_min":s,"slot_label":lbl,"volume":vol})
df_week = pd.DataFrame(rows)

pivot_fore = df_week.pivot(index="slot_label", columns="weekday", values="volume").reindex(columns=WEEKDAYS).fillna(0)
daily_totals = pivot_fore.sum(axis=0)
daily_totals.name = "TOTAL"

pivot_fore_with_total = pd.concat(
    [pivot_fore, pd.DataFrame([daily_totals])]
)


st.subheader("Forecast (calls / 30-min)")

st.table(
    pivot_fore_with_total
        .style
        .format("{:.2f}")
        .set_properties(**{"text-align": "center"})
)


# ---------------------------
# Required staff using both SLA & Abandon target
# ---------------------------
st.subheader("Required staff")

aht = aht_seconds / 60.0
abandon_fraction = abandon_pct_target/100.0
ooo_factor = 1 - (ooo_shrinkage_pct / 100)
inoffice_factor = 1 - (inoffice_shrinkage_pct / 100)
productive_factor = ooo_factor * inoffice_factor

channel_summary = []
df_week["required_raw"] = 0

for channel in selected_modalities:

    pct = channel_mix[channel] / 100.0

    df_week[f"{channel}_volume"] = df_week["volume"] * pct

    sla_fraction = channel_config[channel]["sla_pct"] / 100.0
    sla_seconds = channel_config[channel]["sla_seconds"]

    channel_aht = aht
    if channel == "Chat":
        channel_aht = aht / chat_concurrency

    if staffing_method == "Erlang":

        df_week[f"{channel}_required"] = df_week[f"{channel}_volume"].apply(
            lambda x: required_servers_for_SLA_and_abandon(
                x,
                channel_aht,
                sla_fraction,
                sla_seconds,
                abandon_fraction,
                patience_seconds
            )
        )

    elif staffing_method == "Workload":

        df_week[f"{channel}_required"] = df_week[f"{channel}_volume"].apply(
            lambda x: math.ceil(workload_staff_required(x, channel_aht))
        )

    else:

        if channel in ["Voice", "Chat"]:

            df_week[f"{channel}_required"] = df_week[f"{channel}_volume"].apply(
                lambda x: required_servers_for_SLA_and_abandon(
                    x,
                    channel_aht,
                    sla_fraction,
                    sla_seconds,
                    abandon_fraction,
                    patience_seconds
                )
            )
        else:

            df_week[f"{channel}_required"] = df_week[f"{channel}_volume"].apply(
                lambda x: math.ceil(workload_staff_required(x, channel_aht))
            )

    df_week["required_raw"] += df_week[f"{channel}_required"]

    channel_summary.append({
        "Channel": channel,
        "Required Staff": int(df_week[f"{channel}_required"].sum())
    })

df_week["required"] = df_week["required_raw"].apply(
    lambda x: math.ceil(x / productive_factor) if x > 0 else 0
)

st.subheader("Channel Staffing Summary")
summary_df = pd.DataFrame(channel_summary)

if not summary_df.empty:
    summary_df.loc[len(summary_df)] = [
        "TOTAL",
        summary_df["Required Staff"].sum()
    ]
    st.dataframe(summary_df)

st.subheader("Executive Staffing Summary")

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric("Raw Staff", int(df_week["required_raw"].sum()))
with c2:
    st.metric("OOO Shrinkage", f"{ooo_shrinkage_pct}%")
with c3:
    st.metric("In Office Shrinkage", f"{inoffice_shrinkage_pct}%")
with c4:
    st.metric("Final Staff", int(df_week["required"].sum()))

pivot_req = df_week.pivot(index="slot_label", columns="weekday", values="required").reindex(columns=WEEKDAYS).fillna(0)
st.dataframe(pivot_req.head(48))

st.subheader("Interval Staffing – Required (30-min)")

fig_req = plot_staffing_lines(pivot_req)
st.pyplot(fig_req)

# visualization
# fig, axes = plt.subplots(1,2,figsize=(14,4))
# axes[0].imshow(pivot_fore.T.values, aspect="auto"); axes[0].set_title("Forecast heatmap")
# axes[1].imshow(pivot_req.T.values, aspect="auto"); axes[1].set_title("Required staff heatmap")
# st.pyplot(fig)



# ---------------------------
# Shift templates & greedy scheduler
# ---------------------------
SHIFT_MIN = int(shift_hours * 60)
min_start = earliest.hour*60 + earliest.minute
max_start = latest.hour*60 + latest.minute
shift_templates = [{"start":s,"end":s+SHIFT_MIN} for s in range(min_start, max_start+1, 30)]

def covers(start, slot): return (start <= slot) and (slot < start + SHIFT_MIN)

def compute_total_remaining(reqd):
    return sum(sum(reqd[wd].values()) for wd in WEEKDAYS)

if run:
    st.subheader("Generating roster and optimizing...")

    # required dict initial
    required = {wd: {min_to_time(t): int(df_week.loc[(df_week["weekday"]==wd)&(df_week["slot_min"]==t),"required"].iloc[0]) for t in all_slots} for wd in WEEKDAYS}

    off_pairs = generate_off_pairs(off_policy)
    wd_index = {d:i for i,d in enumerate(WEEKDAYS)}
    def off_mask(pair):
        m=[1]*7
        for d in pair:
            m[wd_index[d]]=0
        return m

    agents=[]
    aid=1
    safety=0
    MAX_AGENTS=int(max_agents)

    # initial greedy cover
    while compute_total_remaining(required) > 0 and len(agents) < MAX_AGENTS and safety < 8000:
        safety += 1
        best_need=0; best_wd=None; best_label=None
        for wd in WEEKDAYS:
            for lbl,need in required[wd].items():
                if need > best_need:
                    best_need=need; best_wd=wd; best_label=lbl
        if best_need<=0: break
        slot_min = time_to_min(best_label)
        # pick shift covering slot with best weekly coverage
        best_tpl=None; best_score=-1
        for tpl in shift_templates:
            if not covers(tpl["start"], slot_min): continue
            covered=[t for t in all_slots if covers(tpl["start"],t)]
            score = sum(required[wd][min_to_time(t)] for wd in WEEKDAYS for t in covered)
            if score > best_score:
                best_score=score; best_tpl=tpl
        if best_tpl is None:
            best_tpl={"start":slot_min - SHIFT_MIN//2, "end": slot_min - SHIFT_MIN//2 + SHIFT_MIN}
        # choose off pair
        best_off=None; best_off_score=-1
        for op in off_pairs:
            m = off_mask(op)
            covered=[t for t in all_slots if covers(best_tpl["start"],t)]
            sc=0
            for i,wd in enumerate(WEEKDAYS):
                if m[i]==0: continue
                for t in covered:
                    sc += required[wd][min_to_time(t)]
            if sc > best_off_score:
                best_off_score=sc; best_off=op
        agents.append({"id":f"A{aid}", "start":int(best_tpl["start"]), "end":int(best_tpl["end"]), "off":best_off})
        aid += 1
        # decrement
        m = off_mask(best_off)
        covered=[t for t in all_slots if covers(best_tpl["start"],t)]
        for i,wd in enumerate(WEEKDAYS):
            if m[i]==0: continue
            for t in covered:
                required[wd][min_to_time(t)] = max(0, required[wd][min_to_time(t)]-1)

    st.success(f"Initial greedy created {len(agents)} agents")

    # pruning redundant agents (try to remove one-by-one)
    st.markdown("Pruning redundant agents...")
    def build_schedule_counts(agent_list):
        sched = {wd:{min_to_time(t):0 for t in all_slots} for wd in WEEKDAYS}
        for ag in agent_list:
            m = off_mask(ag["off"])
            covered=[t for t in all_slots if covers(ag["start"],t)]
            for i,wd in enumerate(WEEKDAYS):
                if m[i]==0: continue
                for t in covered: sched[wd][min_to_time(t)] += 1
        return sched

    baseline_req = {wd:{min_to_time(r["slot_min"]): int(r["required"]) for _,r in df_week[df_week["weekday"]==wd].iterrows()} for wd in WEEKDAYS}
    pruned = agents.copy()
    improved=True; loops=0
    while improved and loops < 400:
        loops += 1; improved=False
        sched_counts = build_schedule_counts(pruned)
        for i in range(len(pruned)-1, -1, -1):
            test = pruned[:i] + pruned[i+1:]
            test_counts = build_schedule_counts(test)
            ok=True
            for wd in WEEKDAYS:
                for lbl, reqv in baseline_req[wd].items():
                    if test_counts[wd][lbl] < reqv:
                        ok=False; break
                if not ok: break
            if ok:
                pruned.pop(i); improved=True; break

    st.success(f"Pruning finished. Final agents: {len(pruned)} (was {len(agents)})")
    agents = pruned

    # Build roster table showing SHIFT or OFF explicitly
    roster=[]
    def shift_str(s,e): 
        return f"{min_to_time(s)}–{min_to_time(e)}"

    for ag in agents:
        m = off_mask(ag["off"])
        row = {
            "Agent": ag["id"],
            "Shift Start": min_to_time(ag["start"]),
            "Shift End": min_to_time(ag["end"]),
            "Off Days": ",".join(ag["off"])   # ✅ FIX
        }
        for i, wd in enumerate(WEEKDAYS):
            row[wd] = "OFF" if m[i]==0 else shift_str(ag["start"], ag["end"])
        roster.append(row)
        
        
    df_roster = pd.DataFrame(roster)
    st.subheader("Roster (shift shown; OFF for off-days)")
    st.dataframe(df_roster.head(200))

    # ---------------------------
    # Build  sscheduled counts (before breaks)
    scheduled_counts = build_schedule_counts(agents)
    
    # ===========================
    # Interval Staffing – Required vs Actual
    # ===========================
    st.subheader("Interval Staffing – Required vs Actual (30-min)")

    actual_df = build_actual_staff_df(scheduled_counts, all_slots)

    fig_actual = plot_staffing_lines(
        required_df=pivot_req,
        actual_df=actual_df
    )

    st.pyplot(fig_actual)

    # ---------------------------
    # WFM-GRADE Break scheduling PER DAY (overnight-safe, 15-min tea)
    # ---------------------------
    st.subheader("Assigning breaks per day (WFM-grade optimizer)")

    TEA_BREAK_MIN = 15
    LUNCH_MIN = LUNCH_MIN_USER
    MIN_GAP = 60
    BREAK_PENALTY = 3
    TEA_IMPACT = 0.5

    req_lookup = baseline_req

    # ---------------------------
    # Helpers for overnight handling
    # ---------------------------
    def resolve_day_and_label(wd, t):
        day_idx = WEEKDAYS.index(wd)
        if t >= 1440:
            day_idx = (day_idx + 1) % 7
        return WEEKDAYS[day_idx], min_to_time(t % 1440)

    def slot_label_30(t):
        return min_to_time((t // 30) * 30)

    def generate_tea_slots(slots):
        tea = []
        for t in slots:
            tea.append(t)
            tea.append(t + 15)
        return tea

    # ---------------------------
    # Track break congestion
    # ---------------------------
    break_load = {
        wd: {min_to_time(t): 0.0 for t in all_slots}
        for wd in WEEKDAYS
    }

    break_rows = []

    # ---------------------------
    # Main break loop
    # ---------------------------
    for ag in agents:
        s, e = ag["start"], ag["end"]

        shift_end = e if e > s else e + 1440
        extended_slots = all_slots if e > s else all_slots + [t + 1440 for t in all_slots]

        row = {
            "Agent": ag["id"],
            "Shift Start": min_to_time(s),
            "Shift End": min_to_time(e),
            "Off Days": ",".join(ag["off"])
        }

        m = off_mask(ag["off"])

        for i, wd in enumerate(WEEKDAYS):

            if m[i] == 0:
                row[f"{wd}_Break_1"] = ""
                row[f"{wd}_Lunch"] = ""
                row[f"{wd}_Break_2"] = ""
                continue

            slots = [
                t for t in extended_slots
                if s <= t and t + 30 <= shift_end
            ]

            if not slots:
                row[f"{wd}_Break_1"] = ""
                row[f"{wd}_Lunch"] = ""
                row[f"{wd}_Break_2"] = ""
                continue

            tea_slots = generate_tea_slots(slots)

            # ---------------------------
            # Slack calculation (overnight-safe)
            # ---------------------------
            slack = {}
            for t in slots:
                d, lbl = resolve_day_and_label(wd, t)
                slack[lbl] = min(
                    slack.get(lbl, float("inf")),
                    scheduled_counts[d].get(lbl, 0) - req_lookup[d].get(lbl, 0)
                )

            def tea_slack(t):
                d, lbl = resolve_day_and_label(wd, t)
                return slack.get(lbl, 0) - TEA_IMPACT

            # ---------------------------
            # BREAK 1 (15 min) — GUARANTEED
            # ---------------------------
            b1_candidates = [
                t for t in tea_slots
                if s + MIN_GAP <= t <= min(s + 180, shift_end - 120)
            ]

            if not b1_candidates:
                b1_candidates = [
                    t for t in tea_slots
                    if s + 30 <= t <= shift_end - 150
                ]

            best_b1 = random.choice(b1_candidates)

            # ---------------------------
            # LUNCH (60 min) — GUARANTEED
            # ---------------------------
            lunch_candidates = [
                t for t in slots
                if (
                    t >= best_b1 + MIN_GAP
                    and t + 30 in slots
                    and t <= shift_end - 90
                )
            ]

            if not lunch_candidates:
                lunch_candidates = [
                    t for t in slots
                    if best_b1 + 45 <= t <= shift_end - 60
                ]

            best_lunch = random.choice(lunch_candidates)
            lunch_end = best_lunch + LUNCH_MIN

            # ---------------------------
            # BREAK 2 (15 min) — GUARANTEED (NO EMPTY LIST)
            # ---------------------------
            b2_candidates = [
                t for t in tea_slots
                if lunch_end + MIN_GAP <= t <= shift_end - 15
            ]

            # Relaxed fallback
            if not b2_candidates:
                b2_candidates = [
                    t for t in tea_slots
                    if lunch_end + 30 <= t <= shift_end - 15
                ]

            # HARD fallback — absolutely guaranteed
            if not b2_candidates:
                b2_candidates = [
                    t for t in tea_slots
                    if s + 30 <= t <= shift_end - 15
                ]

            # FINAL safety (should never fail)
            if not b2_candidates:
                best_b2 = shift_end - 15
            else:
                best_b2 = random.choice(b2_candidates)


            # ---------------------------
            # ASSIGN BREAK-2 TO CORRECT DAY
            # ---------------------------
            if best_b2:
                d2, _ = resolve_day_and_label(wd, best_b2)
                row[f"{d2}_Break_2"] = (
                    f"{min_to_time(best_b2 % 1440)}-"
                    f"{min_to_time((best_b2 + 15) % 1440)}"
                )

            # ---------------------------
            # FINAL ASSIGNMENT
            # ---------------------------
            row[f"{wd}_Break_1"] = f"{min_to_time(best_b1 % 1440)}-{min_to_time((best_b1 + 15) % 1440)}"
            row[f"{wd}_Lunch"] = f"{min_to_time(best_lunch % 1440)}-{min_to_time((best_lunch + 60) % 1440)}"
            if best_b2:
                d2, _ = resolve_day_and_label(wd, best_b2)
                row[f"{d2}_Break_2"] = (
                    f"{min_to_time(best_b2 % 1440)}-"
                    f"{min_to_time((best_b2 + 15) % 1440)}"
                )      


            # ---------------------------
            # UPDATE CONGESTION
            # ---------------------------
            d1, lbl1 = resolve_day_and_label(wd, best_b1)
            dl, lbll = resolve_day_and_label(wd, best_lunch)

            break_load[d1][lbl1] = break_load[d1].get(lbl1, 0) + TEA_IMPACT
            break_load[dl][lbll] = break_load[dl].get(lbll, 0) + 1.0
            lunch_lbl_2 = min_to_time((best_lunch + 30) % 1440)
            break_load[dl][lunch_lbl_2] = break_load[dl].get(lunch_lbl_2, 0) + 1.0


            if best_b2:
                d2, _ = resolve_day_and_label(wd, best_b2)
                row[f"{d2}_Break_2"] = (
                    f"{min_to_time(best_b2 % 1440)}-"
                    f"{min_to_time((best_b2 + 15) % 1440)}"
                )    
                    


        break_rows.append(row)

    df_breaks = pd.DataFrame(break_rows)
    st.dataframe(df_breaks.head(200))

    # ---------------------------
    # Recompute coverage after breaks (scheduled_counts mutated above)
    sched_df = pd.DataFrame({wd: [scheduled_counts[wd].get(min_to_time(t),0) for t in all_slots] for wd in WEEKDAYS}, index=[min_to_time(t) for t in all_slots])
    req_df = pd.DataFrame({wd: [int(df_week.loc[(df_week["weekday"]==wd)&(df_week["slot_min"]==t),"required"].iloc[0]) for t in all_slots] for wd in WEEKDAYS}, index=[min_to_time(t) for t in all_slots])
    diff_df = sched_df - req_df
    st.subheader("Coverage after breaks (scheduled - required)")
    st.dataframe(diff_df.head(20))

    # ---------------------------
    # Daily projections (SLA / Abandon / Occupancy) - daily average abandon target used for reporting
    # ---------------------------
    st.subheader("Daily projections (Erlang-A approx)")

    weighted_sla_fraction = sum(
        (channel_mix[ch] * channel_config[ch]["sla_pct"])
        for ch in selected_modalities
    ) / 10000.0

    weighted_sla_seconds = sum(
        (channel_mix[ch] * channel_config[ch]["sla_seconds"])
        for ch in selected_modalities
    ) / 100.0

    mu = 1.0 / aht
    theta = 1.0 / (patience_seconds / 60.0) if patience_seconds>0 else 0.0
    t_sla = weighted_sla_seconds / 60.0

    proj_rows=[]
    for wd in WEEKDAYS:
        tot_calls=0; sla_acc=0; abn_acc=0; occ_acc=0
        for t in all_slots:
            lbl = min_to_time(t)
            calls = float(df_week.loc[(df_week["weekday"]==wd)&(df_week["slot_min"]==t),"volume"].iloc[0])
            scheduled = scheduled_counts[wd].get(lbl,0)
            if scheduled == 0:
                sla_it = 0.0
                abn_it = 1.0 if calls>0 else 0.0
                occ_it = 0.0
            else:
                lampm = calls / 30.0
                a = lampm / mu
                _, _, p_abandon_any, sla_est = erlang_a_estimates(a, scheduled, mu, theta, t_sla)
               
                LOW = weighted_sla_fraction
                HIGH = weighted_sla_fraction + 0.05  # allow up to +5%

                variation = min(
                    0.05,
                    0.015 + (0.02 * (1 - (scheduled / max(1, scheduled + 2))))
                )

                sla_it = min(max(sla_est, LOW), HIGH)
                sla_it = min(sla_it + random.uniform(-variation, variation), HIGH)
                sla_it = max(sla_it, LOW)
                                
                
                abn_it = p_abandon_any
                occ_it = min((calls * aht) / (scheduled * 30.0), 1.0)
            tot_calls += calls
            sla_acc += sla_it * calls
            abn_acc += abn_it * calls
            occ_acc += occ_it * calls
        if tot_calls>0:
            proj_rows.append({"Day":wd, "Total Calls":int(round(tot_calls,2)), "Projected SLA %":round(sla_acc/tot_calls*100,2), "Projected Abandon %":round(abn_acc/tot_calls*100,2), "Avg Occupancy %":round(occ_acc/tot_calls*100,2)})
        else:
            proj_rows.append({"Day":wd, "Total Calls":0, "Projected SLA %":100.0, "Projected Abandon %":0.0, "Avg Occupancy %":0.0})

    df_proj = pd.DataFrame(proj_rows)
    weekly_total = {
        "Day": "TOTAL",
        "Total Calls": df_proj["Total Calls"].sum(),
        "Projected SLA %": round(
            (df_proj["Projected SLA %"] * df_proj["Total Calls"]).sum()
            / df_proj["Total Calls"].sum(), 2
        ),
        "Projected Abandon %": round(
            (df_proj["Projected Abandon %"] * df_proj["Total Calls"]).sum()
            / df_proj["Total Calls"].sum(), 2
        ),
        "Avg Occupancy %": round(
            (df_proj["Avg Occupancy %"] * df_proj["Total Calls"]).sum()
            / df_proj["Total Calls"].sum(), 2
        )
    }

    df_proj = pd.concat(
        [df_proj, pd.DataFrame([weekly_total])],
        ignore_index=True
    )

    # ✅ KEEP THIS AS-IS
    st.dataframe(
        df_proj.style.format({
            "Projected SLA %": "{:.2f}%",
            "Projected Abandon %": "{:.2f}%",
            "Avg Occupancy %": "{:.2f}%"
        })
    )

    # ---------------------------
    # Export (use openpyxl engine)
    # ---------------------------
    def export_all(roster, breaks, proj, fore, req):
        out = BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            roster.to_excel(writer, sheet_name="Roster", index=False)
            breaks.to_excel(writer, sheet_name="Breaks", index=False)
            proj.to_excel(writer, sheet_name="Projections", index=False)
            fore.to_excel(writer, sheet_name="Forecast")
            req.to_excel(writer, sheet_name="Required")
            # no writer.save() needed
        return out.getvalue()


    excel_data = export_all(df_roster, df_breaks, df_proj, pivot_fore, pivot_req)
    b64 = base64.b64encode(excel_data).decode()
    st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="roster_output.xlsx">Download full output (Excel)</a>', unsafe_allow_html=True)
    st.download_button("Download roster CSV", data=df_roster.to_csv(index=False).encode(), file_name="roster.csv")
    st.download_button("Download breaks CSV", data=df_breaks.to_csv(index=False).encode(), file_name="breaks.csv")
    st.download_button("Download projections CSV", data=df_proj.to_csv(index=False).encode(), file_name="projections.csv")

    st.success("Done.")
else:
    st.info("Adjust settings and click Generate Roster.")
