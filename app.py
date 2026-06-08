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
        from { box-shadow: 0 0 6px #00fff7, 0 0 12px #00fff7; }
        to   { box-shadow: 0 0 12px #00fff7, 0 0 24px #00fff7, 0 0 36px rgba(0,255,247,0.9); }
    }
    </style>
    <div class="version-badge">Version 4.2</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #F2F2F2; }
    [data-testid="stSidebar"] label { color: #000000 !important; font-weight: 600; }
    [data-testid="stSidebar"] .stRadio label span { color: #000000 !important; }
    [data-testid="stSidebar"] .stTimeInput label { color: #000000 !important; }
    [data-testid="stSidebar"] .stTimeInput input { color: #000000 !important; background-color: #FFFFFF !important; font-weight: 600; }
    [data-testid="stSidebar"] input[type="number"] { color: #000000 !important; background-color: #FFFFFF !important; font-weight: 600; }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] { color: #000000 !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    header[data-testid="stHeader"] { display: none !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    .block-container { padding-top: 1rem !important; }
    html[data-theme="light"] input[type="number"] {
        background-color: #ffffff !important; color: #000000 !important;
        -webkit-text-fill-color: #000000 !important; font-weight: 600; pointer-events: none;
    }
    html[data-theme="dark"] input[type="number"] {
        background-color: #262730 !important; color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important; font-weight: 600; pointer-events: none;
    }
    html[data-theme="dark"] button[kind="secondary"] { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------
# Erlang helpers
# ---------------------------
def erlang_c_Pw(a, c):
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


def erlang_a_estimates(a, c, mu, theta, t_sla_min):
    if a <= 0 or c <= 0 or mu <= 0:
        return 0.0, 1.0, 1.0, 0.0
    if c <= a:
        return 1.0, 1.0, 1.0, 0.0
    pw = erlang_c_Pw(a, c)
    expected_wait = 1.0 / ((c - a) * mu) if (c - a) * mu > 0 else 1e6
    p_abandon_any = pw * (1 - math.exp(-theta * expected_wait))
    p_wait_gt_t = pw * math.exp(-(c - a) * mu * t_sla_min)
    p_abandon_before_t = pw * (1 - math.exp(-theta * min(expected_wait, t_sla_min)))
    sla_est = max(0.0, 1.0 - p_wait_gt_t - p_abandon_before_t)
    return pw, p_wait_gt_t, p_abandon_any, sla_est


def required_servers_for_SLA_and_abandon(
    arrivals_per_interval, aht_minutes, sla_fraction,
    sla_seconds, abandon_fraction, patience_seconds
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
    start = max(1, math.ceil(a))
    best_c = None
    best_score = float("inf")
    for c in range(start, 250):
        try:
            pw, p_wait_gt_t, p_abandon, sla_est = erlang_a_estimates(a, c, mu, theta, t)
        except:
            continue
        if p_abandon > abandon_fraction:
            continue
        sla_gap = abs(sla_est - TARGET)
        penalty = sla_gap * 1.8 if sla_est > TARGET else sla_gap
        if penalty < best_score:
            best_score = penalty
            best_c = c
    return best_c if best_c else start


def workload_staff_required(volume, aht_minutes):
    return (volume * aht_minutes) / 60 / 0.5


# ---------------------------
# Helpers
# ---------------------------
WEEKDAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def generate_off_pairs(policy):
    if policy == "Consecutive Off Days":
        return [("Sun","Mon"),("Mon","Tue"),("Tue","Wed"),("Wed","Thu"),("Thu","Fri"),("Fri","Sat"),("Sat","Sun")]
    elif policy == "Split Off Days":
        return [("Mon","Thu"),("Tue","Fri"),("Wed","Sat"),("Thu","Sun"),("Fri","Mon"),("Sat","Tue"),("Sun","Wed")]
    elif policy == "Single Day Off":
        return [("Mon",),("Tue",),("Wed",),("Thu",),("Fri",),("Sat",),("Sun",)]


def parse_weekday(s):
    try:
        return pd.to_datetime(s, dayfirst=True).strftime("%a")[:3]
    except:
        return None


def time_to_min(tstr):
    if pd.isna(tstr):
        return None
    s = str(tstr).strip()
    if " " in s:
        s = s.split()[-1]
    for fmt in ("%H:%M", "%H:%M:%S", "%H.%M"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.hour * 60 + dt.minute
        except:
            pass
    parts = s.split(":")
    if len(parts) >= 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return None
    return None


def min_to_time(m):
    h = (m // 60) % 24
    mm = m % 60
    return f"{h:02d}:{mm:02d}"


def build_actual_staff_df(scheduled_counts, all_slots):
    data = {wd: [scheduled_counts[wd].get(min_to_time(t), 0) for t in all_slots] for wd in WEEKDAYS}
    return pd.DataFrame(data, index=[min_to_time(t) for t in all_slots])


def plot_staffing_lines(required_df, actual_df=None):
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(16, 18), sharex=True)
    for i, wd in enumerate(WEEKDAYS):
        ax = axes[i]
        ax.plot(required_df.index, required_df[wd], label="Required", linewidth=2)
        if actual_df is not None:
            ax.plot(actual_df.index, actual_df[wd], linestyle="--", linewidth=2, label="Actual")
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
# Day-level SLA optimisation
# ---------------------------
def optimise_daily_sla(df_week_in, all_slots, aht_min, weighted_sla_target,
                        weighted_sla_seconds, patience_sec, sla_tolerance):
    mu = 1.0 / aht_min
    theta = 1.0 / (patience_sec / 60.0) if patience_sec > 0 else 1e-9
    t_sla = weighted_sla_seconds / 60.0
    floor_sla = weighted_sla_target - sla_tolerance
    df_out = df_week_in.copy()

    for wd in WEEKDAYS:
        mask = df_out["weekday"] == wd
        day_df = df_out[mask].copy()
        if day_df["volume"].sum() == 0:
            continue
        sorted_idx = day_df.sort_values("volume").index.tolist()

        def day_weighted_sla(day_df_local):
            total_vol = 0.0
            sla_vol = 0.0
            for _, r in day_df_local.iterrows():
                calls = r["volume"]
                c = int(r["required"])
                if calls <= 0:
                    continue
                a = (calls / 30.0) / mu
                if c == 0:
                    total_vol += calls
                    continue
                try:
                    _, _, _, sla_est = erlang_a_estimates(a, c, mu, theta, t_sla)
                except Exception:
                    sla_est = 0.0
                total_vol += calls
                sla_vol += sla_est * calls
            return (sla_vol / total_vol) if total_vol > 0 else 1.0

        for idx in sorted_idx:
            curr_req = int(day_df.at[idx, "required"])
            if curr_req <= 1:
                continue
            day_df.at[idx, "required"] = curr_req - 1
            if day_weighted_sla(day_df) >= floor_sla:
                pass
            else:
                day_df.at[idx, "required"] = curr_req

        df_out.loc[mask, "required"] = day_df["required"].values

    return df_out


# ---------------------------
# Email backlog simulation
# ---------------------------
def simulate_email_backlog(df_week, scheduled_counts, aht_minutes, email_frt_hours, all_slots, email_pct):
    # Blended idle-capacity model: live channels (Voice/Chat/etc.) are served first,
    # and only the leftover idle agent-time clears the email backlog — all at the
    # blended AHT. FRT is a service-level target, NOT a handle rate, so it does not
    # drive capacity here.
    rows = []
    carryover_backlog = 0.0
    interval_min = 30.0
    email_frac = email_pct / 100.0
    for wd in WEEKDAYS:
        for t in all_slots:
            lbl = min_to_time(t)
            vol_row = df_week.loc[(df_week["weekday"] == wd) & (df_week["slot_min"] == t)]
            total_vol = float(vol_row["volume"].iloc[0]) if not vol_row.empty else 0.0
            email_vol = total_vol * email_frac
            non_email_vol = total_vol * (1.0 - email_frac)
            scheduled = scheduled_counts[wd].get(lbl, 0)

            # Total agent-minutes on the floor this interval.
            agent_minutes = scheduled * interval_min
            # Minutes the live (non-email) channels consume at blended AHT.
            live_workload_minutes = non_email_vol * aht_minutes
            # Whatever agent-time is left over is available to work email down.
            idle_minutes = max(0.0, agent_minutes - live_workload_minutes)
            ticket_capacity = idle_minutes / aht_minutes if aht_minutes > 0 else 0.0

            available = carryover_backlog + email_vol
            handled = min(ticket_capacity, available)
            carryover_backlog = max(0.0, available - handled)
            rows.append({
                "Day": wd, "Interval": lbl,
                "Email Forecast": round(email_vol, 1),
                "Scheduled FTE": scheduled,
                "Idle Capacity (mins)": round(idle_minutes, 1),
                "Ticket Capacity (30min)": round(ticket_capacity, 2),
                "Tickets Handled": round(handled, 1),
                "Backlog (Carryover)": round(carryover_backlog, 1),
            })
    return pd.DataFrame(rows)


# ---------------------------
# Interval open numbers
# ---------------------------
def build_interval_open_numbers(scheduled_counts, all_slots, break_deductions,
                                 inoffice_shrinkage_pct, ooo_shrinkage_pct):
    productive_factor = (1 - ooo_shrinkage_pct / 100) * (1 - inoffice_shrinkage_pct / 100)
    rows_pre, rows_post = {}, {}
    for wd in WEEKDAYS:
        pre_col, post_col = [], []
        for t in all_slots:
            lbl = min_to_time(t)
            sched = scheduled_counts[wd].get(lbl, 0)
            break_ded = break_deductions[wd].get(lbl, 0.0)
            after_breaks = max(0.0, sched - break_ded)
            pre_col.append(round(after_breaks, 2))
            post_col.append(round(after_breaks * productive_factor, 2))
        rows_pre[wd] = pre_col
        rows_post[wd] = post_col
    idx = [min_to_time(t) for t in all_slots]
    return pd.DataFrame(rows_pre, index=idx), pd.DataFrame(rows_post, index=idx)


# ---------------------------
# UI – sidebar
# ---------------------------
st.title("AI Schedules Generator")
st.markdown("<p style='font-size:12px; color:Blue;'>Tool developed by <b>Gurpreet Singh</b></p>",
            unsafe_allow_html=True)

with st.sidebar:
    st.header("Settings")
    uploaded = st.file_uploader("Upload forecast CSV (one week, DD-MM-YYYY)", type=["csv"])
    aht_seconds = st.number_input("AHT (seconds)", min_value=1, max_value=3000, value=360, step=1)
    if not (0 <= aht_seconds <= 3000):
        st.error("❌ Please enter AHT between 0 and 3000 seconds.")
        st.stop()

    st.markdown("### Workload Configuration")
    selected_modalities = st.multiselect("Select Modalities", ["Voice", "Chat", "Email", "Backoffice"], default=["Voice"])
    staffing_method = st.selectbox("Staffing Method", ["Auto", "Erlang", "Workload"], index=0)

    channel_mix, channel_config = {}, {}
    total_mix = 0.0
    for channel in selected_modalities:
        pct = st.number_input(f"{channel} %", min_value=0.0, max_value=100.0,
                               value=round(100 / len(selected_modalities), 2), key=f"{channel}_pct")
        channel_mix[channel] = pct
        total_mix += pct
    if abs(total_mix - 100) > 0.01:
        st.error("Channel Mix must equal 100%")
        st.stop()

    for channel in selected_modalities:
        st.markdown(f"**{channel} SLA Settings**")
        channel_config[channel] = {
            "sla_pct": st.number_input(f"{channel} SLA %", min_value=1, max_value=100, value=80, key=f"{channel}_sla"),
            "sla_seconds": st.number_input(f"{channel} Service Time (sec)", min_value=1, max_value=86400,
                                            value=20 if channel in ["Voice", "Chat"] else 14400, key=f"{channel}_st")
        }

    chat_concurrency = 1.0
    if "Chat" in selected_modalities:
        st.markdown("### Chat Configuration")
        chat_concurrency = st.selectbox("Chat Concurrency", [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0], index=2)

    email_frt_hours = 6.0
    if "Email" in selected_modalities:
        st.markdown("### Email Configuration")
        email_frt_hours = st.number_input("Email FRT Target (hours)", min_value=0.5, max_value=72.0,
                                           value=6.0, step=0.5,
                                           help="First Response Time target for email. Used in backlog simulation.")

    needs_queue_waiting = "Voice" in selected_modalities
    if needs_queue_waiting:
        abandon_pct_target = st.number_input("Abandon Target (%)", min_value=0, max_value=50, value=5)
        patience_seconds = st.number_input("Average Patience (seconds)", min_value=1, max_value=600, value=120)
    else:
        st.info("Email/Backoffice only selected. Abandon and Patience are not required.")
        abandon_pct_target = 100
        patience_seconds = 999999

    ooo_shrinkage_pct = st.number_input("Out-of-Office Shrinkage (%)", min_value=0, max_value=100, value=15, step=1,
                                         help="Leave, training, absenteeism, attrition buffer")
    inoffice_shrinkage_pct = st.number_input("In-Office Shrinkage (%)", min_value=0, max_value=100, value=20, step=1,
                                              help="Meetings, Coaching, Huddles, AUX, System Issues")
    if not (0 <= ooo_shrinkage_pct <= 100):
        st.error("❌ Please enter value between 0 and 100")
        st.stop()

    off_policy = st.radio("Weekly Off Pattern",
                           options=["Consecutive Off Days", "Split Off Days", "Single Day Off"],
                           help="Choose whether weekly offs should be consecutive or separated")

    lunch_option = st.radio("Lunch Break Duration",
                             options=["30 minutes", "1 hour", "1 hour 30 minutes"], index=1,
                             help="Select lunch duration for agents")
    LUNCH_MIN_USER = 30 if lunch_option == "30 minutes" else (60 if lunch_option == "1 hour" else 90)

    earliest = st.time_input("Earliest shift start", value=time(0, 0))
    latest = st.time_input("Latest shift start", value=time(23, 0))
    shift_hours = st.selectbox("Agent Shift Length (Hours)",
                                [8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0], index=4)
    max_agents = st.number_input("Max agents cap", min_value=10, max_value=5000, value=800)

    st.markdown("### SLA Optimisation")
    sla_optimisation_mode = st.toggle(
        "Enable Day-Level SLA Optimisation",
        value=False,
        help=(
            "When ON, low-volume intervals can miss their individual SLA target. "
            "Staffing is trimmed so the volume-weighted DAILY SLA still meets target "
            "(within the tolerance below). This reduces total headcount."
        )
    )
    sla_tolerance_pct = st.slider(
        "Daily SLA Tolerance (% below target)",
        min_value=0.0, max_value=5.0, value=1.0, step=0.5,
        help=(
            "How many % points below the SLA target the daily weighted SLA may dip. "
            "e.g. target=80%, tolerance=1% → daily SLA must stay ≥79%. "
            "Higher tolerance = more trimming = fewer agents."
        ),
        disabled=not sla_optimisation_mode
    )

    run = st.button("Generate Roster")

if uploaded is None:
    st.info("Upload CSV to proceed.")
    st.stop()

# ---------------------------
# Read & canonicalize forecast
# ---------------------------
try:
    df_in = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

cols = {c.lower(): c for c in df_in.columns}
col_map = {}
for k in cols:
    if "date" in k: col_map["date"] = cols[k]
    if "interval" in k or "time" in k or "slot" in k: col_map.setdefault("interval", cols[k])
    if "volume" in k or "calls" in k or "forecast" in k: col_map["volume"] = cols[k]

if not all(k in col_map for k in ("date", "interval", "volume")):
    st.error("CSV must contain date, interval, volume columns.")
    st.stop()

df = df_in.rename(columns={col_map["date"]: "date", col_map["interval"]: "interval",
                             col_map["volume"]: "volume"})[["date", "interval", "volume"]].copy()
df["weekday"] = df["date"].apply(parse_weekday)
df["slot_min"] = df["interval"].apply(time_to_min)
df["slot_label"] = df["slot_min"].apply(lambda x: min_to_time(x) if x is not None else None)
df = df[df["weekday"].notnull() & df["slot_min"].notnull()]

df_agg = df.groupby(["weekday", "slot_min", "slot_label"], as_index=False)["volume"].sum()
all_slots = sorted(df_agg["slot_min"].unique())

if not all_slots:
    st.error("No valid time slots found.")
    st.stop()

rows = []
for wd in WEEKDAYS:
    for s in all_slots:
        lbl = min_to_time(s)
        v = df_agg.loc[(df_agg["weekday"] == wd) & (df_agg["slot_min"] == s), "volume"]
        rows.append({"weekday": wd, "slot_min": s, "slot_label": lbl,
                      "volume": float(v.iloc[0]) if not v.empty else 0.0})
df_week = pd.DataFrame(rows)

pivot_fore = df_week.pivot(index="slot_label", columns="weekday", values="volume").reindex(columns=WEEKDAYS).fillna(0)
daily_totals = pivot_fore.sum(axis=0)
daily_totals.name = "TOTAL"
pivot_fore_with_total = pd.concat([pivot_fore, pd.DataFrame([daily_totals])])

st.subheader("Forecast (calls / 30-min)")
st.table(pivot_fore_with_total.style.format("{:.2f}").set_properties(**{"text-align": "center"}))

# ---------------------------
# Required staff (calculations only)
# ---------------------------
aht = aht_seconds / 60.0
abandon_fraction = abandon_pct_target / 100.0
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
    channel_aht = aht / chat_concurrency if channel == "Chat" else aht

    if staffing_method == "Erlang" or (staffing_method == "Auto" and channel in ["Voice", "Chat"]):
        df_week[f"{channel}_required"] = df_week[f"{channel}_volume"].apply(
            lambda x: required_servers_for_SLA_and_abandon(
                x, channel_aht, sla_fraction, sla_seconds, abandon_fraction, patience_seconds))
    else:
        df_week[f"{channel}_required"] = df_week[f"{channel}_volume"].apply(
            lambda x: math.ceil(workload_staff_required(x, channel_aht)))

    df_week["required_raw"] += df_week[f"{channel}_required"]
    channel_summary.append({"Channel": channel, "Required Staff": int(df_week[f"{channel}_required"].sum())})

df_week["required"] = df_week["required_raw"].apply(
    lambda x: math.ceil(x / productive_factor) if x > 0 else 0)

# ---------------------------
# Day-level SLA optimisation (optional)
# ---------------------------
if sla_optimisation_mode and "Voice" in selected_modalities:
    weighted_sla_target_val = sum(
        channel_mix[ch] * channel_config[ch]["sla_pct"] for ch in selected_modalities
    ) / 10000.0
    weighted_sla_seconds_val = sum(
        channel_mix[ch] * channel_config[ch]["sla_seconds"] for ch in selected_modalities
    ) / 100.0
    df_week = optimise_daily_sla(
        df_week, all_slots, aht,
        weighted_sla_target_val,
        weighted_sla_seconds_val,
        patience_seconds,
        sla_tolerance_pct / 100.0
    )

pivot_req = df_week.pivot(index="slot_label", columns="weekday",
                           values="required").reindex(columns=WEEKDAYS).fillna(0)

# ---------------------------
# Shift templates & greedy scheduler
# ---------------------------
SHIFT_MIN = int(shift_hours * 60)
min_start = earliest.hour * 60 + earliest.minute
max_start = latest.hour * 60 + latest.minute
shift_templates = [{"start": s, "end": s + SHIFT_MIN} for s in range(min_start, max_start + 1, 30)]


def covers(start, slot):
    return start <= slot < start + SHIFT_MIN


def compute_total_remaining(reqd):
    return sum(sum(reqd[wd].values()) for wd in WEEKDAYS)


if run:
    st.subheader("Generating roster and optimizing...")

    required = {
        wd: {min_to_time(t): int(df_week.loc[(df_week["weekday"] == wd) & (df_week["slot_min"] == t), "required"].iloc[0])
             for t in all_slots}
        for wd in WEEKDAYS
    }

    off_pairs = generate_off_pairs(off_policy)
    wd_index = {d: i for i, d in enumerate(WEEKDAYS)}

    def off_mask(pair):
        m = [1] * 7
        for d in pair:
            m[wd_index[d]] = 0
        return m

    agents, aid, safety, MAX_AGENTS = [], 1, 0, int(max_agents)

    while compute_total_remaining(required) > 0 and len(agents) < MAX_AGENTS and safety < 8000:
        safety += 1
        best_need, best_wd, best_label = 0, None, None
        for wd in WEEKDAYS:
            for lbl, need in required[wd].items():
                if need > best_need:
                    best_need, best_wd, best_label = need, wd, lbl
        if best_need <= 0:
            break
        slot_min = time_to_min(best_label)
        best_tpl, best_score = None, -1
        for tpl in shift_templates:
            if not covers(tpl["start"], slot_min):
                continue
            covered = [t for t in all_slots if covers(tpl["start"], t)]
            score = sum(required[wd][min_to_time(t)] for wd in WEEKDAYS for t in covered)
            if score > best_score:
                best_score, best_tpl = score, tpl
        if best_tpl is None:
            best_tpl = {"start": slot_min - SHIFT_MIN // 2, "end": slot_min - SHIFT_MIN // 2 + SHIFT_MIN}
        best_off, best_off_score = None, -1
        for op in off_pairs:
            m = off_mask(op)
            covered = [t for t in all_slots if covers(best_tpl["start"], t)]
            sc = sum(required[wd][min_to_time(t)] for i, wd in enumerate(WEEKDAYS) if m[i] for t in covered)
            if sc > best_off_score:
                best_off_score, best_off = sc, op
        agents.append({"id": f"A{aid}", "start": int(best_tpl["start"]),
                        "end": int(best_tpl["end"]), "off": best_off})
        aid += 1
        m = off_mask(best_off)
        covered = [t for t in all_slots if covers(best_tpl["start"], t)]
        for i, wd in enumerate(WEEKDAYS):
            if m[i] == 0:
                continue
            for t in covered:
                required[wd][min_to_time(t)] = max(0, required[wd][min_to_time(t)] - 1)

    st.success(f"Initial greedy created {len(agents)} agents")
    st.markdown("Pruning redundant agents...")

    def build_schedule_counts(agent_list):
        sched = {wd: {min_to_time(t): 0 for t in all_slots} for wd in WEEKDAYS}
        for ag in agent_list:
            m = off_mask(ag["off"])
            covered = [t for t in all_slots if covers(ag["start"], t)]
            for i, wd in enumerate(WEEKDAYS):
                if m[i] == 0:
                    continue
                for t in covered:
                    sched[wd][min_to_time(t)] += 1
        return sched

    baseline_req = {
        wd: {min_to_time(r["slot_min"]): int(r["required"])
             for _, r in df_week[df_week["weekday"] == wd].iterrows()}
        for wd in WEEKDAYS
    }
    pruned = agents.copy()
    improved, loops = True, 0
    while improved and loops < 400:
        loops += 1
        improved = False
        for i in range(len(pruned) - 1, -1, -1):
            test = pruned[:i] + pruned[i + 1:]
            test_counts = build_schedule_counts(test)
            ok = all(test_counts[wd][lbl] >= reqv
                     for wd in WEEKDAYS
                     for lbl, reqv in baseline_req[wd].items())
            if ok:
                pruned.pop(i)
                improved = True
                break

    st.success(f"Pruning finished. Final agents: {len(pruned)} (was {len(agents)})")
    agents = pruned

    def shift_str(s, e):
        return f"{min_to_time(s)}–{min_to_time(e)}"

    roster = []
    for ag in agents:
        m = off_mask(ag["off"])
        row = {"Agent": ag["id"], "Shift Start": min_to_time(ag["start"]),
               "Shift End": min_to_time(ag["end"]), "Off Days": ",".join(ag["off"])}
        for i, wd in enumerate(WEEKDAYS):
            row[wd] = "OFF" if m[i] == 0 else shift_str(ag["start"], ag["end"])
        roster.append(row)

    df_roster = pd.DataFrame(roster)
    st.subheader("Roster (shift shown; OFF for off-days)")
    st.dataframe(df_roster.head(200))

    scheduled_counts = build_schedule_counts(agents)

    st.subheader("Interval Staffing – Required vs Actual (30-min)")
    actual_df = build_actual_staff_df(scheduled_counts, all_slots)
    fig_actual = plot_staffing_lines(required_df=pivot_req, actual_df=actual_df)
    st.pyplot(fig_actual)

    # ---------------------------
    # Break scheduling
    # ---------------------------
    st.subheader("Assigning breaks per day (WFM-grade optimizer)")

    LUNCH_MIN = LUNCH_MIN_USER
    MIN_GAP = 60
    TEA_IMPACT = 0.5
    req_lookup = baseline_req

    def resolve_day_and_label(wd, t):
        day_idx = WEEKDAYS.index(wd)
        if t >= 1440:
            day_idx = (day_idx + 1) % 7
        return WEEKDAYS[day_idx], min_to_time(t % 1440)

    def generate_tea_slots(slots):
        tea = []
        for t in slots:
            tea.append(t)
            tea.append(t + 15)
        return tea

    break_load = {wd: {min_to_time(t): 0.0 for t in all_slots} for wd in WEEKDAYS}
    break_deductions = {wd: {min_to_time(t): 0.0 for t in all_slots} for wd in WEEKDAYS}
    break_rows = []

    for ag in agents:
        s, e = ag["start"], ag["end"]
        shift_end = e if e > s else e + 1440
        extended_slots = all_slots if e > s else all_slots + [t + 1440 for t in all_slots]

        row = {"Agent": ag["id"], "Shift Start": min_to_time(s),
               "Shift End": min_to_time(e), "Off Days": ",".join(ag["off"])}
        m = off_mask(ag["off"])

        for i, wd in enumerate(WEEKDAYS):
            if m[i] == 0:
                row[f"{wd}_Break_1"] = ""
                row[f"{wd}_Lunch"] = ""
                row[f"{wd}_Break_2"] = ""
                continue

            slots = [t for t in extended_slots if s <= t and t + 30 <= shift_end]
            if not slots:
                row[f"{wd}_Break_1"] = ""
                row[f"{wd}_Lunch"] = ""
                row[f"{wd}_Break_2"] = ""
                continue

            tea_slots = generate_tea_slots(slots)

            # Break 1
            b1_cands = [t for t in tea_slots if s + MIN_GAP <= t <= min(s + 180, shift_end - 120)]
            if not b1_cands:
                b1_cands = [t for t in tea_slots if s + 30 <= t <= shift_end - 150]
            best_b1 = random.choice(b1_cands)

            # Lunch
            lunch_cands = [t for t in slots if t >= best_b1 + MIN_GAP and t + 30 in slots and t <= shift_end - 90]
            if not lunch_cands:
                lunch_cands = [t for t in slots if best_b1 + 45 <= t <= shift_end - 60]
            best_lunch = random.choice(lunch_cands)
            lunch_end = best_lunch + LUNCH_MIN

            # Break 2
            b2_cands = [t for t in tea_slots if lunch_end + MIN_GAP <= t <= shift_end - 15]
            if not b2_cands:
                b2_cands = [t for t in tea_slots if lunch_end + 30 <= t <= shift_end - 15]
            if not b2_cands:
                b2_cands = [t for t in tea_slots if s + 30 <= t <= shift_end - 15]
            best_b2 = shift_end - 15 if not b2_cands else random.choice(b2_cands)

            row[f"{wd}_Break_1"] = f"{min_to_time(best_b1 % 1440)}-{min_to_time((best_b1 + 15) % 1440)}"
            row[f"{wd}_Lunch"] = f"{min_to_time(best_lunch % 1440)}-{min_to_time((best_lunch + LUNCH_MIN) % 1440)}"
            d2, _ = resolve_day_and_label(wd, best_b2)
            row[f"{d2}_Break_2"] = f"{min_to_time(best_b2 % 1440)}-{min_to_time((best_b2 + 15) % 1440)}"

            d1, lbl1 = resolve_day_and_label(wd, best_b1)
            dl, lbll = resolve_day_and_label(wd, best_lunch)
            break_load[d1][lbl1] = break_load[d1].get(lbl1, 0) + TEA_IMPACT
            break_load[dl][lbll] = break_load[dl].get(lbll, 0) + 1.0
            break_load[dl][min_to_time((best_lunch + 30) % 1440)] = (
                break_load[dl].get(min_to_time((best_lunch + 30) % 1440), 0) + 1.0)

            b1_lbl = min_to_time((best_b1 // 30) * 30 % 1440)
            if b1_lbl in break_deductions[d1]:
                break_deductions[d1][b1_lbl] = break_deductions[d1].get(b1_lbl, 0) + 0.5
            for lmin in range(0, LUNCH_MIN, 30):
                lt = best_lunch + lmin
                dl2, _ = resolve_day_and_label(wd, lt)
                lt_lbl = min_to_time((lt // 30) * 30 % 1440)
                if lt_lbl in break_deductions[dl2]:
                    break_deductions[dl2][lt_lbl] = break_deductions[dl2].get(lt_lbl, 0) + 1.0
            b2_lbl = min_to_time((best_b2 // 30) * 30 % 1440)
            if b2_lbl in break_deductions[d2]:
                break_deductions[d2][b2_lbl] = break_deductions[d2].get(b2_lbl, 0) + 0.5

        break_rows.append(row)

    df_breaks = pd.DataFrame(break_rows)
    st.dataframe(df_breaks.head(200))

    sched_df = pd.DataFrame(
        {wd: [scheduled_counts[wd].get(min_to_time(t), 0) for t in all_slots] for wd in WEEKDAYS},
        index=[min_to_time(t) for t in all_slots])
    req_df = pd.DataFrame(
        {wd: [int(df_week.loc[(df_week["weekday"] == wd) & (df_week["slot_min"] == t), "required"].iloc[0])
              for t in all_slots] for wd in WEEKDAYS},
        index=[min_to_time(t) for t in all_slots])
    diff_df = sched_df - req_df
    st.subheader("Coverage after breaks (scheduled - required)")
    st.dataframe(diff_df.head(20))

    # Open numbers
    st.subheader("📋 Schedule Open Numbers – After Breaks, Before Shrinkage")
    df_open_pre, df_open_post = build_interval_open_numbers(
        scheduled_counts, all_slots, break_deductions, inoffice_shrinkage_pct, ooo_shrinkage_pct)
    st.caption("FTE-equivalent seats available on the floor after removing break time. No shrinkage applied.")
    st.dataframe(df_open_pre)

    st.subheader("📋 Schedule Open Numbers – After Breaks & Shrinkage")
    st.caption(f"In-Office Shrinkage {inoffice_shrinkage_pct}% + OOO Shrinkage {ooo_shrinkage_pct}% "
               f"= Productive Factor {productive_factor * 100:.1f}%")
    st.dataframe(df_open_post)

    # Email backlog
    email_pct_val = channel_mix.get("Email", 0)
    if email_pct_val > 0:
        st.subheader("📧 Email Backlog Simulation (Interval Level)")
        st.caption(f"FRT target = {email_frt_hours}h (reference only) | "
                   f"Email capacity = idle agent-mins ÷ AHT, where idle = (Scheduled FTE × 30) − non-email workload | "
                   f"Email volume = {email_pct_val:.1f}% of total forecast")
        df_backlog = simulate_email_backlog(df_week, scheduled_counts, aht, email_frt_hours, all_slots, email_pct_val)
        backlog_summary = (df_backlog.groupby("Day")
                           .agg(Email_Forecast=("Email Forecast", "sum"),
                                Tickets_Handled=("Tickets Handled", "sum"),
                                Closing_Backlog=("Backlog (Carryover)", "last"),
                                Peak_Backlog=("Backlog (Carryover)", "max"))
                           .reindex(WEEKDAYS).reset_index())
        backlog_summary.columns = ["Day", "Email Forecast", "Tickets Handled", "Closing Backlog", "Peak Backlog"]
        st.dataframe(backlog_summary.style.format(
            {"Email Forecast": "{:.0f}", "Tickets Handled": "{:.0f}",
             "Closing Backlog": "{:.0f}", "Peak Backlog": "{:.0f}"}))
        st.markdown("**Full interval-level backlog detail:**")
        st.dataframe(df_backlog)
    else:
        df_backlog = pd.DataFrame()

    # ---------------------------
    # Daily projections — Occupancy on Scheduled FTEs (FIXED)
    # ---------------------------
    st.subheader("Daily projections — Occupancy on Scheduled FTEs")
    st.caption("Occupancy = (Volume × AHT_min) / (Scheduled_FTE × 30 min × productive factor) — "
               "measured on post-shrinkage available time, includes email worked during idle | "
               "SLA = raw Erlang-A estimate (uncapped) | "
               "Avg Concurrent FTE = average agents logged in per staffed interval "
               "(NOT total roster headcount, which is the larger 'Final agents' number above)")

    weighted_sla_seconds = sum(channel_mix[ch] * channel_config[ch]["sla_seconds"]
                                for ch in selected_modalities) / 100.0
    mu = 1.0 / aht
    theta = 1.0 / (patience_seconds / 60.0) if patience_seconds > 0 else 0.0
    t_sla = weighted_sla_seconds / 60.0

    proj_rows = []
    for wd in WEEKDAYS:
        tot_calls = sla_acc = abn_acc = occ_acc = 0.0
        for t in all_slots:
            lbl = min_to_time(t)
            calls = float(df_week.loc[(df_week["weekday"] == wd) & (df_week["slot_min"] == t), "volume"].iloc[0])
            scheduled = scheduled_counts[wd].get(lbl, 0)
            if scheduled == 0:
                sla_it, abn_it, occ_it = 0.0, (1.0 if calls > 0 else 0.0), 0.0
            else:
                a = (calls / 30.0) / mu
                _, _, p_abandon_any, sla_est = erlang_a_estimates(a, scheduled, mu, theta, t_sla)
                # Raw Erlang-A service level — no clamping to the target band.
                sla_it = max(0.0, min(1.0, sla_est))
                abn_it = p_abandon_any
                # Occupancy on PRODUCTIVE (post-shrinkage) logged-in time. The volume
                # already blends live + email, so idle time absorbed by email is
                # reflected here; dividing by available time (not raw bodies) gives the
                # true occupancy of agents who are actually on the floor.
                productive_capacity = scheduled * 30.0 * productive_factor
                occ_it = min((calls * aht) / productive_capacity, 1.0) if productive_capacity > 0 else 0.0
            tot_calls += calls
            sla_acc += sla_it * calls
            abn_acc += abn_it * calls
            occ_acc += occ_it * calls

        # ✅ FIX: average FTE only over intervals where agents are actually scheduled
        staffed_slots = [t for t in all_slots if scheduled_counts[wd].get(min_to_time(t), 0) > 0]
        fte_avg = round(
            sum(scheduled_counts[wd].get(min_to_time(t), 0) for t in staffed_slots) / len(staffed_slots), 2
        ) if staffed_slots else 0

        if tot_calls > 0:
            proj_rows.append({
                "Day": wd,
                "Total Calls": int(round(tot_calls)),
                "Avg Concurrent FTE": fte_avg,
                "Projected SLA %": round(sla_acc / tot_calls * 100, 2),
                "Projected Abandon %": round(abn_acc / tot_calls * 100, 2),
                "Avg Occupancy % (on Scheduled FTE)": round(occ_acc / tot_calls * 100, 2),
            })
        else:
            proj_rows.append({
                "Day": wd, "Total Calls": 0, "Avg Concurrent FTE": 0,
                "Projected SLA %": 100.0, "Projected Abandon %": 0.0,
                "Avg Occupancy % (on Scheduled FTE)": 0.0,
            })

    df_proj = pd.DataFrame(proj_rows)
    total_calls_sum = df_proj["Total Calls"].sum()

    # ✅ FIX: TOTAL row uses weighted average across all staffed intervals across the week
    all_staffed_vals = [
        scheduled_counts[wd].get(min_to_time(t), 0)
        for wd in WEEKDAYS for t in all_slots
        if scheduled_counts[wd].get(min_to_time(t), 0) > 0
    ]
    weekly_fte_avg = round(sum(all_staffed_vals) / len(all_staffed_vals), 2) if all_staffed_vals else 0

    weekly_total = {
        "Day": "TOTAL",
        "Total Calls": total_calls_sum,
        "Avg Concurrent FTE": weekly_fte_avg,
        "Projected SLA %": round(
            (df_proj["Projected SLA %"] * df_proj["Total Calls"]).sum() / total_calls_sum, 2),
        "Projected Abandon %": round(
            (df_proj["Projected Abandon %"] * df_proj["Total Calls"]).sum() / total_calls_sum, 2),
        "Avg Occupancy % (on Scheduled FTE)": round(
            (df_proj["Avg Occupancy % (on Scheduled FTE)"] * df_proj["Total Calls"]).sum() / total_calls_sum, 2),
    }
    df_proj = pd.concat([df_proj, pd.DataFrame([weekly_total])], ignore_index=True)
    st.dataframe(df_proj.style.format({
        "Projected SLA %": "{:.2f}%",
        "Projected Abandon %": "{:.2f}%",
        "Avg Occupancy % (on Scheduled FTE)": "{:.2f}%",
    }))

    # ---------------------------
    # Export
    # ---------------------------
    def export_all(roster_df, breaks_df, proj_df, fore_df, req_df,
                   open_pre_df, open_post_df, backlog_df):
        out = BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            roster_df.to_excel(writer, sheet_name="Roster", index=False)
            breaks_df.to_excel(writer, sheet_name="Breaks", index=False)
            proj_df.to_excel(writer, sheet_name="Projections", index=False)
            fore_df.to_excel(writer, sheet_name="Forecast")
            req_df.to_excel(writer, sheet_name="Required")
            open_pre_df.to_excel(writer, sheet_name="Open_After_Breaks_PreShrink")
            open_post_df.to_excel(writer, sheet_name="Open_After_Breaks_PostShrink")
            if not backlog_df.empty:
                backlog_df.to_excel(writer, sheet_name="Email_Backlog", index=False)
        return out.getvalue()

    excel_data = export_all(df_roster, df_breaks, df_proj, pivot_fore, pivot_req,
                             df_open_pre, df_open_post, df_backlog)
    b64 = base64.b64encode(excel_data).decode()
    st.markdown(
        f'<a href="data:application/octet-stream;base64,{b64}" download="roster_output.xlsx">'
        f'📥 Download full output (Excel)</a>',
        unsafe_allow_html=True
    )
    st.download_button("Download roster CSV", data=df_roster.to_csv(index=False).encode(), file_name="roster.csv")
    st.download_button("Download breaks CSV", data=df_breaks.to_csv(index=False).encode(), file_name="breaks.csv")
    st.download_button("Download projections CSV", data=df_proj.to_csv(index=False).encode(), file_name="projections.csv")
    st.download_button("Download open numbers (pre-shrink) CSV", data=df_open_pre.to_csv().encode(), file_name="open_pre_shrink.csv")
    st.download_button("Download open numbers (post-shrink) CSV", data=df_open_post.to_csv().encode(), file_name="open_post_shrink.csv")
    if not df_backlog.empty:
        st.download_button("Download email backlog CSV", data=df_backlog.to_csv(index=False).encode(), file_name="email_backlog.csv")

    st.success("Done.")
else:
    st.info("Adjust settings and click Generate Roster.")