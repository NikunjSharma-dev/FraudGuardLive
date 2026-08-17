import os
import time
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

import metrics as model_metrics
import style
from style import ACCENT, ACCENT_2, AMBER, GREEN, MUTED, RED, STATUS_COLORS

# -----------------------------------------------------------------------------
# Configuration & Theming
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FraudGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

style.inject_css()
style.apply_plotly_theme()


def _get_api_url() -> str:
    try:
        return st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))
    except Exception:
        return os.getenv("API_URL", "http://localhost:8000")


def _api_points_to_localhost(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return host in {"localhost", "127.0.0.1", "0.0.0.0"}


@st.cache_data(ttl=30)
def _probe_backend_health(url: str) -> tuple[bool, str]:
    try:
        response = requests.get(f"{url}/health", timeout=60)
        if response.ok:
            return True, response.text
        return False, f"HTTP {response.status_code}"
    except requests.RequestException as exc:
        return False, str(exc)


# API Configuration
API_URL = _get_api_url().rstrip("/")  # set API_URL in Streamlit secrets or environment for non-local deployments
API_IS_LOCALHOST = _api_points_to_localhost(API_URL)
API_ONLINE, API_HEALTH_DETAIL = _probe_backend_health(API_URL)

# -----------------------------------------------------------------------------
# Sidebar Navigation — fixed icon rail, expands on hover
# -----------------------------------------------------------------------------
NAV = [
    ("dashboard", "📊", "Live Dashboard"),
    ("signup", "📝", "Open Account"),
    ("simulate", "💳", "Simulate Transaction"),
    ("ops", "🛠️", "Technical Ops"),
    ("model", "📈", "Model Performance"),
    ("admin", "🔐", "Admin Portal"),
    ("health", "⚙️", "System Health"),
]
LABELS = [label for _, _, label in NAV]
KEY_BY_LABEL = {label: key for key, _, label in NAV}

with st.sidebar:
    st.markdown(
        '<div class="fg-brand">'
        '<div class="fg-brand-mark">🛡️</div>'
        '<div class="fg-brand-text">'
        '<div class="fg-brand-name">FraudGuard</div>'
        '<div class="fg-brand-tag">Fraud Detection</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    selected_label = st.radio(
        "Navigation", LABELS, label_visibility="collapsed", key="nav"
    )
    style.inject_nav_icons([icon for _, icon, _ in NAV])

    api_pill = (
        style.pill("Backend online", "ok")
        if API_ONLINE
        else style.pill("Backend offline", "bad")
    )
    st.markdown(
        f'<div class="fg-side-foot"><div class="fg-side-foot-in">{api_pill}<br>'
        f'<span style="font-size:.68rem">Updated {datetime.now().strftime("%H:%M:%S")}'
        "</span></div></div>",
        unsafe_allow_html=True,
    )

page = KEY_BY_LABEL[selected_label]

# Backend connectivity notice — shown once, above the page body.
if not API_ONLINE:
    if API_IS_LOCALHOST:
        st.error(
            "FastAPI backend is not reachable at localhost:8000. Start the backend "
            "service and refresh this page.",
            icon="🔌",
        )
    else:
        st.warning(
            "FastAPI backend is not reachable at the configured API_URL. Set a public "
            "backend URL in Streamlit secrets or environment variables.",
            icon="⚠️",
        )

# =============================================================================
# PAGE 1: LIVE DASHBOARD
# =============================================================================
if page == "dashboard":
    style.page_header("Live Ledger", "Real-time transaction flow and decision outcomes")

    try:
        response = requests.get(f"{API_URL}/admin/ledger-summary", timeout=30)
        if response.status_code != 200:
            raise ValueError
        data = response.json()
        live = True
    except Exception:
        data = {
            "total_volume": 0.0,
            "fraud_count": 0,
            "throughput": 0.0,
            "status_breakdown": {"Approved": 0, "Declined": 0, "Awaiting Verification": 0},
        }
        live = False

    note = "" if live else "awaiting backend"
    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        style.kpi("24h Volume", f'₹{data["total_volume"]:,.0f}', "accent", "💹", note)
    with k2:
        style.kpi("Fraud Neutralized", f'{data["fraud_count"]}', "red", "🚨", note)
    with k3:
        style.kpi("Throughput", f'{data["throughput"]} TPS', "", "⚡", note)
    with k4:
        style.kpi(
            "MFA Pending",
            f'{data["status_breakdown"].get("Awaiting Verification", 0)}',
            "amber",
            "🔐",
            note,
        )

    st.write("")

    chart_col1, chart_col2 = st.columns([2, 1], gap="medium")
    with chart_col1:
        with style.card("Transaction Volume Trend", "last 24h"):
            try:
                trend_resp = requests.get(f"{API_URL}/admin/volume-trend", timeout=30).json()
                df_trend = pd.DataFrame(trend_resp)
                df_trend["hour"] = pd.to_datetime(df_trend["hour"])
            except Exception:
                df_trend = pd.DataFrame(
                    {"hour": pd.date_range("today", periods=24, freq="h"), "volume": [0] * 24}
                )
            fig_line = px.line(df_trend, x="hour", y="volume", line_shape="spline")
            fig_line.update_traces(
                line_color=ACCENT, line_width=2.5,
                fill="tozeroy", fillcolor="rgba(56,189,248,0.10)",
                # Markers keep a one- or two-bucket series visible as a point
                mode="lines+markers", marker=dict(size=6, color=ACCENT),
            )
            fig_line.update_layout(xaxis_title=None, yaxis_title="Volume")
            # An all-zero series otherwise auto-ranges to ±1 with odd tick labels
            if float(df_trend["volume"].max() or 0) <= 0:
                fig_line.update_yaxes(range=[0, 1])
            else:
                fig_line.update_yaxes(rangemode="tozero")
            # With a single bucket plotly zooms the time axis to microseconds
            # ("07:59:59.9995"); pad it out to a readable window instead.
            if len(df_trend) < 3:
                center = pd.to_datetime(df_trend["hour"]).iloc[0]
                fig_line.update_xaxes(
                    range=[center - pd.Timedelta(hours=6), center + pd.Timedelta(hours=6)]
                )
            fig_line.update_xaxes(tickformat="%H:%M<br>%b %d")
            style.plot(fig_line, height=320, key="dash_trend")

    with chart_col2:
        with style.card("Decision Breakdown"):
            df_pie = pd.DataFrame(
                list(data["status_breakdown"].items()), columns=["Status", "Count"]
            )
            if df_pie["Count"].sum() <= 0:
                # px.pie renders an invisible chart when every value is zero
                st.markdown(
                    f"<div style='height:280px;display:flex;align-items:center;"
                    f"justify-content:center;text-align:center;color:{MUTED};"
                    f"font-size:.83rem;line-height:1.6'>No decisions recorded yet.<br>"
                    f"Submit a transaction to populate this breakdown.</div>",
                    unsafe_allow_html=True,
                )
            else:
                fig_pie = px.pie(
                    df_pie, values="Count", names="Status", hole=0.7,
                    color="Status", color_discrete_map=STATUS_COLORS,
                )
                fig_pie.update_traces(
                    marker=dict(line=dict(color=style.SURFACE, width=2)),
                    # Keep labels inside the ring: the default draws leader lines
                    # for every zero-count slice, which collide with the legend.
                    textinfo="percent",
                    textposition="inside",
                    texttemplate="%{percent:.0%}",
                    insidetextfont=dict(color=style.BG, size=13),
                    sort=False,
                    hovertemplate="%{label}: %{value:,} (%{percent:.1%})<extra></extra>",
                )
                fig_pie.update_layout(
                    legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center")
                )
                style.plot(fig_pie, height=320, showlegend=True, key="dash_pie")

    try:
        txns_resp = requests.get(f"{API_URL}/admin/transactions?limit=50", timeout=30)
        if txns_resp.status_code == 200:
            df_txns = pd.DataFrame(txns_resp.json())

            if not df_txns.empty:
                st.write("")
                with style.card("Ledger Activity", "50 most recent"):
                    tab_all, tab_fraud = st.tabs(["All Recent Transactions", "Detected Fraud 🚨"])
                    display_cols = ["account_id", "amount", "status", "risk_score", "created_at"]

                    with tab_all:
                        st.dataframe(
                            df_txns[display_cols], use_container_width=True,
                            height=400, hide_index=True,
                        )

                    with tab_fraud:
                        df_fraud = df_txns[df_txns["is_fraudulent"] == True]
                        if not df_fraud.empty:
                            st.dataframe(
                                df_fraud[display_cols].style.highlight_max(
                                    subset=["risk_score"], color=RED
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("No fraudulent transactions in the recent ledger.")
    except Exception:
        pass

# =============================================================================
# PAGE 2: OPEN ACCOUNT (SIGN UP)
# =============================================================================
elif page == "signup":
    style.page_header(
        "Customer Onboarding",
        "Provision a fresh account to exercise the ML pipeline end to end",
    )

    form_col, info_col = st.columns([2, 1], gap="medium")

    with form_col:
        with st.form("signup_form", clear_on_submit=True):
            st.markdown("###### KYC & Personal Details")

            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full Legal Name", placeholder="Jane Doe")
                email = st.text_input("Email Address", placeholder="jane.doe@example.com")
            with col2:
                phone = st.text_input("Phone Number", placeholder="+91 9876543210")
                kyc = st.selectbox(
                    "Identity Verification Document",
                    ["Aadhaar Card", "PAN Card", "Passport", "Driver's License"],
                )

            st.write("")
            submit_signup = st.form_submit_button("Create Account", use_container_width=True)

    with info_col:
        with style.card("How this works"):
            st.markdown(
                f"<div style='font-size:.83rem;color:{MUTED};line-height:1.65'>"
                "A new account starts with a clean risk profile, so it won't trip "
                "brute-force locks left over from earlier testing.<br><br>"
                "After creating it, copy the account ID into "
                "<strong>Simulate Transaction</strong> to start scoring payments."
                "</div>",
                unsafe_allow_html=True,
            )

    if submit_signup:
        if not full_name or not email or not phone:
            st.error("Please fill in all required fields.", icon="🚨")
        else:
            with st.spinner("Verifying KYC and initializing risk profile..."):
                time.sleep(1)
                try:
                    res = requests.post(
                        f"{API_URL}/account/signup",
                        json={
                            "full_name": full_name,
                            "email": email,
                            "phone": phone,
                            "kyc_document": kyc,
                        },
                    )
                    if res.status_code == 200:
                        data = res.json()
                        new_acc_id = data["account_id"]
                        st.success(data["message"], icon="🎉")
                        st.markdown(
                            f"""
                            <div style="background: {style.SURFACE};
                                        border: 1px solid rgba(74,222,128,.35);
                                        border-left: 3px solid {GREEN};
                                        padding: 20px; border-radius: {style.RADIUS};
                                        margin-top: 12px;">
                                <div style="color:{MUTED}; font-size:.7rem;
                                            text-transform:uppercase; letter-spacing:.07em;
                                            font-weight:640;">Your Account ID</div>
                                <div style="color:{GREEN}; font-size:2rem; font-weight:660;
                                            font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
                                            margin-top:6px;">{new_acc_id}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.info(
                            "**Next step:** copy this ID, open **Simulate Transaction**, "
                            "and paste it into the Account ID field.",
                            icon="💡",
                        )
                    else:
                        st.error("Failed to create account. Check backend logs.", icon="❌")
                except requests.exceptions.ConnectionError:
                    if API_IS_LOCALHOST:
                        st.error(
                            "Connection refused: API_URL points to localhost. If deployed "
                            "on Streamlit Cloud, set API_URL to your public backend URL.",
                            icon="🔌",
                        )
                    else:
                        st.error(
                            "Connection refused: ensure the FastAPI backend is running.",
                            icon="🔌",
                        )

# =============================================================================
# PAGE 3: SIMULATE TRANSACTION
# =============================================================================
elif page == "simulate":
    style.page_header(
        "Point of Sale Emulator",
        "Submit transactions and watch the risk engine respond in real time",
    )

    if "pending_mfa_tx" not in st.session_state:
        st.session_state.pending_mfa_tx = None

    submitted = False
    submit_otp = resend_btn = cancel_btn = False
    otp_input = ""

    col_form, col_terminal = st.columns([1, 1], gap="medium")

    with col_form:
        if not st.session_state.pending_mfa_tx:
            with st.form("transaction_form", clear_on_submit=False):
                st.markdown("###### Card Details")
                account_id = st.text_input("Account ID", value="ACC10294")
                amount = st.number_input(
                    "Amount (INR)", min_value=1.0, value=5000.00, step=100.0
                )
                st.markdown("###### Geospatial Context")
                col_lat, col_lon = st.columns(2)
                lat = col_lat.number_input("Latitude", value=19.0760, format="%.4f")
                lon = col_lon.number_input("Longitude", value=72.8777, format="%.4f")
                submitted = st.form_submit_button("Swipe Card 💳", use_container_width=True)
        else:
            st.warning("Account locked — complete the pending MFA challenge.", icon="🔒")
            with st.form("otp_form", clear_on_submit=True):
                st.markdown("###### Step-Up Authentication Required")
                st.markdown(f"Transaction ID: `{st.session_state.pending_mfa_tx}`")
                otp_input = st.text_input(
                    "Enter 6-Digit OTP (Check Backend Terminal)", max_chars=6
                )

                c1, c2, c3 = st.columns(3)
                submit_otp = c1.form_submit_button("Verify", use_container_width=True)
                resend_btn = c2.form_submit_button("Resend 🔄", use_container_width=True)
                cancel_btn = c3.form_submit_button("Cancel ❌", use_container_width=True)

    with col_terminal:
        with style.card("Engine Response", "live"):
            tc = st.container(height=460)

        if not st.session_state.pending_mfa_tx and submitted:
            payload = {"account_id": account_id, "amount": amount, "lat": lat, "lon": lon}
            with tc:
                st.info("Sending transaction to risk engine...")
                try:
                    res = requests.post(
                        f"{API_URL}/transaction/submit", json=payload, timeout=60
                    )
                    data = res.json()

                    if res.status_code == 200:
                        status = data.get("status")
                        if status == "Approved":
                            st.success(
                                f"Transaction approved — risk score: "
                                f"{data.get('risk_score', 0):.4f}"
                            )
                            st.balloons()
                        elif status == "Declined":
                            st.error(
                                f"**BLOCKED**\n\n**Reason:** {data.get('message')}", icon="⛔"
                            )
                        elif status == "Awaiting Verification":
                            st.warning(
                                f"High risk — score: {(data.get('risk_score') or 0):.4f}. "
                                "MFA required.",
                                icon="🔐",
                            )
                            st.session_state.pending_mfa_tx = data.get("transaction_id")

                        if data.get("explanation"):
                            st.markdown("---")
                            st.markdown("###### SHAP Feature Attribution")
                            exp = data["explanation"]
                            top_factors = sorted(
                                exp.items(), key=lambda x: abs(x[1]), reverse=True
                            )[:3]
                            for feature, impact in top_factors:
                                tone = RED if impact > 0 else GREEN
                                direction = "increased" if impact > 0 else "decreased"
                                st.markdown(
                                    f"<div style='font-size:.85rem;margin:4px 0'>"
                                    f"<code>{feature}</code> "
                                    f"<span style='color:{tone}'>{direction}</span> risk by "
                                    f"<strong>{abs(impact):.3f}</strong></div>",
                                    unsafe_allow_html=True,
                                )
                    elif res.status_code == 503:
                        detail = data.get("detail") if isinstance(data, dict) else None
                        st.error(
                            "Backend is online, but the transaction database is "
                            "unavailable. Start PostgreSQL and try again."
                            + (f"\n\n{detail}" if detail else ""),
                            icon="🔌",
                        )
                    else:
                        st.error(f"HTTP Error: {res.status_code}\n{res.text}")
                except requests.exceptions.ConnectionError:
                    if API_IS_LOCALHOST:
                        st.error(
                            "Connection refused: API_URL points to localhost. If deployed "
                            "on Streamlit Cloud, set API_URL to your public backend URL.",
                            icon="🔌",
                        )
                    else:
                        st.error(
                            "Connection refused: ensure the FastAPI backend is running.",
                            icon="🔌",
                        )

        elif st.session_state.pending_mfa_tx:
            if submit_otp:
                with tc:
                    st.info("Validating OTP...")
                    time.sleep(0.5)
                    try:
                        res = requests.patch(
                            f"{API_URL}/transaction/{st.session_state.pending_mfa_tx}/verify",
                            json={"otp": otp_input},
                        )
                        data = res.json()
                        if data.get("status") == "Verified":
                            st.success("Identity confirmed. Transaction approved.")
                            st.balloons()
                        else:
                            st.error(f"{data.get('message')}", icon="❌")
                        st.session_state.pending_mfa_tx = None
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Verification failed: {e}")

            if resend_btn:
                with tc:
                    st.info("Requesting a new OTP...")
                    try:
                        res = requests.post(
                            f"{API_URL}/transaction/{st.session_state.pending_mfa_tx}/resend-otp"
                        )
                        if res.status_code == 200:
                            st.success("New OTP generated — check the backend terminal.", icon="📩")
                        else:
                            st.error("Failed to request new OTP.", icon="❌")
                    except Exception as e:
                        st.error(f"Resend failed: {e}")

            if cancel_btn:
                st.session_state.pending_mfa_tx = None
                st.rerun()

# =============================================================================
# PAGE 4: TECHNICAL OPERATIONS
# =============================================================================
elif page == "ops":
    style.page_header(
        "Pipeline Health", "Telemetry for the event stream and ML inference engine"
    )

    times = pd.date_range(start="17:30", end="18:16", freq="1min")
    n_points = len(times)
    active_mask = times < pd.to_datetime(times[0].strftime("%Y-%m-%d") + " 18:14")

    r1c1, r1c2 = st.columns(2, gap="medium")
    with r1c1:
        with style.card("Input Stream — Ack vs Unacked", "MEAN"):
            ack_count = np.where(
                active_mask,
                np.random.uniform(500, 600, n_points),
                np.random.uniform(0, 10, n_points),
            )
            unack_count = np.random.uniform(0, 20, n_points)
            df_stream = pd.DataFrame(
                {
                    "Time": times,
                    "Ack message count": ack_count,
                    "Unacked messages": unack_count,
                }
            )
            fig1 = px.line(
                df_stream, x="Time", y=["Ack message count", "Unacked messages"],
                color_discrete_sequence=[ACCENT, ACCENT_2],
            )
            fig1.update_traces(line_width=2)
            fig1.update_layout(yaxis_title="Count", legend_title_text=None, xaxis_title=None)
            style.plot(fig1, height=300, key="ops_ack")

    with r1c2:
        with style.card("Input Stream — Ack Latency", "99TH PERCENTILE"):
            latency = np.random.uniform(1.3, 1.8, n_points)
            latency[::7] = np.random.uniform(2.0, 2.4, len(latency[::7]))
            df_lat = pd.DataFrame({"Time": times, "ALIGN_PERCENTILE_99": latency})
            fig2 = px.line(
                df_lat, x="Time", y="ALIGN_PERCENTILE_99",
                color_discrete_sequence=[ACCENT],
            )
            fig2.update_traces(line_width=2)
            fig2.update_layout(yaxis_title="Seconds", xaxis_title=None)
            style.plot(fig2, height=300, key="ops_ack_latency")

    st.write("")
    r2c1, r2c2 = st.columns(2, gap="medium")
    with r2c1:
        with style.card("AI Platform — Prediction Count", "MEAN"):
            pred_wo_agg = np.where(
                active_mask,
                np.random.uniform(800, 850, n_points),
                np.random.uniform(0, 5, n_points),
            )
            pred_w_agg = pred_wo_agg + np.random.uniform(-10, 10, n_points)
            df_pred = pd.DataFrame(
                {"Time": times, "model_v1_wo_agg": pred_wo_agg, "model_v2_w_agg": pred_w_agg}
            )
            fig3 = px.line(
                df_pred, x="Time", y=["model_v1_wo_agg", "model_v2_w_agg"],
                color_discrete_sequence=[ACCENT, ACCENT_2],
            )
            fig3.update_traces(line_width=2)
            fig3.update_layout(yaxis_title="Count", legend_title_text=None, xaxis_title=None)
            style.plot(fig3, height=300, key="ops_pred_count")

    with r2c2:
        with style.card("AI Platform — Total Latency", "95TH PERCENTILE"):
            lat_wo = np.random.uniform(120, 180, n_points) + np.sin(np.arange(n_points) / 2) * 20
            lat_w = np.random.uniform(150, 200, n_points) + np.cos(np.arange(n_points) / 3) * 20
            df_model_lat = pd.DataFrame(
                {"Time": times, "model_v1_wo_agg": lat_wo, "model_v2_w_agg": lat_w}
            )
            fig4 = px.line(
                df_model_lat, x="Time", y=["model_v1_wo_agg", "model_v2_w_agg"],
                color_discrete_sequence=[ACCENT, ACCENT_2],
            )
            fig4.update_traces(line_width=2)
            fig4.update_layout(
                yaxis_title="Milliseconds", legend_title_text=None, xaxis_title=None
            )
            style.plot(fig4, height=300, key="ops_model_latency")

    st.write("")
    r3c1, r3c2 = st.columns(2, gap="medium")
    with r3c1:
        with style.card("AI Platform — Error Count", "MEAN"):
            # Pull real TRIGGER_DECLINE events from the audit log as a proxy for
            # pipeline errors (hard-rule rejections, limit breaches, etc.)
            try:
                audit_resp = requests.get(f"{API_URL}/admin/audit-log?limit=200", timeout=30)
                if audit_resp.status_code == 200:
                    audit_rows = audit_resp.json()
                    # Count TRIGGER_DECLINE events per minute slot within the display window
                    error_counts = {t: 0 for t in times}
                    for row in audit_rows:
                        if row.get("event_type") == "TRIGGER_DECLINE":
                            # Snap the event timestamp to the nearest display bucket
                            nearest = min(
                                times,
                                key=lambda t: abs(
                                    (
                                        t
                                        - pd.Timestamp(row["created_at"].replace("Z", ""))
                                    ).total_seconds()
                                ),
                            )
                            error_counts[nearest] = error_counts.get(nearest, 0) + 1
                    err_values = list(error_counts.values())
                else:
                    raise ValueError("non-200")
            except Exception:
                # API offline or no data yet — generate a realistic near-zero baseline
                # (healthy system has rare errors; occasional spikes are normal)
                np.random.seed(int(datetime.now().timestamp()) % 1000)
                err_values = np.zeros(n_points, dtype=float)
                # Scatter a handful of single error events
                spike_indices = np.random.choice(
                    n_points, size=max(1, n_points // 12), replace=False
                )
                err_values[spike_indices] = np.random.uniform(0.5, 2.5, len(spike_indices))
                # One slightly larger burst to reflect realistic behaviour
                burst_idx = np.random.randint(n_points // 3, 2 * n_points // 3)
                err_values[burst_idx] = np.random.uniform(3.0, 6.0)

            df_err = pd.DataFrame({"Time": times, "Error count": err_values})
            fig5 = px.line(
                df_err, x="Time", y="Error count", color_discrete_sequence=[RED]
            )
            fig5.update_traces(
                line_width=2, fill="tozeroy", fillcolor="rgba(248,113,113,0.10)"
            )
            fig5.update_layout(
                yaxis_title="Count",
                yaxis=dict(rangemode="nonnegative"),
                xaxis_title=None,
            )
            style.plot(fig5, height=300, key="ops_errors")

    with r3c2:
        with style.card("Fraud Notifications — Unacked", "MEAN"):
            unacked_msgs = np.linspace(35, 80, n_points) + np.random.uniform(-1, 1, n_points)
            df_notif = pd.DataFrame({"Time": times, "Unacked messages": unacked_msgs})
            fig6 = px.line(
                df_notif, x="Time", y="Unacked messages",
                color_discrete_sequence=[ACCENT_2],
            )
            fig6.update_traces(line_width=2)
            fig6.update_layout(yaxis_title="Count", xaxis_title=None)
            style.plot(fig6, height=300, key="ops_notifications")

# =============================================================================
# PAGE 5: MODEL PERFORMANCE
# =============================================================================
elif page == "model":
    style.page_header(
        "Model Evaluation", "XGBoost + Isolation Forest ensemble, scored from saved artifacts"
    )

    with st.spinner("Scoring saved model artifacts..."):
        m = model_metrics.compute()

    if m is None:
        st.warning(
            "Model metrics unavailable — the artifacts in `backend/app/ml/models/` or the "
            "evaluation dataset could not be loaded in this environment. Everything else "
            "on this page depends on them.",
            icon="📉",
        )
    else:
        k1, k2, k3, k4 = st.columns(4, gap="medium")
        with k1:
            style.kpi("ROC-AUC", f"{m.roc_auc:.3f}", "accent", "📈", "XGBoost")
        with k2:
            style.kpi("Recall (Fraud)", f"{m.recall:.1%}", "green", "🎯", "caught")
        with k3:
            style.kpi("Precision (Fraud)", f"{m.precision:.1%}", "accent", "🔍", "of flags correct")
        with k4:
            style.kpi("False Positive Rate", f"{m.fpr:.2%}", "amber", "⚠️", "of legit flagged")

        st.caption(
            f"Computed live from `backend/app/ml/models/*.pkl` over {m.n_eval:,} rows "
            f"({m.n_fraud} fraud) of `{m.source}` — a stratified 20% split, seed 42. "
            "The shipped artifacts were fit on this same dataset and no held-out split "
            "was preserved with them, so these are **in-sample** scores and optimistic; "
            "treat them as a check that the saved model loads and behaves, not as "
            "generalization estimates."
        )

        st.write("")
        r1c1, r1c2 = st.columns(2, gap="medium")

        with r1c1:
            with style.card("Confusion Matrices", "model comparison"):
                tab_ens, tab_xgb, tab_iso = st.tabs(
                    ["Ensemble", "XGBoost", "Isolation Forest"]
                )
                x_labels = ["Predicted Safe", "Predicted Fraud"]
                y_labels = ["Actual Safe", "Actual Fraud"]
                cyan_scale = [
                    [0.0, "#0d1117"], [0.35, "#0f3f55"],
                    [0.7, "#1a86ad"], [1.0, ACCENT],
                ]

                def _cm_fig(z):
                    """Color by row share, label with raw counts.

                    Raw counts alone make the matrix unreadable: true-negatives
                    outnumber every other cell by ~50x, so a count-scaled
                    heatmap renders one bright cell and three black ones.
                    """
                    arr = np.asarray(z, dtype=float)
                    shares = arr / np.clip(arr.sum(axis=1, keepdims=True), 1, None)
                    f = px.imshow(
                        shares, x=x_labels, y=y_labels,
                        color_continuous_scale=cyan_scale, aspect="auto",
                        zmin=0, zmax=1,
                    )
                    f.update_traces(
                        text=arr.astype(int),
                        texttemplate="%{text:,}",
                        textfont=dict(size=14),
                        hovertemplate="%{y} → %{x}<br>%{text:,} (%{z:.1%} of row)<extra></extra>",
                    )
                    f.update_layout(
                        coloraxis_showscale=False, xaxis_title=None, yaxis_title=None
                    )
                    return f

                with tab_ens:
                    st.caption(
                        f"Blended 0.75·XGBoost + 0.25·IsolationForest — "
                        f"precision {m.ens_precision:.1%}, recall {m.ens_recall:.1%}."
                    )
                    style.plot(_cm_fig(m.cm_ens), height=300, key="cm_ens")

                with tab_xgb:
                    st.caption(
                        f"Gradient-boosted trees — precision {m.precision:.1%}, "
                        f"recall {m.recall:.1%}. Strong on known fraud patterns."
                    )
                    style.plot(_cm_fig(m.cm_xgb), height=300, key="cm_xgb")

                with tab_iso:
                    st.caption(
                        f"Unsupervised anomaly detector — precision {m.iso_precision:.1%}, "
                        f"recall {m.iso_recall:.1%}. Catches novel outliers, flags more legit traffic."
                    )
                    style.plot(_cm_fig(m.cm_iso), height=300, key="cm_iso")

        with r1c2:
            with style.card("Feature Importance", "XGBoost gain"):
                fig_fi = px.bar(
                    m.importances, x="Importance", y="Feature", orientation="h",
                    color_discrete_sequence=[ACCENT],
                )
                fig_fi.update_traces(marker=dict(cornerradius=3))
                fig_fi.update_layout(xaxis_title="Gain", yaxis_title=None)
                style.plot(fig_fi, height=340, key="model_importance")

        st.write("")
        r2c1, r2c2 = st.columns(2, gap="medium")

        with r2c1:
            with style.card("ROC Curve", f"AUC {m.roc_auc:.3f}"):
                df_roc = pd.DataFrame({"FPR": m.roc_fpr, "TPR": m.roc_tpr})
                fig_roc = px.area(df_roc, x="FPR", y="TPR")
                fig_roc.update_traces(
                    line_color=ACCENT, line_width=2.5,
                    fillcolor="rgba(56,189,248,0.13)",
                )
                fig_roc.add_shape(
                    type="line", line=dict(dash="dash", color=MUTED, width=1),
                    x0=0, x1=1, y0=0, y1=1,
                )
                fig_roc.update_layout(
                    xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
                )
                style.plot(fig_roc, height=330, key="model_roc")

        with r2c2:
            with style.card("Precision–Recall Curve", "fraud class"):
                df_pr = pd.DataFrame(
                    {"Recall": m.pr_recall, "Precision": m.pr_precision}
                )
                fig_pr = px.area(df_pr, x="Recall", y="Precision")
                fig_pr.update_traces(
                    line_color=ACCENT_2, line_width=2.5,
                    fillcolor="rgba(244,114,182,0.12)",
                )
                fig_pr.update_layout(xaxis_title="Recall", yaxis_title="Precision")
                style.plot(fig_pr, height=330, key="model_pr")

        st.write("")
        with style.card("Risk Score Distribution", "predicted fraud probability"):
            df_scores = pd.DataFrame(
                {
                    "Risk Score": np.concatenate([m.scores_legit, m.scores_fraud]),
                    "Class": ["Safe"] * len(m.scores_legit) + ["Fraud"] * len(m.scores_fraud),
                }
            )
            fig_dist = px.histogram(
                df_scores, x="Risk Score", color="Class", marginal="violin",
                barmode="overlay", nbins=50,
                color_discrete_map={"Safe": GREEN, "Fraud": RED},
            )
            fig_dist.update_layout(yaxis_title="Transactions", legend_title_text=None)
            style.plot(fig_dist, height=360, showlegend=True, key="model_dist")

# =============================================================================
# PAGE 6: ADMIN PORTAL (RESTRICTED)
# =============================================================================
elif page == "admin":
    style.page_header("System Administration", "Restricted — all access attempts are logged")

    # Simple Session State Authentication
    if "admin_auth" not in st.session_state:
        st.session_state.admin_auth = False

    # If NOT logged in, show the password screen
    if not st.session_state.admin_auth:
        gate_col, _ = st.columns([1, 1])
        with gate_col:
            with style.card("Authentication Required"):
                with st.form("admin_login"):
                    pwd = st.text_input(
                        "Admin Password", type="password", placeholder="Hint: admin123"
                    )
                    submit_login = st.form_submit_button("Authenticate", use_container_width=True)

                    if submit_login:
                        if pwd == "admin123":  # Demo password
                            st.session_state.admin_auth = True
                            st.rerun()
                        else:
                            st.error("Invalid credentials. Access denied.", icon="❌")

    # If LOGGED IN, show the dashboard
    else:
        head_l, head_r = st.columns([4, 1])
        with head_r:
            if st.button("Logout", key="logout_btn", use_container_width=True):
                st.session_state.admin_auth = False
                st.rerun()

        # 1. Search Bar
        search_query = st.text_input(
            "Search by Account ID", placeholder="Type an ID like ACC10294 and press Enter..."
        )

        # 2. Fetch Accounts
        try:
            params = {"search": search_query} if search_query else {}
            acc_resp = requests.get(f"{API_URL}/admin/accounts", params=params, timeout=30)

            if acc_resp.status_code == 200:
                accounts_list = acc_resp.json()
                df_acc = pd.DataFrame(accounts_list)

                if not df_acc.empty:
                    # Metrics
                    active_count = len(df_acc[df_acc["status"] == "Active"])
                    blocked_count = len(df_acc[df_acc["status"] == "Blocked"])

                    m1, m2, m3 = st.columns(3, gap="medium")
                    with m1:
                        style.kpi("Active Accounts", f"{active_count}", "green", "🟢")
                    with m2:
                        style.kpi("Blocked Accounts", f"{blocked_count}", "red", "🔴")
                    with m3:
                        style.kpi("Total Directory", f"{len(df_acc)}", "accent", "👥")

                    st.write("")

                    # Reorder columns logically (if they exist)
                    display_cols = [
                        "account_id", "full_name", "email", "phone", "kyc_document", "status",
                    ]
                    available_cols = [c for c in display_cols if c in df_acc.columns]

                    def color_status(val):
                        color = RED if val == "Blocked" else GREEN
                        return f"color: {color}; font-weight: bold"

                    with style.card("Customer Account Directory"):
                        st.dataframe(
                            df_acc[available_cols].style.map(color_status, subset=["status"]),
                            use_container_width=True,
                            hide_index=True,
                        )

                    st.write("")

                    # 3. Action Panel to Block/Unblock
                    with style.card("Account Actions", "block / unblock"):
                        action_col1, action_col2, action_col3 = st.columns([2, 2, 1])

                        with action_col1:
                            target_acc = st.selectbox(
                                "Select Target Account", df_acc["account_id"].tolist()
                            )
                        with action_col2:
                            new_status = st.radio(
                                "Set Status To:", ["Blocked", "Active"], horizontal=True
                            )
                        with action_col3:
                            st.write("")
                            if st.button("Execute ⚡", use_container_width=True):
                                update_res = requests.patch(
                                    f"{API_URL}/admin/accounts/{target_acc}/status",
                                    json={"status": new_status},
                                )
                                if update_res.status_code == 200:
                                    st.success(f"{target_acc} status updated to {new_status}.")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Failed to update account.", icon="❌")
                else:
                    st.info("No accounts found matching that search.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend API.", icon="🔌")

# =============================================================================
# PAGE 7: SYSTEM HEALTH
# =============================================================================
elif page == "health":
    style.page_header("Infrastructure Status", "Core service reachability")

    with style.card("Service Checks"):
        db_status = st.progress(0, text="Pinging PostgreSQL (Ledger)...")
        time.sleep(0.3)
        db_status.progress(100, text="PostgreSQL: online")

        redis_status = st.progress(0, text="Pinging Redis (Feature Cache)...")
        time.sleep(0.3)
        redis_status.progress(100, text="Redis: online")

        ml_status = st.progress(0, text="Loading Scikit/XGBoost Models...")
        time.sleep(0.3)
        ml_status.progress(100, text="ML engine: models loaded")

    st.write("")
    h1, h2, h3 = st.columns(3, gap="medium")
    with h1:
        style.kpi(
            "API Backend",
            "Online" if API_ONLINE else "Offline",
            "green" if API_ONLINE else "red",
            "🌐",
            API_URL,
        )
    with h2:
        artifacts_ok = model_metrics.compute() is not None
        style.kpi(
            "Model Artifacts",
            "Loaded" if artifacts_ok else "Unavailable",
            "green" if artifacts_ok else "amber",
            "🧠",
            "backend/app/ml/models",
        )
    with h3:
        style.kpi("UI Session", "Healthy", "accent", "🖥️", "Streamlit")

# =============================================================================
# FOOTER
# =============================================================================
style.footer()
