"""
FraudGuard — centralized dark theme.

Everything visual lives here: the palette, the CSS injected into the page, the
Plotly template every chart inherits, and the small card/KPI helpers the pages
build their layout from. Pages should not carry inline style strings.
"""
from contextlib import contextmanager

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
BG = "#0d1117"          # page background — near-black, not pure black
BG_DEEP = "#0a0e14"     # sidebar rail, deepest surface
SURFACE = "#161b22"     # card background
SURFACE_HI = "#1c2229"  # hovered / raised card
BORDER = "#22272e"      # 1px card border

ACCENT = "#38bdf8"      # primary — key metrics, active nav, primary series
ACCENT_ALT = "#22d3ee"  # primary, cooler end (gradients, secondary emphasis)
ACCENT_2 = "#f472b6"    # magenta — second data series only, used sparingly

TEXT = "#e6edf3"        # primary text, off-white
MUTED = "#8b949e"       # secondary text

GREEN = "#4ade80"       # safe / legit / approved
RED = "#f87171"         # flagged / fraud / declined
AMBER = "#fbbf24"       # awaiting verification / pending MFA

GRID = "rgba(139, 148, 158, 0.12)"   # low-opacity gridlines
AXIS = "rgba(139, 148, 158, 0.25)"

# Ordered series colors for multi-trace charts. Cyan leads, magenta seconds.
COLORWAY = [ACCENT, ACCENT_2, GREEN, AMBER, ACCENT_ALT, RED]

STATUS_COLORS = {
    "Approved": GREEN,
    "Declined": RED,
    "Awaiting Verification": AMBER,
}

RADIUS = "10px"
SIDEBAR_W = 76      # collapsed icon rail
SIDEBAR_W_OPEN = 236  # expanded on hover


# ─────────────────────────────────────────────────────────────────────────────
# Plotly template
# ─────────────────────────────────────────────────────────────────────────────
def _build_template() -> go.layout.Template:
    """Dark Plotly template: transparent plot area, faint grid, accent traces."""
    tpl = go.layout.Template(pio.templates["plotly_dark"])

    tpl.layout.paper_bgcolor = "rgba(0,0,0,0)"
    tpl.layout.plot_bgcolor = "rgba(0,0,0,0)"
    tpl.layout.colorway = COLORWAY
    tpl.layout.font = dict(
        family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        color=MUTED,
        size=12,
    )
    tpl.layout.title = dict(font=dict(color=TEXT, size=14))
    tpl.layout.margin = dict(l=8, r=8, t=28, b=8)
    tpl.layout.hoverlabel = dict(
        bgcolor=SURFACE_HI,
        bordercolor=BORDER,
        font=dict(color=TEXT, size=12),
    )
    tpl.layout.legend = dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=11),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        title=dict(font=dict(color=MUTED, size=11)),
    )

    axis = dict(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=AXIS,
        tickfont=dict(color=MUTED, size=11),
        title=dict(font=dict(color=MUTED, size=12)),
        showspikes=False,
        # Without this the tight card margins clip tick labels and axis titles
        automargin=True,
    )
    tpl.layout.xaxis = axis
    tpl.layout.yaxis = axis

    # Sequential scale for heatmaps/imshow — dark base climbing to cyan
    tpl.layout.colorscale.sequential = [
        [0.0, "#0d1117"], [0.25, "#0f3547"], [0.5, "#12607f"],
        [0.75, "#1e93bd"], [1.0, ACCENT],
    ]
    tpl.layout.coloraxis = dict(
        colorbar=dict(
            outlinewidth=0,
            tickfont=dict(color=MUTED, size=10),
            thickness=10,
        )
    )
    return tpl


def apply_plotly_theme() -> None:
    """Register the FraudGuard template and make it the default for all figures."""
    pio.templates["fraudguard"] = _build_template()
    pio.templates.default = "fraudguard"


def plot(
    fig,
    height: int | None = None,
    showlegend: bool | None = None,
    key: str | None = None,
) -> None:
    """Apply the theme and render.

    `theme=None` is deliberate: Streamlit's own chart theme would otherwise
    override the template's colors and backgrounds.

    `key` is required whenever two charts on the same page could be built with
    identical parameters (e.g. the confusion matrices across tabs) — Streamlit
    derives element IDs from those parameters and raises
    StreamlitDuplicateElementId on a collision.
    """
    style_fig(fig, height=height, showlegend=showlegend)
    st.plotly_chart(fig, use_container_width=True, theme=None, key=key)


def style_fig(fig, height: int | None = None, showlegend: bool | None = None):
    """Final pass applied to every figure before rendering."""
    fig.update_layout(
        template="fraudguard",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=24, b=8),
    )
    # automargin expands these as needed; the base margin just sets the floor
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if height is not None:
        fig.update_layout(height=height)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
:root {{
  --fg-bg: {BG};
  --fg-bg-deep: {BG_DEEP};
  --fg-surface: {SURFACE};
  --fg-surface-hi: {SURFACE_HI};
  --fg-border: {BORDER};
  --fg-accent: {ACCENT};
  --fg-accent-2: {ACCENT_2};
  --fg-text: {TEXT};
  --fg-muted: {MUTED};
  --fg-green: {GREEN};
  --fg-red: {RED};
  --fg-amber: {AMBER};
  --fg-radius: {RADIUS};
}}

/* ── Base ───────────────────────────────────────────────────────────────── */
.stApp {{ background: var(--fg-bg); }}

[data-testid="stAppViewContainer"] > .main .block-container,
.main .block-container {{
  padding: 1.6rem 1.75rem 3rem 1.75rem;
  max-width: 1500px;
}}

/* Kill Streamlit chrome that fights the design */
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; }}
[data-testid="stHeader"] {{ background: transparent; height: 0; }}

html, body, [class*="css"] {{ color: var(--fg-text); }}

h1, h2, h3, h4, h5, h6 {{ color: var(--fg-text); letter-spacing: -0.01em; }}
h1 {{ font-size: 1.7rem; font-weight: 650; }}
h2 {{ font-size: 1.25rem; font-weight: 620; }}
h3 {{ font-size: 1.02rem; font-weight: 600; }}
p, span, label, li {{ color: var(--fg-text); }}
a {{ color: var(--fg-accent); }}
hr {{ border-color: var(--fg-border); }}

/* ── Page header ────────────────────────────────────────────────────────── */
.fg-page-head {{
  display: flex; align-items: baseline; gap: 14px;
  margin: 0 0 20px 0; padding-bottom: 14px;
  border-bottom: 1px solid var(--fg-border);
}}
.fg-page-title {{
  font-size: 1.6rem; font-weight: 650; color: var(--fg-text);
  letter-spacing: -0.02em; line-height: 1.2;
}}
.fg-page-sub {{ font-size: 0.86rem; color: var(--fg-muted); }}

/* ── Sidebar: fixed icon rail, expands on hover ─────────────────────────── */
[data-testid="stSidebar"] {{
  background: var(--fg-bg-deep);
  border-right: 1px solid var(--fg-border);
  width: {SIDEBAR_W}px !important;
  min-width: {SIDEBAR_W}px !important;
  transition: width .18s ease, min-width .18s ease;
  overflow: visible;
  z-index: 999;
}}
[data-testid="stSidebar"]:hover {{
  width: {SIDEBAR_W_OPEN}px !important;
  min-width: {SIDEBAR_W_OPEN}px !important;
  box-shadow: 8px 0 28px rgba(0,0,0,.45);
}}
[data-testid="stSidebar"] > div {{ overflow-x: hidden; }}
[data-testid="stSidebarContent"] {{ padding: 6px 0 10px 0; }}
/* Hide the collapse/resize affordances so the rail width stays ours.
   stSidebarCollapsedControl is the floating "expand" arrow that otherwise
   overlaps the brand mark. */
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarResizeHandle"],
[data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
/* Streamlit reserves ~78px of empty header above the sidebar content */
[data-testid="stSidebarHeader"] {{ height: 0; min-height: 0; padding: 0; }}
[data-testid="stSidebarUserContent"] {{ padding-top: 14px; }}

.fg-brand {{
  display: flex; align-items: center; gap: 11px;
  padding: 4px 0 16px 0; margin: 0 0 10px 0;
  border-bottom: 1px solid var(--fg-border);
  white-space: nowrap; overflow: hidden;
}}
.fg-brand-mark {{
  flex: 0 0 40px; width: 40px; height: 40px; margin-left: 18px;
  border-radius: var(--fg-radius);
  background: linear-gradient(140deg, var(--fg-accent), #0e7490);
  display: flex; align-items: center; justify-content: center;
  font-size: 20px; box-shadow: 0 0 18px rgba(56,189,248,.28);
}}
.fg-brand-text {{ opacity: 0; transition: opacity .16s ease; }}
[data-testid="stSidebar"]:hover .fg-brand-text {{ opacity: 1; }}
.fg-brand-name {{ font-size: 0.98rem; font-weight: 650; color: var(--fg-text); line-height: 1.15; }}
.fg-brand-tag {{ font-size: 0.7rem; color: var(--fg-muted); letter-spacing: .04em; text-transform: uppercase; }}

/* Nav radio → icon tiles */
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 4px; padding: 0 12px; }}
[data-testid="stSidebar"] [role="radiogroup"] > label {{
  display: flex; align-items: center; gap: 12px;
  padding: 0; margin: 0; height: 44px;
  border-radius: 9px; cursor: pointer;
  border: 1px solid transparent;
  transition: background .14s ease, border-color .14s ease;
  white-space: nowrap; overflow: hidden;
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:hover {{
  background: rgba(56,189,248,.07);
}}
/* Hide the native radio dot, keep the label clickable */
[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child {{
  display: none !important;
}}
/* Collapsed rail: the label text shrinks to nothing and the icon is drawn by a
   ::before pseudo-element (Streamlit escapes HTML inside widget labels, so the
   icon cannot be markup). Hovering the sidebar restores the text. */
[data-testid="stSidebar"] [role="radiogroup"] label p {{
  font-size: 0 !important;
  color: var(--fg-muted); margin: 0;
  transition: font-size .16s ease;
  white-space: nowrap;
}}
[data-testid="stSidebar"]:hover [role="radiogroup"] label p {{
  font-size: 0.87rem !important;
}}
[data-testid="stSidebar"] [role="radiogroup"] label p::before {{
  display: inline-block; width: 26px; margin: 0 12px 0 13px;
  font-size: 1.05rem; text-align: center; vertical-align: middle;
}}

/* Active item */
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {{
  background: rgba(56,189,248,.12);
  border-color: rgba(56,189,248,.30);
  box-shadow: inset 2px 0 0 var(--fg-accent);
}}
[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) p {{
  color: var(--fg-accent); font-weight: 600;
}}
/* The radio group's own "Navigation" label */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {{ display: none; }}

.fg-side-foot {{
  padding: 12px 0 0 0; margin-top: 12px;
  border-top: 1px solid var(--fg-border);
  white-space: nowrap; overflow: hidden;
}}
.fg-side-foot-in {{
  opacity: 0; transition: opacity .16s ease;
  font-size: 0.7rem; color: var(--fg-muted); padding-left: 18px;
}}
[data-testid="stSidebar"]:hover .fg-side-foot-in {{ opacity: 1; }}

/* ── KPI cards ──────────────────────────────────────────────────────────── */
.fg-kpi {{
  background: var(--fg-surface);
  border: 1px solid var(--fg-border);
  border-radius: var(--fg-radius);
  padding: 17px 18px 16px 18px;
  box-shadow: 0 1px 3px rgba(0,0,0,.35);
  height: 100%;
  transition: border-color .15s ease, transform .15s ease;
}}
.fg-kpi:hover {{ border-color: #2d343d; transform: translateY(-1px); }}
.fg-kpi-top {{
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 9px;
}}
.fg-kpi-label {{
  font-size: 0.68rem; font-weight: 640; color: var(--fg-muted);
  text-transform: uppercase; letter-spacing: .07em;
}}
.fg-kpi-ico {{ font-size: 0.82rem; opacity: .55; }}
.fg-kpi-value {{
  font-size: 1.85rem; font-weight: 660; color: var(--fg-text);
  line-height: 1.12; letter-spacing: -0.02em;
  font-variant-numeric: tabular-nums;
}}
.fg-kpi-value.accent {{ color: var(--fg-accent); }}
.fg-kpi-value.green  {{ color: var(--fg-green); }}
.fg-kpi-value.red    {{ color: var(--fg-red); }}
.fg-kpi-value.amber  {{ color: var(--fg-amber); }}
.fg-kpi-foot {{ font-size: 0.72rem; color: var(--fg-muted); margin-top: 6px; }}

/* ── Chart cards (st.container(border=True)) ────────────────────────────── */
/* Scoped to containers whose OWN first child holds a .fg-card-head marker.
   The child combinators matter: a bare :has(.fg-card-head) also matches every
   ancestor block, which paints a border around the whole page. Selecting on
   testids rather than st-emotion-cache-* hashes keeps this stable across
   Streamlit builds. */
[data-testid="stVerticalBlockBorderWrapper"]:has(
    > div > [data-testid="stElementContainer"] .fg-card-head) {{
  background: var(--fg-surface);
  border: 1px solid var(--fg-border) !important;
  border-radius: var(--fg-radius);
  box-shadow: 0 1px 3px rgba(0,0,0,.35);
}}
[data-testid="stVerticalBlockBorderWrapper"]:has(
    > div > [data-testid="stElementContainer"] .fg-card-head) > div {{
  padding: 15px 17px 13px 17px;
}}

.fg-card-head {{
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; padding: 2px 2px 11px 2px; margin-bottom: 11px;
  border-bottom: 1px solid var(--fg-border);
}}
.fg-card-head--bare {{ padding: 0; margin: 0; border-bottom: none; }}
.fg-card-title {{
  font-size: 0.86rem; font-weight: 620; color: var(--fg-text);
  letter-spacing: -0.005em;
}}
.fg-card-note {{
  font-size: 0.68rem; color: var(--fg-muted);
  text-transform: uppercase; letter-spacing: .06em; font-weight: 600;
}}

/* ── Status pills ───────────────────────────────────────────────────────── */
.fg-pill {{
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 600;
}}
.fg-pill.ok   {{ background: rgba(74,222,128,.12); color: var(--fg-green);
                 border: 1px solid rgba(74,222,128,.28); }}
.fg-pill.bad  {{ background: rgba(248,113,113,.12); color: var(--fg-red);
                 border: 1px solid rgba(248,113,113,.28); }}
.fg-pill.warn {{ background: rgba(251,191,36,.12); color: var(--fg-amber);
                 border: 1px solid rgba(251,191,36,.28); }}
.fg-dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor; }}

/* ── Widgets ────────────────────────────────────────────────────────────── */
.stTextInput input, .stNumberInput input, .stTextArea textarea,
[data-baseweb="select"] > div {{
  background: var(--fg-bg-deep) !important;
  border: 1px solid var(--fg-border) !important;
  border-radius: 8px !important;
  color: var(--fg-text) !important;
}}
.stTextInput input:focus, .stNumberInput input:focus {{
  border-color: var(--fg-accent) !important;
  box-shadow: 0 0 0 2px rgba(56,189,248,.16) !important;
}}
input::placeholder, textarea::placeholder {{ color: #6e7681 !important; }}

.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
  background: var(--fg-surface-hi);
  color: var(--fg-text);
  border: 1px solid var(--fg-border);
  border-radius: 8px;
  font-weight: 560; font-size: 0.86rem;
  padding: 0.46rem 1rem;
  transition: all .14s ease;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover,
.stDownloadButton > button:hover {{
  border-color: var(--fg-accent);
  color: var(--fg-accent);
  background: rgba(56,189,248,.08);
}}
.stFormSubmitButton > button {{
  background: rgba(56,189,248,.14);
  border-color: rgba(56,189,248,.35);
  color: var(--fg-accent);
}}
.stFormSubmitButton > button:hover {{
  background: rgba(56,189,248,.22);
}}

[data-testid="stForm"] {{
  background: transparent;
  border: 1px solid var(--fg-border);
  border-radius: var(--fg-radius);
  padding: 18px;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
  gap: 4px; border-bottom: 1px solid var(--fg-border); background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
  height: 36px; padding: 0 14px; background: transparent;
  color: var(--fg-muted); font-size: 0.83rem; font-weight: 560;
  border-radius: 8px 8px 0 0;
}}
.stTabs [aria-selected="true"] {{
  color: var(--fg-accent) !important;
  background: rgba(56,189,248,.08);
}}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--fg-accent); }}

/* Dataframes */
[data-testid="stDataFrame"] {{
  border: 1px solid var(--fg-border);
  border-radius: 8px;
  overflow: hidden;
}}

/* Metrics (st.metric) */
[data-testid="stMetric"] {{
  background: var(--fg-surface);
  border: 1px solid var(--fg-border);
  border-radius: var(--fg-radius);
  padding: 14px 16px;
}}
[data-testid="stMetricLabel"] p {{
  color: var(--fg-muted) !important; font-size: 0.72rem !important;
  text-transform: uppercase; letter-spacing: .06em; font-weight: 620;
}}
[data-testid="stMetricValue"] {{
  color: var(--fg-text); font-size: 1.6rem; font-variant-numeric: tabular-nums;
}}

/* Alerts — flatten Streamlit's default fills into the dark surface */
[data-testid="stAlert"] {{
  border-radius: 9px; border: 1px solid var(--fg-border);
  background: var(--fg-surface); color: var(--fg-text);
}}

/* Progress */
.stProgress > div > div > div > div {{
  background: linear-gradient(90deg, var(--fg-accent), {ACCENT_ALT});
}}
.stProgress > div > div > div {{ background: var(--fg-border); }}

/* Expander */
[data-testid="stExpander"] {{
  border: 1px solid var(--fg-border); border-radius: var(--fg-radius);
  background: var(--fg-surface);
}}

/* Plotly toolbar — hide until hover, it's visual noise */
.js-plotly-plot .modebar {{ opacity: 0; transition: opacity .15s ease; }}
[data-testid="stVerticalBlockBorderWrapper"]:hover .js-plotly-plot .modebar {{ opacity: .55; }}

/* Footer */
.fg-foot {{
  text-align: center; color: var(--fg-muted); font-size: 0.75rem;
  padding: 26px 0 6px 0; margin-top: 30px;
  border-top: 1px solid var(--fg-border);
}}
.fg-foot a {{ color: var(--fg-muted); text-decoration: underline; }}
.fg-foot a:hover {{ color: var(--fg-accent); }}

/* Scrollbars */
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-track {{ background: var(--fg-bg); }}
::-webkit-scrollbar-thumb {{ background: #262c36; border-radius: 5px; }}
::-webkit-scrollbar-thumb:hover {{ background: #333b47; }}
</style>
"""


def inject_css() -> None:
    """Inject the stylesheet once per run. Call immediately after set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)


def inject_nav_icons(icons: list[str]) -> None:
    """Bind each nav item's icon to its position in the radio group."""
    rules = "\n".join(
        f'[data-testid="stSidebar"] [role="radiogroup"] > label:nth-of-type({i}) '
        f'p::before {{ content: "{ico}"; }}'
        for i, ico in enumerate(icons, start=1)
    )
    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────
def page_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="fg-page-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="fg-page-head"><div class="fg-page-title">{title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, tone: str = "", icon: str = "", foot: str = "") -> None:
    """Render one KPI stat card. `tone` ∈ {'', 'accent', 'green', 'red', 'amber'}."""
    ico = f'<div class="fg-kpi-ico">{icon}</div>' if icon else ""
    ft = f'<div class="fg-kpi-foot">{foot}</div>' if foot else ""
    st.markdown(
        f'<div class="fg-kpi">'
        f'<div class="fg-kpi-top"><div class="fg-kpi-label">{label}</div>{ico}</div>'
        f'<div class="fg-kpi-value {tone}">{value}</div>{ft}</div>',
        unsafe_allow_html=True,
    )


@contextmanager
def card(title: str = "", note: str = ""):
    """A bordered surface with a small header bar. Yields inside the container.

    The .fg-card-head element is always emitted — the CSS uses it to tell our
    cards apart from Streamlit's other block wrappers, so a card without a
    title still needs the (bare) marker.
    """
    with st.container(border=True):
        if title:
            nt = f'<div class="fg-card-note">{note}</div>' if note else ""
            st.markdown(
                f'<div class="fg-card-head">'
                f'<div class="fg-card-title">{title}</div>{nt}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="fg-card-head fg-card-head--bare"></div>',
                unsafe_allow_html=True,
            )
        yield


def pill(text: str, tone: str = "ok") -> str:
    """Inline status pill markup. `tone` ∈ {'ok', 'bad', 'warn'}."""
    return f'<span class="fg-pill {tone}"><span class="fg-dot"></span>{text}</span>'


def footer() -> None:
    st.markdown(
        '<div class="fg-foot">'
        "Built by <strong>Nikunj Sharma</strong> &nbsp;·&nbsp; Roll No. 230107046"
        " &nbsp;·&nbsp; IIT Guwahati &nbsp;·&nbsp; "
        '<a href="https://github.com/NikunjSharma-dev/fraud-detection-system/tree/main" '
        'target="_blank">github.com/NikunjSharma-dev</a></div>',
        unsafe_allow_html=True,
    )
