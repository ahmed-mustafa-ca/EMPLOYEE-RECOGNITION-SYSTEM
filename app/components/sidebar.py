import streamlit as st


_CSS = """
<style>
/* ── Global ──────────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hero card ───────────────────────────────────────────────────────────── */
.hero-card {
    background: linear-gradient(135deg, #1a237e 0%, #1565c0 60%, #0288d1 100%);
    border-radius: 14px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 1.5rem;
    color: #fff;
}
.hero-card h1 { font-size: 1.9rem; font-weight: 700; margin-bottom: .4rem; }
.hero-card p  { font-size: 1rem;   opacity: .85; margin: 0; }

/* ── Feature cards ───────────────────────────────────────────────────────── */
.feature-card {
    background: #1e2130;
    border: 1px solid #2a2f45;
    border-radius: 12px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    height: 100%;
    transition: border-color .2s;
}
.feature-card:hover { border-color: #4f8ef7; }
.feature-icon  { font-size: 2rem; margin-bottom: .5rem; }
.feature-card h3 { font-size: 1rem;  font-weight: 600; margin-bottom: .4rem; }
.feature-card p  { font-size: .82rem; color: #8b92a5; margin: 0; }

/* ── Status badges ───────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: .18rem .55rem;
    border-radius: 999px;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .02em;
}
.badge-present { background: #0d3b2e; color: #00d48b; border: 1px solid #00d48b55; }
.badge-late    { background: #3b2a0d; color: #f7a84f; border: 1px solid #f7a84f55; }
.badge-absent  { background: #3b0d0d; color: #f75f5f; border: 1px solid #f75f5f55; }

/* ── Section header ──────────────────────────────────────────────────────── */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #c5cae9;
    border-left: 3px solid #4f8ef7;
    padding-left: .6rem;
    margin: 1.2rem 0 .8rem;
}

/* ── Info box ────────────────────────────────────────────────────────────── */
.info-box {
    background: #1a2540;
    border: 1px solid #2a3a60;
    border-radius: 8px;
    padding: .8rem 1rem;
    font-size: .85rem;
    color: #90a4d4;
    margin-bottom: .8rem;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #12151f; }
[data-testid="stSidebar"] hr { border-color: #2a2f45; }

/* ── Dataframe tweaks ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Primary button ──────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #1565c0;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: .45rem 1.4rem;
    transition: background .2s;
}
.stButton > button[kind="primary"]:hover { background: #1976d2; }
</style>
"""


def inject_css() -> None:
    """Inject global CSS. Call once per page before any other Streamlit output."""
    st.markdown(_CSS, unsafe_allow_html=True)


def render_sidebar() -> None:
    inject_css()
    with st.sidebar:
        st.markdown("## ERS")
        st.markdown("<p style='color:#8b92a5;font-size:.82rem;margin-top:-.5rem;'>Employee Recognition System</p>", unsafe_allow_html=True)
        st.markdown("---")
        st.page_link("main_app.py",                label="Home",             icon="🏠")
        st.page_link("pages/live_recognition.py",  label="Live Recognition", icon="🎥")
        st.page_link("pages/register_employee.py", label="Register Employee",icon="➕")
        st.page_link("pages/attendance.py",        label="Attendance",       icon="📋")
        st.page_link("pages/dashboard.py",         label="Dashboard",        icon="📊")
        st.markdown("---")
        st.caption("v1.0.0 · Employee Recognition System")
