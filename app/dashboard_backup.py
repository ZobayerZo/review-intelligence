
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "absa"

RESULTS_PATH = DATA_DIR / "absa_pilot_results.parquet"
SUMMARY_PATH = DATA_DIR / "absa_pilot_summary.csv"

sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Review Intelligence | AI Analytics",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Visual theme
BG = "#0B1020"
CARD = "#131B2D"
ACCENT = "#8B7CFF"
TEAL = "#28C7B7"
GREEN = "#28C88A"
RED = "#F16C78"
AMBER = "#F2B95F"
TEXT = "#EDF1FA"
MUTED = "#9AA8BF"

SENTIMENT_COLORS = {
    "Positive": GREEN,
    "Neutral": AMBER,
    "Negative": RED
}

SENTIMENT_SCORE = {
    "Positive": 1,
    "Neutral": 0,
    "Negative": -1
}


# ============================================================
# 2. CUSTOM DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(104,83,220,.12), transparent 38%),
        radial-gradient(ellipse at 95% 15%, rgba(40,199,183,.08), transparent 32%),
        #0B1020;
    color: #EDF1FA;
}

.block-container {
    max-width: 1600px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #101729;
    border-right: 1px solid rgba(148,163,184,.12);
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg, #17223A, #111A2D);
    border: 1px solid rgba(139,124,255,.19);
    border-radius: 15px;
    padding: 18px 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,.15);
    transition: transform .25s ease, border-color .25s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(139,124,255,.65);
}

[data-testid="stMetricLabel"] {
    color: #A9B6CE !important;
}

[data-testid="stMetricValue"] {
    color: #F5F7FF !important;
    font-weight: 800;
}

[data-testid="stMetricDelta"] {
    font-size: .78rem;
}

h1, h2, h3 {
    letter-spacing: -.5px;
}

h1 {
    font-weight: 800;
}

div[data-testid="stPlotlyChart"] {
    background: rgba(19,27,45,.72);
    border: 1px solid rgba(148,163,184,.10);
    border-radius: 15px;
    padding: 8px;
}

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(148,163,184,.15);
    border-radius: 12px;
    overflow: hidden;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(139,124,255,.5);
    background: linear-gradient(110deg, #6E61D9, #397DAD);
    color: white;
    font-weight: 700;
    transition: all .2s ease;
}

.stButton > button:hover {
    border-color: #B9B0FF;
    transform: translateY(-2px);
    color: white;
}

.ri-brand {
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -.7px;
    background: linear-gradient(100deg, #B6A8FF, #58D8CF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.ri-subtitle {
    color: #91A0BA;
    font-size: .78rem;
    margin-top: -8px;
}

.ri-hero {
    background: linear-gradient(115deg,
        rgba(109,91,221,.23),
        rgba(38,179,173,.11),
        rgba(19,27,45,.7));
    border: 1px solid rgba(139,124,255,.23);
    border-radius: 20px;
    padding: 27px 30px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}

.ri-hero:after {
    content: "";
    position: absolute;
    width: 240px;
    height: 240px;
    right: -75px;
    top: -125px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(139,124,255,.25), transparent 70%);
    animation: floatglow 6s ease-in-out infinite alternate;
}

@keyframes floatglow {
    from { transform: translate(0,0); }
    to { transform: translate(-20px,25px); }
}

.ri-eyebrow {
    color: #9C91FF;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.ri-hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #F5F7FF;
    margin: 7px 0;
}

.ri-hero-desc {
    color: #AAB7CC;
    font-size: .9rem;
}

.ri-section {
    color: #EAF0FC;
    font-weight: 700;
    font-size: 1.12rem;
    margin: 15px 0 12px 0;
}

.ri-insight {
    background: linear-gradient(145deg, #17243A, #121A2C);
    border: 1px solid rgba(148,163,184,.13);
    border-radius: 14px;
    padding: 17px;
    min-height: 125px;
    margin-bottom: 10px;
}

.ri-insight-title {
    color: #A8B7D0;
    font-size: .76rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .7px;
}

.ri-insight-body {
    color: #F2F5FC;
    font-size: .93rem;
    line-height: 1.6;
    margin-top: 10px;
}

.ri-footer {
    color: #74839D;
    font-size: .73rem;
    text-align: center;
    padding: 20px 0 5px 0;
}

.ri-pill {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 30px;
    background: rgba(139,124,255,.15);
    border: 1px solid rgba(139,124,255,.25);
    color: #C6BDFF;
    font-size: .72rem;
    font-weight: 700;
}

#MainMenu, footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    aspects = pd.read_parquet(RESULTS_PATH)
    reviews = pd.read_csv(SUMMARY_PATH)

    aspects["confidence"] = pd.to_numeric(
        aspects["confidence"], errors="coerce"
    )

    aspects["rating"] = pd.to_numeric(
        aspects["rating"], errors="coerce"
    )

    reviews["rating"] = pd.to_numeric(
        reviews["rating"], errors="coerce"
    )

    aspects["aspect_sentiment"] = (
        aspects["aspect_sentiment"].astype(str)
    )

    aspects["normalized_aspect"] = (
        aspects["normalized_aspect"]
        .fillna("")
        .astype(str)
    )

    aspects["review_id"] = aspects["review_id"].astype(str)
    reviews["review_id"] = reviews["review_id"].astype(str)

    if "parent_asin" not in aspects.columns:
        aspects["parent_asin"] = None

    if "asin" not in aspects.columns:
        aspects["asin"] = None

    if "title" not in aspects.columns:
        aspects["title"] = ""

    return aspects, reviews


if not RESULTS_PATH.exists() or not SUMMARY_PATH.exists():
    st.error("ABSA pilot files are missing. Run src/run_absa_pilot.py first.")
    st.stop()

df, review_df = load_data()


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def hero(eyebrow, title, description):

    st.markdown(
        f"""
        <div class="ri-hero">
            <div class="ri-eyebrow">{eyebrow}</div>
            <div class="ri-hero-title">{title}</div>
            <div class="ri-hero-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def section(title):
    st.markdown(
        f'<div class="ri-section">{title}</div>',
        unsafe_allow_html=True
    )


def make_chart(fig, height=380):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="DM Sans",
            color=TEXT,
            size=12
        ),
        margin=dict(l=20, r=20, t=45, b=35),
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hoverlabel=dict(
            bgcolor="#202B43",
            bordercolor="#8174ED",
            font=dict(color="white")
        )
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,.10)",
        zerolinecolor="rgba(148,163,184,.15)"
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,.10)",
        zerolinecolor="rgba(148,163,184,.15)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True
        }
    )


def insight_card(title, body):

    st.markdown(
        f"""
        <div class="ri-insight">
            <div class="ri-insight-title">{title}</div>
            <div class="ri-insight-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def safe_percent(numerator, denominator):

    return (
        numerator / denominator * 100
        if denominator else 0
    )


def filtered_metrics(aspects, reviews):

    total = len(reviews)
    detected = int(reviews["has_aspects"].sum())
    predictions = len(aspects)

    avg_confidence = (
        aspects["confidence"].mean() * 100
        if predictions else 0
    )

    return total, detected, predictions, avg_confidence


def sentiment_distribution(aspects):

    return (
        aspects["aspect_sentiment"]
        .value_counts()
        .reindex(
            ["Positive", "Neutral", "Negative"],
            fill_value=0
        )
    )


def export_button(data, filename, label="Download CSV"):

    csv = data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# 5. SIDEBAR AND NAVIGATION
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="ri-brand">✦ Review Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ri-subtitle">AI-powered customer intelligence</div>',
        unsafe_allow_html=True
    )

    st.divider()

    pages = {
        "Executive Overview": "◈",
        "Sentiment Analytics": "◉",
        "Aspect Intelligence": "⌘",
        "Product Comparison": "▣",
        "Review Explorer": "▤",
        "AI Insights": "✧",
        "Live AI Analysis": "✦"
    }

    page = st.radio(
        "WORKSPACE",
        list(pages.keys()),
        format_func=lambda p: f"{pages[p]}   {p}",
        label_visibility="visible"
    )

    st.divider()

    st.markdown("**GLOBAL FILTERS**")

    all_categories = sorted(
        df["category"].dropna().unique().tolist()
    )

    selected_categories = st.multiselect(
        "Product categories",
        all_categories,
        default=all_categories
    )

    selected_sentiments = st.multiselect(
        "Sentiment",
        ["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"]
    )

    min_confidence = st.slider(
        "Minimum model confidence",
        0.0,
        1.0,
        0.0,
        0.05,
        format="%.2f"
    )

    st.divider()

    st.markdown("**DATASET STATUS**")
    st.markdown(
        f'<span class="ri-pill">PILOT DATASET</span>',
        unsafe_allow_html=True
    )

    st.caption(f"{len(review_df):,} sampled reviews")
    st.caption(f"{len(df):,} aspect predictions")
    st.caption("Amazon Reviews 2023")
    st.caption("DeBERTa-v3 ABSA")

    if st.button("Refresh cached data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# 6. GLOBAL FILTERS
# ============================================================

filtered_reviews = review_df[
    review_df["category"].isin(selected_categories)
].copy()

filtered_aspects = df[
    df["category"].isin(selected_categories)
    & df["aspect_sentiment"].isin(selected_sentiments)
    & (df["confidence"] >= min_confidence)
].copy()

total_reviews, detected_reviews, total_aspects, avg_conf = (
    filtered_metrics(filtered_aspects, filtered_reviews)
)

coverage = safe_percent(
    detected_reviews, total_reviews
)


# ============================================================
# 7. KPI DISPLAY
# ============================================================

def show_kpis():

    a, b, c, d = st.columns(4)

    a.metric("Reviews analyzed", f"{total_reviews:,}")
    b.metric("Aspect predictions", f"{total_aspects:,}")
    c.metric("Aspect coverage", f"{coverage:.1f}%")
    d.metric("Mean model confidence", f"{avg_conf:.1f}%")


# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================

if page == "Executive Overview":

    hero(
        "CUSTOMER INTELLIGENCE WORKSPACE",
        "Understand what customers really think.",
        "Transform unstructured customer reviews into aspect-level product intelligence."
    )

    show_kpis()

    st.divider()

    left, right = st.columns([1, 1.3])

    with left:

        section("Sentiment composition")

        counts = sentiment_distribution(filtered_aspects)

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=counts.index,
                    values=counts.values,
                    hole=.72,
                    marker=dict(
                        colors=[
                            GREEN, AMBER, RED
                        ],
                        line=dict(
                            color=BG,
                            width=3
                        )
                    ),
                    textinfo="percent",
                    hovertemplate=(
                        "%{label}<br>"
                        "Predictions: %{value}<br>"
                        "Share: %{percent}"
                        "<extra></extra>"
                    )
                )
            ]
        )

        fig.update_layout(
            annotations=[
                dict(
                    text=f"{total_aspects}<br>Aspects",
                    x=.5,
                    y=.5,
                    showarrow=False,
                    font=dict(size=19, color=TEXT)
                )
            ],
            showlegend=True
        )

        make_chart(fig, 360)

    with right:

        section("Sentiment across categories")

        if not filtered_aspects.empty:

            chart_data = (
                filtered_aspects.groupby(
                    ["category", "aspect_sentiment"]
                )
                .size()
                .reset_index(name="predictions")
            )

            fig = px.bar(
                chart_data,
                x="category",
                y="predictions",
                color="aspect_sentiment",
                barmode="stack",
                color_discrete_map=SENTIMENT_COLORS,
                labels={
                    "category": "Category",
                    "predictions": "Predictions",
                    "aspect_sentiment": "Sentiment"
                }
            )

            make_chart(fig, 360)

        else:
            st.info("No predictions match the selected filters.")

    section("Category intelligence")

    category_data = (
        filtered_reviews.groupby("category")
        .agg(
            reviews=("review_id", "nunique"),
            reviews_with_aspects=("has_aspects", "sum"),
            aspects=("aspect_count", "sum")
        )
        .reset_index()
    )

    category_data["coverage"] = (
        category_data["reviews_with_aspects"]
        / category_data["reviews"].replace(0, pd.NA)
        * 100
    ).round(1)

    st.dataframe(
        category_data,
        use_container_width=True,
        hide_index=True
    )

    section("Most discussed aspects")

    if not filtered_aspects.empty:

        top = (
            filtered_aspects.groupby("normalized_aspect")
            .size()
            .sort_values(ascending=False)
            .head(8)
            .reset_index(name="mentions")
        )

        fig = px.bar(
            top.sort_values("mentions"),
            x="mentions",
            y="normalized_aspect",
            orientation="h",
            color="mentions",
            color_continuous_scale=["#34466C", ACCENT, TEAL]
        )

        fig.update_layout(coloraxis_showscale=False)

        make_chart(fig, 340)


# ============================================================
# PAGE 2: SENTIMENT ANALYTICS
# ============================================================

elif page == "Sentiment Analytics":

    hero(
        "SENTIMENT LAB",
        "Measure customer perception.",
        "Explore sentiment composition, category differences and confidence patterns."
    )

    show_kpis()

    section("Sentiment distribution")

    counts = sentiment_distribution(filtered_aspects)

    sentiment_table = pd.DataFrame({
        "Sentiment": counts.index,
        "Predictions": counts.values
    })

    sentiment_table["Share (%)"] = (
        sentiment_table["Predictions"]
        / max(total_aspects, 1) * 100
    ).round(2)

    c1, c2 = st.columns([1.1, 1])

    with c1:

        fig = px.bar(
            sentiment_table,
            x="Sentiment",
            y="Predictions",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            text="Share (%)"
        )

        make_chart(fig, 350)

    with c2:

        fig = px.pie(
            sentiment_table,
            names="Sentiment",
            values="Predictions",
            hole=.58,
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS
        )

        make_chart(fig, 350)

    section("Aspect sentiment heatmap")

    if not filtered_aspects.empty:

        heat_data = (
            filtered_aspects.groupby(
                ["category", "normalized_aspect"]
            )
            .agg(
                positive=(
                    "aspect_sentiment",
                    lambda x: (x == "Positive").sum()
                ),
                neutral=(
                    "aspect_sentiment",
                    lambda x: (x == "Neutral").sum()
                ),
                negative=(
                    "aspect_sentiment",
                    lambda x: (x == "Negative").sum()
                )
            )
            .reset_index()
        )

        heat_data["total"] = (
            heat_data["positive"]
            + heat_data["neutral"]
            + heat_data["negative"]
        )

        heat_data["positive_share"] = (
            heat_data["positive"]
            / heat_data["total"].replace(0, 1)
            * 100
        )

        top_heat_aspects = (
            heat_data.groupby("normalized_aspect")["total"]
            .sum()
            .nlargest(15)
            .index
        )

        heat_data = heat_data[
            heat_data["normalized_aspect"].isin(top_heat_aspects)
        ]

        heat_pivot = heat_data.pivot(
            index="category",
            columns="normalized_aspect",
            values="positive_share"
        )

        fig = px.imshow(
            heat_pivot,
            aspect="auto",
            color_continuous_scale=[
                "#D65353",
                "#D69E36",
                "#16A36A"
            ],
            zmin=0,
            zmax=100,
            labels={
                "color": "Positive share (%)",
                "x": "Aspect",
                "y": "Category"
            }
        )

        make_chart(fig, 390)

    section("Model confidence distribution")

    if not filtered_aspects.empty:

        fig = px.histogram(
            filtered_aspects,
            x="confidence",
            color="aspect_sentiment",
            nbins=20,
            barmode="overlay",
            opacity=.8,
            color_discrete_map=SENTIMENT_COLORS
        )

        make_chart(fig, 330)

    st.caption(
        "Heatmap values show the percentage of positive predictions "
        "among predictions for each category/aspect combination. "
        "They are not accuracy scores."
    )


# ============================================================
# PAGE 3: ASPECT INTELLIGENCE
# ============================================================

elif page == "Aspect Intelligence":

    hero(
        "ASPECT DISCOVERY",
        "Explore product features in detail.",
        "Discover which features attract praise, criticism and mixed opinions."
    )

    show_kpis()

    if filtered_aspects.empty:
        st.info("No aspects match your filters.")
        st.stop()

    aspect_summary = (
        filtered_aspects.groupby(
            ["normalized_aspect", "category"]
        )
        .agg(
            mentions=("review_id", "nunique"),
            positive=(
                "aspect_sentiment",
                lambda x: (x == "Positive").sum()
            ),
            neutral=(
                "aspect_sentiment",
                lambda x: (x == "Neutral").sum()
            ),
            negative=(
                "aspect_sentiment",
                lambda x: (x == "Negative").sum()
            ),
            confidence=("confidence", "mean")
        )
        .reset_index()
    )

    aspect_summary["negative_share"] = (
        aspect_summary["negative"]
        / aspect_summary["mentions"].replace(0, 1)
        * 100
    ).round(1)

    aspect_summary["positive_share"] = (
        aspect_summary["positive"]
        / aspect_summary["mentions"].replace(0, 1)
        * 100
    ).round(1)

    left, right = st.columns(2)

    with left:

        section("Most discussed features")

        top = (
            aspect_summary.groupby("normalized_aspect")["mentions"]
            .sum()
            .nlargest(12)
            .sort_values()
            .reset_index()
        )

        fig = px.bar(
            top,
            x="mentions",
            y="normalized_aspect",
            orientation="h",
            color="mentions",
            color_continuous_scale=["#394A75", ACCENT, TEAL]
        )

        fig.update_layout(coloraxis_showscale=False)

        make_chart(fig, 390)

    with right:

        section("Features with most negative feedback")

        negative = (
            aspect_summary.groupby("normalized_aspect")
            .agg(
                mentions=("mentions", "sum"),
                negative=("negative", "sum")
            )
            .reset_index()
        )

        negative["negative_share"] = (
            negative["negative"]
            / negative["mentions"].replace(0, 1)
            * 100
        )

        negative = negative[
            negative["mentions"] >= 2
        ].nlargest(12, "negative_share")

        fig = px.bar(
            negative.sort_values("negative_share"),
            x="negative_share",
            y="normalized_aspect",
            orientation="h",
            color="negative_share",
            color_continuous_scale=["#6E567B", RED]
        )

        fig.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Negative prediction share (%)"
        )

        make_chart(fig, 390)

    section("Explore an individual aspect")

    aspect_options = sorted(
        filtered_aspects["normalized_aspect"].unique()
    )

    chosen_aspect = st.selectbox(
        "Select aspect",
        aspect_options
    )

    detail = filtered_aspects[
        filtered_aspects["normalized_aspect"] == chosen_aspect
    ]

    c1, c2, c3 = st.columns(3)

    c1.metric("Mentions", len(detail))
    c2.metric(
        "Positive",
        int((detail["aspect_sentiment"] == "Positive").sum())
    )
    c3.metric(
        "Negative",
        int((detail["aspect_sentiment"] == "Negative").sum())
    )

    st.dataframe(
        detail[
            [
                "category",
                "aspect",
                "aspect_sentiment",
                "confidence",
                "source_sentence"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 4: PRODUCT COMPARISON
# ============================================================

elif page == "Product Comparison":

    hero(
        "PRODUCT INTELLIGENCE",
        "Compare customer perceptions.",
        "Compare products using the sentiment associated with their extracted aspects."
    )

    if not {"parent_asin", "asin"}.intersection(df.columns):
        st.warning("Product identifiers are unavailable in this dataset.")
        st.stop()

    product_data = df.copy()

    product_data["product_id"] = (
        product_data["parent_asin"]
        .fillna(product_data["asin"])
        .fillna("Unknown")
        .astype(str)
    )

    product_data = product_data[
        product_data["product_id"] != "Unknown"
    ]

    product_counts = (
        product_data.groupby("product_id")
        .agg(
            aspect_mentions=("review_id", "count"),
            reviews=("review_id", "nunique")
        )
        .reset_index()
    )

    product_counts = product_counts[
        product_counts["reviews"] >= 1
    ].sort_values("aspect_mentions", ascending=False)

    st.caption(
        "Product identifiers are based on Amazon ASIN/parent ASIN. "
        "This pilot may contain too few reviews per product for "
        "reliable product-level conclusions."
    )

    if len(product_counts) < 2:

        st.info("At least two identifiable products are required.")

    else:

        product_options = product_counts["product_id"].tolist()

        left, right = st.columns(2)

        with left:
            product_a = st.selectbox(
                "First product",
                product_options,
                index=0
            )

        with right:

            default_b = 1 if len(product_options) > 1 else 0

            product_b = st.selectbox(
                "Second product",
                product_options,
                index=default_b
            )

        if product_a == product_b:

            st.warning("Please select two different products.")

        else:

            comparison = product_data[
                product_data["product_id"].isin(
                    [product_a, product_b]
                )
            ]

            comparison_table = (
                comparison.groupby(
                    ["product_id", "normalized_aspect"]
                )
                .agg(
                    mentions=("review_id", "nunique"),
                    positive=(
                        "aspect_sentiment",
                        lambda x: (x == "Positive").sum()
                    ),
                    neutral=(
                        "aspect_sentiment",
                        lambda x: (x == "Neutral").sum()
                    ),
                    negative=(
                        "aspect_sentiment",
                        lambda x: (x == "Negative").sum()
                    )
                )
                .reset_index()
            )

            comparison_table["positive_share"] = (
                comparison_table["positive"]
                / comparison_table["mentions"].replace(0, 1)
                * 100
            )

            pivot = comparison_table.pivot(
                index="normalized_aspect",
                columns="product_id",
                values="positive_share"
            )

            section("Positive sentiment share by aspect")

            fig = go.Figure()

            for product in pivot.columns:

                fig.add_trace(
                    go.Bar(
                        name=product,
                        x=pivot.index,
                        y=pivot[product],
                        hovertemplate=(
                            "Aspect: %{x}<br>"
                            "Positive share: %{y:.1f}%"
                            "<extra></extra>"
                        )
                    )
                )

            fig.update_layout(
                barmode="group",
                yaxis_title="Positive prediction share (%)",
                xaxis_title="Product aspect"
            )

            make_chart(fig, 400)

            section("Comparison details")

            st.dataframe(
                comparison_table,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# PAGE 5: REVIEW EXPLORER
# ============================================================

elif page == "Review Explorer":

    hero(
        "REVIEW EXPLORER",
        "Investigate customer feedback.",
        "Search reviews, inspect individual aspects and trace model predictions to their source sentences."
    )

    show_kpis()

    explorer = filtered_aspects.copy()

    c1, c2 = st.columns([2, 1])

    with c1:

        search = st.text_input(
            "Search review or aspect",
            placeholder="Search battery, camera, screen..."
        )

    with c2:

        min_rating, max_rating = st.select_slider(
            "Star rating",
            options=[1, 2, 3, 4, 5],
            value=(1, 5)
        )

    explorer = explorer[
        explorer["rating"].between(min_rating, max_rating)
    ]

    if search:

        mask = (
            explorer["review_text"].str.contains(
                search, case=False, na=False
            )
            |
            explorer["aspect"].str.contains(
                search, case=False, na=False
            )
        )

        explorer = explorer[mask]

    section(f"{len(explorer):,} matching aspect predictions")

    if not explorer.empty:

        display = explorer[
            [
                "review_id",
                "category",
                "rating",
                "aspect",
                "aspect_sentiment",
                "confidence"
            ]
        ].sort_values("confidence")

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        section("Inspect original review")

        review_ids = explorer["review_id"].dropna().unique().tolist()

        selected_id = st.selectbox(
            "Select review",
            review_ids
        )

        selected = explorer[
            explorer["review_id"] == selected_id
        ]

        first = selected.iloc[0]

        st.markdown(f"**Category:** {first['category']}")
        st.markdown(f"**Star rating:** {first['rating']} / 5")

        with st.container(border=True):

            st.write(first["review_text"])

        section("Aspect-level predictions")

        for _, row in selected.iterrows():

            sentiment = row["aspect_sentiment"]

            color = SENTIMENT_COLORS.get(
                sentiment, MUTED
            )

            st.markdown(
                f"""
                <div class="ri-insight">
                    <b>{row['aspect']}</b>
                    <span style="color:{color};font-weight:800">
                        &nbsp; {sentiment}
                    </span>
                    <div style="color:#AAB7CC;margin-top:8px">
                        Confidence: {row['confidence']:.1%}
                    </div>
                    <div style="margin-top:10px;color:#EDF1FA">
                        {row['source_sentence']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        export_button(
            display,
            "review_explorer_results.csv"
        )

    else:

        st.info("No reviews match the selected filters.")


# ============================================================
# PAGE 6: AI INSIGHTS
# ============================================================

elif page == "AI Insights":

    hero(
        "CUSTOMER INTELLIGENCE",
        "Turn predictions into actionable observations.",
        "Automatically identify strengths, potential weaknesses and areas requiring attention."
    )

    show_kpis()

    if filtered_aspects.empty:

        st.info("No aspect predictions are available.")

    else:

        grouped = (
            filtered_aspects.groupby("normalized_aspect")
            .agg(
                mentions=("review_id", "nunique"),
                positive=(
                    "aspect_sentiment",
                    lambda x: (x == "Positive").sum()
                ),
                neutral=(
                    "aspect_sentiment",
                    lambda x: (x == "Neutral").sum()
                ),
                negative=(
                    "aspect_sentiment",
                    lambda x: (x == "Negative").sum()
                )
            )
            .reset_index()
        )

        grouped["negative_share"] = (
            grouped["negative"]
            / grouped["mentions"].replace(0, 1)
        )

        grouped["positive_share"] = (
            grouped["positive"]
            / grouped["mentions"].replace(0, 1)
        )

        # Require a minimum number of mentions before highlighting.
        qualified = grouped[grouped["mentions"] >= 3]

        section("Potential product strengths")

        strengths = qualified[
            qualified["positive_share"] >= .60
        ].sort_values(
            ["mentions", "positive_share"],
            ascending=False
        ).head(5)

        if strengths.empty:

            insight_card(
                "No strong positive pattern detected",
                "The current filtered dataset does not contain enough "
                "repeated positive aspect predictions to highlight."
            )

        else:

            cols = st.columns(min(3, len(strengths)))

            for index, (_, row) in enumerate(strengths.iterrows()):

                with cols[index % len(cols)]:

                    insight_card(
                        f"✦ {row['normalized_aspect']}",
                        f"{int(row['mentions'])} mentions; "
                        f"{row['positive_share']:.0%} positive predictions."
                    )

        section("Potential customer pain points")

        weaknesses = qualified[
            qualified["negative_share"] >= .30
        ].sort_values(
            ["negative_share", "mentions"],
            ascending=False
        ).head(5)

        if weaknesses.empty:

            insight_card(
                "No repeated complaint pattern detected",
                "No aspect currently meets the selected negative-feedback threshold."
            )

        else:

            cols = st.columns(min(3, len(weaknesses)))

            for index, (_, row) in enumerate(weaknesses.iterrows()):

                with cols[index % len(cols)]:

                    insight_card(
                        f"⚠ {row['normalized_aspect']}",
                        f"{int(row['mentions'])} mentions; "
                        f"{row['negative_share']:.0%} negative predictions. "
                        "Worth investigating."
                    )

        section("Sentiment uncertainty")

        uncertain = filtered_aspects[
            filtered_aspects["confidence"] < .70
        ].sort_values("confidence")

        c1, c2 = st.columns(2)

        c1.metric(
            "Lower-confidence predictions",
            len(uncertain)
        )

        c2.metric(
            "Share below 70% confidence",
            f"{safe_percent(len(uncertain), len(filtered_aspects)):.1f}%"
        )

        if not uncertain.empty:

            st.dataframe(
                uncertain[
                    [
                        "category",
                        "aspect",
                        "aspect_sentiment",
                        "confidence",
                        "source_sentence"
                    ]
                ].head(15),
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "These are automated, rule-based observations from model predictions, "
            "not independent human-verified business findings. "
            "Minimum three mentions are required for aspect-level highlights."
        )

        export_button(
            grouped,
            "customer_intelligence_insights.csv",
            "Download insight data"
        )


# ============================================================
# PAGE 7: LIVE AI ANALYSIS
# ============================================================

elif page == "Live AI Analysis":

    hero(
        "LIVE AI LAB",
        "Analyze a new customer review.",
        "Extract product features and estimate the sentiment expressed toward each feature."
    )

    st.info(
        "The model runs locally. The first analysis may take a few seconds "
        "while the pretrained model is loaded."
    )

    category = st.selectbox(
        "Product category",
        ["Electronics", "Beauty", "Home_Kitchen"]
    )

    review_text = st.text_area(
        "Customer review",
        height=180,
        placeholder=(
            "Example: The camera quality is excellent, "
            "but the battery life is disappointing."
        )
    )

    if st.button(
        "✦ Analyze review",
        type="primary",
        use_container_width=True
    ):

        if not review_text.strip():

            st.warning("Please enter a customer review.")

        else:

            with st.spinner(
                "Extracting aspects and evaluating sentiment..."
            ):

                try:

                    from src.absa_sentiment import analyze_review

                    predictions = analyze_review(
                        review_text,
                        category
                    )

                    if not predictions:

                        st.warning(
                            "No product aspects were detected. "
                            "Try providing more detail about the product."
                        )

                    else:

                        st.success(
                            f"Successfully analyzed {len(predictions)} aspects."
                        )

                        output = pd.DataFrame(predictions)

                        st.dataframe(
                            output[
                                [
                                    "aspect",
                                    "sentiment",
                                    "confidence",
                                    "sentence"
                                ]
                            ],
                            use_container_width=True,
                            hide_index=True
                        )

                        fig = px.bar(
                            output,
                            x="aspect",
                            y="confidence",
                            color="sentiment",
                            color_discrete_map=SENTIMENT_COLORS,
                            range_y=[0, 1],
                            labels={
                                "aspect": "Product aspect",
                                "confidence": "Model confidence",
                                "sentiment": "Predicted sentiment"
                            }
                        )

                        make_chart(fig, 330)

                        export_button(
                            output,
                            "live_review_analysis.csv",
                            "Download prediction results"
                        )

                except Exception as error:

                    st.error(f"Analysis failed: {error}")


# ============================================================
# 8. FOOTER
# ============================================================

st.markdown(
    """
    <div class="ri-footer">
        ✦ Review Intelligence Platform
        &nbsp; | &nbsp; Amazon Reviews 2023
        &nbsp; | &nbsp; Aspect-Based Sentiment Analysis
        <br><br>
        Research prototype · Predictions are model-generated
        and should not be treated as verified ground truth.
    </div>
    """,
    unsafe_allow_html=True
)
