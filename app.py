from flask import Flask, request, render_template_string, redirect, url_for
import os
import io
import base64
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = Flask(__name__)

# Keep this CSV in the same folder as app.py
DATA_FILE = "sales_data_large.csv"


# =========================================================
# DATA LOADING
# =========================================================

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"{DATA_FILE} not found. Keep sales_data_large.csv beside app.py."
        )

    df = pd.read_csv(DATA_FILE)
    df.columns = df.columns.str.strip()

    # Support older dataset column names too
    if "Sales_Amount" not in df.columns and "Sales" in df.columns:
        df["Sales_Amount"] = pd.to_numeric(df["Sales"], errors="coerce")

    if "Cost_Amount" not in df.columns:
        if "Unit_Cost" in df.columns and "Quantity" in df.columns:
            df["Cost_Amount"] = (
                pd.to_numeric(df["Unit_Cost"], errors="coerce")
                * pd.to_numeric(df["Quantity"], errors="coerce")
            )
        else:
            df["Cost_Amount"] = 0.0

    if "Mode" not in df.columns:
        df["Mode"] = "Online"

    required = [
        "Date",
        "Mode",
        "Product",
        "Sales_Amount",
        "Cost_Amount",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Mode"] = df["Mode"].fillna("").astype(str).str.strip()
    df["Product"] = df["Product"].fillna("").astype(str).str.strip()
    df["Sales_Amount"] = pd.to_numeric(
        df["Sales_Amount"], errors="coerce"
    )
    df["Cost_Amount"] = pd.to_numeric(
        df["Cost_Amount"], errors="coerce"
    )

    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    if "Review_Count" in df.columns:
        df["Review_Count"] = pd.to_numeric(
            df["Review_Count"], errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Date",
            "Product",
            "Sales_Amount",
            "Cost_Amount",
        ]
    ).copy()

    return df


# =========================================================
# CHART HELPERS
# =========================================================

def fig_to_base64(fig):
    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        bbox_inches="tight",
        dpi=120,
    )

    buffer.seek(0)

    image = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    plt.close(fig)
    return image


def make_sales_chart(df):
    daily = (
        df.groupby("Date")["Sales_Amount"]
        .sum()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(
        daily.index,
        daily.values,
        marker="o",
        linewidth=2,
    )

    ax.set_title("Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales Amount")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig_to_base64(fig)


def make_cost_chart(df):
    daily = (
        df.groupby("Date")["Cost_Amount"]
        .sum()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(
        daily.index,
        daily.values,
        marker="o",
        linewidth=2,
    )

    ax.set_title("Cost Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cost Amount")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig_to_base64(fig)


def make_rating_chart(df):
    if "Rating" not in df.columns:
        return ""

    work = df.dropna(subset=["Rating"]).copy()
    if work.empty:
        return ""

    if "Review_Count" in work.columns:
        values = (
            work.groupby("Rating")["Review_Count"]
            .sum()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
        )
        y_label = "Number of Reviews"
    else:
        values = (
            work["Rating"]
            .value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
        )
        y_label = "Review Count"

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(
        values.index.astype(str),
        values.values,
    )

    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel(y_label)

    fig.tight_layout()
    return fig_to_base64(fig)


def make_rating_trend_chart(df):
    if "Rating" not in df.columns:
        return ""

    work = df.dropna(subset=["Rating"]).copy()
    if work.empty:
        return ""

    monthly = (
        work.set_index("Date")["Rating"]
        .resample("ME")
        .mean()
        .dropna()
    )

    if monthly.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(
        monthly.index,
        monthly.values,
        marker="o",
        linewidth=2,
    )

    ax.set_title("Average Rating Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Rating")
    ax.set_ylim(1, 5)
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig_to_base64(fig)


# =========================================================
# PERIOD
# =========================================================

def filter_period(df, period_code):
    latest = df["Date"].max()

    if period_code == "1M":
        start = latest - pd.DateOffset(months=1)
        return df[df["Date"] >= start].copy(), "1 Month"

    if period_code == "6M":
        start = latest - pd.DateOffset(months=6)
        return df[df["Date"] >= start].copy(), "6 Months"

    start = latest - pd.DateOffset(years=1)
    return df[df["Date"] >= start].copy(), "1 Year"


# =========================================================
# SINGLE-PAGE HTML TEMPLATE
# =========================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Sales and Data Analysis System</title>

<style>
* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #f4f6f8;
    color: #1f2937;
}

.container {
    width: 92%;
    max-width: 1120px;
    margin: 30px auto;
}

.header {
    background: #111827;
    color: white;
    text-align: center;
    padding: 28px 20px;
}

.header h1 {
    margin: 0;
    font-size: 30px;
}

.header p {
    margin: 8px 0 0;
    opacity: 0.86;
}


/* =========================
   OPENING
========================= */

.opening-screen {
    position: fixed;
    inset: 0;
    background: #000;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
}

.open-button {
    background: #7dd3fc;
    color: #082f49;
    border: none;
    border-radius: 10px;
    padding: 18px 42px;
    font-size: 18px;
    font-weight: bold;
    letter-spacing: 1px;
    cursor: pointer;
    box-shadow: 0 0 30px rgba(125, 211, 252, 0.35);
}

.open-button:hover {
    background: #38bdf8;
}


/* =========================
   TITLE
========================= */

.title-screen {
    position: fixed;
    inset: 0;
    background: #000;
    color: white;
    display: none;
    align-items: center;
    justify-content: center;
    text-align: center;
    z-index: 9998;
}

.title-screen h1 {
    margin: 0;
    font-size: 46px;
    letter-spacing: 4px;
    padding: 20px;
}


/* =========================
   PAGE 1
========================= */

.selection-card {
    max-width: 680px;
    margin: 55px auto;
}

.selection-card h2 {
    text-align: center;
    margin-top: 0;
}

.subtitle {
    text-align: center;
    color: #6b7280;
}

.field {
    margin: 26px 0;
}

.field > label {
    display: block;
    font-weight: bold;
    margin-bottom: 9px;
}

.mode-row {
    display: flex;
    justify-content: center;
    gap: 18px;
}

.mode-box {
    border: 1px solid #d1d5db;
    background: #fafafa;
    padding: 15px 28px;
    border-radius: 10px;
}

.mode-box label {
    cursor: pointer;
    font-weight: bold;
}

select {
    width: 100%;
    padding: 12px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: white;
    font-size: 15px;
}

.design-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}

.design-box {
    border: 1px solid #d1d5db;
    padding: 14px;
    border-radius: 9px;
    text-align: center;
    cursor: pointer;
}

.selection-note {
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-top: 18px;
}


/* =========================
   PAGE 2
========================= */

.page-title {
    text-align: center;
}

.selected-info {
    text-align: center;
    color: #6b7280;
    margin-bottom: 25px;
}

.metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 18px;
}

.metric {
    background: white;
    padding: 24px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 3px 14px rgba(0, 0, 0, 0.07);
}

.metric-title {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 23px;
    font-weight: bold;
}

.period-card {
    background: white;
    padding: 25px;
    border-radius: 14px;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0 3px 14px rgba(0, 0, 0, 0.07);
}

.period-buttons {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 20px;
}

.period-button {
    text-decoration: none;
    color: #111827;
    background: white;
    border: 1px solid #d1d5db;
    padding: 11px 22px;
    border-radius: 8px;
    font-weight: bold;
}

.period-button:hover,
.period-button.active {
    background: #2563eb;
    color: white;
    border-color: #2563eb;
}

.chart-card {
    background: white;
    padding: 25px;
    border-radius: 14px;
    margin-top: 25px;
    text-align: center;
    box-shadow: 0 3px 14px rgba(0, 0, 0, 0.07);
}

.chart-card img {
    width: 100%;
    max-width: 900px;
    height: auto;
}


/* =========================
   REVIEW PAGE
========================= */

.review-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
}

.rating-number {
    font-size: 30px;
    font-weight: bold;
}

.stars {
    font-size: 24px;
    margin-top: 5px;
}

.review-card {
    background: white;
    padding: 25px;
    border-radius: 14px;
    margin-top: 25px;
    box-shadow: 0 3px 14px rgba(0, 0, 0, 0.07);
}

.review-table-wrapper {
    overflow-x: auto;
}

.review-table {
    width: 100%;
    border-collapse: collapse;
}

.review-table th,
.review-table td {
    border: 1px solid #ddd;
    padding: 10px;
    text-align: left;
}

.review-table th {
    background: #f0f2f5;
}

.review-button,
.back-button {
    display: block;
    width: max-content;
    margin: 30px auto;
    padding: 13px 26px;
    border-radius: 8px;
    text-decoration: none;
    font-weight: bold;
}

.review-button {
    background: #111827;
    color: white;
}

.back-button {
    background: #e5e7eb;
    color: #111827;
}


/* =========================
   FOOTER
========================= */

footer {
    background: #111827;
    color: white;
    text-align: center;
    padding: 22px;
    margin-top: 40px;
}


/* =========================
   RESPONSIVE
========================= */

@media (max-width: 850px) {
    .metrics {
        grid-template-columns: repeat(2, 1fr);
    }

    .review-metrics {
        grid-template-columns: 1fr;
    }

    .design-row {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 550px) {
    .metrics {
        grid-template-columns: 1fr;
    }

    .mode-row {
        flex-direction: column;
    }

    .header h1 {
        font-size: 23px;
    }
}
</style>

</head>


<body>


<!-- =====================================================
     OPENING SCREEN
====================================================== -->

{% if show_opening %}

<div
    id="openingScreen"
    class="opening-screen"
>

    <button
        id="openButton"
        class="open-button"
        type="button"
    >
        CLICK ME TO OPEN
    </button>

</div>


<div
    id="titleScreen"
    class="title-screen"
>

    <h1>
        SALES AND DATA ANALYSIS SYSTEM
    </h1>

</div>

{% endif %}


<!-- =====================================================
     MAIN APPLICATION
====================================================== -->

<div id="mainPage">


<header class="header">

    <h1>
        SALES AND DATA ANALYSIS SYSTEM
    </h1>

    <p>
        Online & Offline Shopping Analysis
    </p>

</header>


<div class="container">


{% if error %}

<div class="error">
    {{ error }}
</div>

{% endif %}


<!-- =====================================================
     PAGE 1
====================================================== -->

{% if page == "selection" %}

<section class="card selection-card">

    <h2>
        Product Selection
    </h2>

    <p class="subtitle">
        Select Mode, Product and Design
    </p>


    <form
        id="selectionForm"
        method="POST"
    >


        <!-- MODE -->

        <div class="field">

            <label>
                Mode
            </label>


            <div class="mode-row">


                <div class="mode-box">

                    <label>

                        <input
                            class="mode-check"
                            type="checkbox"
                            name="mode"
                            value="Online"
                        >

                        Online

                    </label>

                </div>


                <div class="mode-box">

                    <label>

                        <input
                            class="mode-check"
                            type="checkbox"
                            name="mode"
                            value="Offline"
                        >

                        Offline

                    </label>

                </div>


            </div>

        </div>


        <!-- PRODUCT -->

        <div class="field">

            <label>
                Product
            </label>


            <select
                id="product"
                name="product"
                required
            >

                <option value="">
                    Select Product
                </option>


                {% for product in products %}

                <option value="{{ product }}">
                    {{ product }}
                </option>

                {% endfor %}


            </select>

        </div>


        <!-- DESIGN -->

        <div class="field">

            <label>
                Design
            </label>


            <div class="design-row">


                <label class="design-box">

                    <input
                        type="radio"
                        name="design"
                        value="Classic"
                        checked
                    >

                    Classic

                </label>


                <label class="design-box">

                    <input
                        type="radio"
                        name="design"
                        value="Modern"
                    >

                    Modern

                </label>


                <label class="design-box">

                    <input
                        type="radio"
                        name="design"
                        value="Minimal"
                    >

                    Minimal

                </label>


            </div>

        </div>


        <p class="selection-note">
            Select Mode, Product and Design to continue automatically.
        </p>


    </form>

</section>

{% endif %}


<!-- =====================================================
     PAGE 2
====================================================== -->

{% if page == "analysis" %}

<div class="page-title">

    <h2>
        {{ product }}
    </h2>

</div>


<div class="selected-info">

    Mode:
    <strong>{{ mode }}</strong>

    &nbsp; | &nbsp;

    Period:
    <strong>{{ period }}</strong>

    &nbsp; | &nbsp;

    Design:
    <strong>{{ design }}</strong>

</div>


<div class="metrics">


    <div class="metric">

        <div class="metric-title">
            Total Sales
        </div>

        <div class="metric-value">
            ₹{{ total_sales }}
        </div>

    </div>


    <div class="metric">

        <div class="metric-title">
            Total Cost
        </div>

        <div class="metric-value">
            ₹{{ total_cost }}
        </div>

    </div>


    <div class="metric">

        <div class="metric-title">
            Average Sales
        </div>

        <div class="metric-value">
            ₹{{ average_sales }}
        </div>

    </div>


    <div class="metric">

        <div class="metric-title">
            Average Cost
        </div>

        <div class="metric-value">
            ₹{{ average_cost }}
        </div>

    </div>


</div>


<section class="period-card">

    <h2>
        Time Period
    </h2>


    <div class="period-buttons">


        <a
            class="period-button {% if period == '1 Month' %}active{% endif %}"
            href="{{ url_for(
                'analysis',
                mode=mode,
                product=product,
                design=design,
                period='1M'
            ) }}"
        >
            1 Month
        </a>


        <a
            class="period-button {% if period == '6 Months' %}active{% endif %}"
            href="{{ url_for(
                'analysis',
                mode=mode,
                product=product,
                design=design,
                period='6M'
            ) }}"
        >
            6 Months
        </a>


        <a
            class="period-button {% if period == '1 Year' %}active{% endif %}"
            href="{{ url_for(
                'analysis',
                mode=mode,
                product=product,
                design=design,
                period='1Y'
            ) }}"
        >
            1 Year
        </a>


    </div>

</section>


<section class="chart-card">

    <h2>
        📈 Sales Visualization
    </h2>

    <img
        src="data:image/png;base64,{{ sales_image }}"
        alt="Sales Visualization"
    >

</section>


<section class="chart-card">

    <h2>
        📊 Cost Visualization
    </h2>

    <img
        src="data:image/png;base64,{{ cost_image }}"
        alt="Cost Visualization"
    >

</section>


<a
    class="review-button"
    href="{{ url_for(
        'reviews',
        mode=mode,
        product=product,
        design=design
    ) }}"
>
    Review & Rating
</a>


<a
    class="back-button"
    href="{{ url_for('home') }}"
>
    Back to Selection
</a>


{% endif %}


<!-- =====================================================
     PAGE 3 - REVIEW & RATING
====================================================== -->

{% if page == "reviews" %}

<div class="page-title">

    <h2>
        Review & Rating
    </h2>

</div>


<div class="selected-info">

    Product:
    <strong>{{ product }}</strong>

    &nbsp; | &nbsp;

    Mode:
    <strong>{{ mode }}</strong>

    &nbsp; | &nbsp;

    Design:
    <strong>{{ design }}</strong>

</div>


<div class="review-metrics">


    <div class="metric">

        <div class="metric-title">
            Average Rating
        </div>

        <div class="rating-number">
            {{ average_rating }}
        </div>

        <div class="stars">
            ★★★★★
        </div>

    </div>


    <div class="metric">

        <div class="metric-title">
            Total Reviews
        </div>

        <div class="metric-value">
            {{ total_reviews }}
        </div>

    </div>


    <div class="metric">

        <div class="metric-title">
            Positive Reviews
        </div>

        <div class="metric-value">
            {{ positive_percentage }}%
        </div>

    </div>


</div>


<section class="chart-card">

    <h2>
        ⭐ Rating Distribution
    </h2>


    {% if rating_image %}

    <img
        src="data:image/png;base64,{{ rating_image }}"
        alt="Rating Distribution"
    >

    {% else %}

    <p>
        Rating data is not available.
    </p>

    {% endif %}

</section>


<section class="chart-card">

    <h2>
        📈 Average Rating Trend
    </h2>


    {% if rating_trend_image %}

    <img
        src="data:image/png;base64,{{ rating_trend_image }}"
        alt="Average Rating Trend"
    >

    {% else %}

    <p>
        Rating trend is not available.
    </p>

    {% endif %}

</section>


<section class="review-card">

    <h2>
        Customer Review Summary
    </h2>


    <p>
        Average rating:
        <strong>{{ average_rating }}</strong>
        out of 5
    </p>


    <p>
        Total reviews:
        <strong>{{ total_reviews }}</strong>
    </p>


    <p>
        4-star and 5-star reviews:
        <strong>{{ positive_percentage }}%</strong>
    </p>

</section>


<section class="review-card">

    <h2>
        Customer Reviews
    </h2>


    <div class="review-table-wrapper">

        {{ review_table | safe }}

    </div>

</section>


<a
    class="back-button"
    href="{{ url_for(
        'analysis',
        mode=mode,
        product=product,
        design=design,
        period='1Y'
    ) }}"
>
    Back to Sales Analysis
</a>


{% endif %}


</div>


<footer>
    Sales and Data Analysis System
</footer>


</div>


<script>

/* =====================================================
   OPENING
===================================================== */

document.addEventListener(
    "DOMContentLoaded",
    function () {

        const openButton =
            document.getElementById(
                "openButton"
            );

        const openingScreen =
            document.getElementById(
                "openingScreen"
            );

        const titleScreen =
            document.getElementById(
                "titleScreen"
            );

        const mainPage =
            document.getElementById(
                "mainPage"
            );


        if (
            openButton &&
            openingScreen &&
            titleScreen &&
            mainPage
        ) {

            openButton.onclick =
                function () {

                    openingScreen.style.display =
                        "none";

                    titleScreen.style.display =
                        "flex";


                    /*
                     * Title appears once.
                     * Page 1 opens automatically.
                     */

                    setTimeout(
                        function () {

                            titleScreen.style.display =
                                "none";

                            mainPage.style.display =
                                "block";

                            window.scrollTo(
                                0,
                                0
                            );

                        },
                        1500
                    );

                };

        }


        /* =================================================
           PAGE 1

           No Confirm button.
           When Mode + Product + Design are selected,
           form automatically submits.
        ================================================= */

        const form =
            document.getElementById(
                "selectionForm"
            );

        const product =
            document.getElementById(
                "product"
            );

        const modeChecks =
            document.querySelectorAll(
                ".mode-check"
            );

        const designChecks =
            document.querySelectorAll(
                "input[name='design']"
            );


        let submitted = false;


        function continueAutomatically() {

            if (
                !form ||
                !product ||
                submitted
            ) {
                return;
            }


            const mode =
                document.querySelector(
                    ".mode-check:checked"
                );

            const design =
                document.querySelector(
                    "input[name='design']:checked"
                );


            if (
                mode &&
                product.value &&
                design
            ) {

                submitted = true;

                form.submit();

            }

        }


        modeChecks.forEach(
            function (box) {

                box.addEventListener(
                    "change",
                    function () {

                        if (this.checked) {

                            modeChecks.forEach(
                                function (other) {

                                    if (
                                        other !== box
                                    ) {

                                        other.checked =
                                            false;

                                    }

                                }
                            );

                        }

                        continueAutomatically();

                    }
                );

            }
        );


        if (product) {

            product.addEventListener(
                "change",
                continueAutomatically
            );

        }


        designChecks.forEach(
            function (radio) {

                radio.addEventListener(
                    "change",
                    continueAutomatically
                );

            }
        );

    }
);

</script>

</body>
</html>
"""


# =========================================================
# PAGE 1
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    try:

        df = load_data()

        products = sorted(
            df["Product"].dropna().unique().tolist(),
            key=lambda x: str(x).lower()
        )


        if request.method == "POST":

            modes = request.form.getlist("mode")

            product = request.form.get(
                "product",
                ""
            ).strip()

            design = request.form.get(
                "design",
                "Classic"
            )


            if len(modes) != 1:

                return render_template_string(
                    HTML,
                    page="selection",
                    products=products,
                    error="Please select Online or Offline.",
                    show_opening=False
                )


            if not product:

                return render_template_string(
                    HTML,
                    page="selection",
                    products=products,
                    error="Please select a product.",
                    show_opening=False
                )


            return redirect(
                url_for(
                    "analysis",
                    mode=modes[0],
                    product=product,
                    design=design,
                    period="1Y"
                )
            )


        return render_template_string(
            HTML,
            page="selection",
            products=products,
            error=None,
            show_opening=True
        )


    except Exception as e:

        return render_template_string(
            """
            <div style="
                font-family:Arial;
                text-align:center;
                padding:60px;
            ">
                <h2>Project Error</h2>
                <p>{{ error }}</p>
            </div>
            """,
            error=str(e)
        )


# =========================================================
# PAGE 2
# =========================================================

@app.route("/analysis")
def analysis():

    try:

        df = load_data()

        mode = request.args.get(
            "mode",
            "Online"
        )

        product = request.args.get(
            "product",
            ""
        )

        design = request.args.get(
            "design",
            "Classic"
        )

        period_code = request.args.get(
            "period",
            "1Y"
        )


        if mode not in [
            "Online",
            "Offline"
        ]:

            return redirect(
                url_for("home")
            )


        if not product:

            return redirect(
                url_for("home")
            )


        selected = df[
            (df["Mode"].str.lower() == mode.lower())
            &
            (df["Product"] == product)
        ].copy()


        if selected.empty:

            return render_template_string(
                HTML,
                page="selection",
                products=sorted(
                    df["Product"].unique(),
                    key=lambda x: str(x).lower()
                ),
                error="No data found for the selected product and mode.",
                show_opening=False
            )


        period_df, period_name = filter_period(
            selected,
            period_code
        )


        if period_df.empty:

            period_df = selected.copy()


        total_sales = period_df[
            "Sales_Amount"
        ].sum()


        total_cost = period_df[
            "Cost_Amount"
        ].sum()


        average_sales = period_df[
            "Sales_Amount"
        ].mean()


        average_cost = period_df[
            "Cost_Amount"
        ].mean()


        return render_template_string(

            HTML,

            page="analysis",

            show_opening=False,

            product=product,

            mode=mode,

            design=design,

            period=period_name,

            total_sales=
                f"{total_sales:,.2f}",

            total_cost=
                f"{total_cost:,.2f}",

            average_sales=
                f"{average_sales:,.2f}",

            average_cost=
                f"{average_cost:,.2f}",

            sales_image=
                make_sales_chart(
                    period_df
                ),

            cost_image=
                make_cost_chart(
                    period_df
                )

        )


    except Exception as e:

        return render_template_string(
            """
            <div style="
                font-family:Arial;
                text-align:center;
                padding:60px;
            ">
                <h2>Project Error</h2>
                <p>{{ error }}</p>
            </div>
            """,
            error=str(e)
        )


# =========================================================
# PAGE 3 - REVIEW & RATING
# =========================================================

@app.route("/reviews")
def reviews():

    try:

        df = load_data()

        mode = request.args.get(
            "mode",
            "Online"
        )

        product = request.args.get(
            "product",
            ""
        )

        design = request.args.get(
            "design",
            "Classic"
        )


        if not product:

            return redirect(
                url_for("home")
            )


        selected = df[
            (df["Mode"].str.lower() == mode.lower())
            &
            (df["Product"] == product)
        ].copy()


        if selected.empty:

            selected = df[
                df["Product"] == product
            ].copy()


        # Average rating

        if (
            "Rating" in selected.columns
            and not selected["Rating"].dropna().empty
        ):

            average_rating = float(
                selected["Rating"]
                .dropna()
                .mean()
            )

        else:

            average_rating = 0.0


        # Review count

        if "Review_Count" in selected.columns:

            total_reviews = int(
                selected["Review_Count"]
                .fillna(0)
                .sum()
            )


            if "Rating" in selected.columns:

                positive_reviews = int(
                    selected[
                        selected["Rating"] >= 4
                    ]["Review_Count"]
                    .fillna(0)
                    .sum()
                )

            else:

                positive_reviews = 0

        else:

            total_reviews = len(
                selected
            )


            if "Rating" in selected.columns:

                positive_reviews = int(
                    (
                        selected["Rating"] >= 4
                    ).sum()
                )

            else:

                positive_reviews = 0


        if total_reviews > 0:

            positive_percentage = round(
                positive_reviews
                / total_reviews
                * 100,
                2
            )

        else:

            positive_percentage = 0


        # Review table

        review_columns = [
            "Date",
            "Rating"
        ]


        if "Review_Text" in selected.columns:

            review_columns.append(
                "Review_Text"
            )


        available_columns = [
            c
            for c in review_columns
            if c in selected.columns
        ]


        review_df = (
            selected[available_columns]
            .sort_values(
                "Date",
                ascending=False
            )
            .head(20)
            .copy()
        )


        if "Date" in review_df.columns:

            review_df["Date"] = (
                review_df["Date"]
                .dt.strftime("%Y-%m-%d")
            )


        if "Rating" in review_df.columns:

            review_df["Rating"] = (
                review_df["Rating"]
                .apply(
                    lambda value:
                    f"{float(value):.1f} ★"
                )
            )


        review_table = (
            review_df.to_html(
                index=False,
                classes="review-table"
            )
        )


        return render_template_string(

            HTML,

            page="reviews",

            show_opening=False,

            product=product,

            mode=mode,

            design=design,

            average_rating=
                f"{average_rating:.2f}",

            total_reviews=
                f"{total_reviews:,}",

            positive_percentage=
                positive_percentage,

            rating_image=
                make_rating_chart(
                    selected
                ),

            rating_trend_image=
                make_rating_trend_chart(
                    selected
                ),

            review_table=
                review_table
        )


    except Exception as e:

        return render_template_string(
            """
            <div style="
                font-family:Arial;
                text-align:center;
                padding:60px;
            ">
                <h2>Project Error</h2>
                <p>{{ error }}</p>
            </div>
            """,
            error=str(e)
        )


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
