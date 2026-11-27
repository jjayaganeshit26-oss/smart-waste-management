
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Smart Waste Management",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Smart Waste Management & Recycling Analytics")
st.caption("Big Data Analytics and Machine Learning Prototype")

# Generate demo data if no official dataset is uploaded
def generate_demo_data():
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=180)
    wards = ["Ward 1", "Ward 2", "Ward 3", "Ward 4", "Ward 5"]
    types = ["Organic", "Plastic", "Paper", "Metal"]

    base = {
        "Ward 1": 200, "Ward 2": 300, "Ward 3": 150,
        "Ward 4": 250, "Ward 5": 180
    }
    factor = {
        "Organic": 1.5, "Plastic": 0.8,
        "Paper": 0.6, "Metal": 0.3
    }
    rate = {
        "Organic": 0.20, "Plastic": 0.65,
        "Paper": 0.75, "Metal": 0.85
    }

    rows = []
    for date in dates:
        for ward in wards:
            for waste_type in types:
                amount = max(
                    10,
                    base[ward] * factor[waste_type]
                    + np.random.normal(0, 20)
                )
                rows.append({
                    "Date": date,
                    "Ward": ward,
                    "Waste_Type": waste_type,
                    "Collected_KG": round(amount, 2),
                    "Vehicle_Trips": max(1, int(np.ceil(amount / 100))),
                    "Recycled_KG": round(amount * rate[waste_type], 2)
                })

    return pd.DataFrame(rows)

# Dataset upload
uploaded = st.sidebar.file_uploader(
    "Upload municipal waste CSV",
    type=["csv"]
)

if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Uploaded dataset loaded")
else:
    df = generate_demo_data()
    st.sidebar.info("Demo dataset in use (simulated data)")

required = [
    "Date", "Ward", "Waste_Type",
    "Collected_KG", "Vehicle_Trips", "Recycled_KG"
]

if not all(col in df.columns for col in required):
    st.error("Dataset is missing required columns.")
    st.stop()

# Data preprocessing
df = df[required].copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

for col in ["Collected_KG", "Vehicle_Trips", "Recycled_KG"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Date", "Ward", "Waste_Type"])
df = df.drop_duplicates()
df = df.dropna(subset=["Collected_KG", "Vehicle_Trips", "Recycled_KG"])

df = df[
    (df["Collected_KG"] >= 0) &
    (df["Recycled_KG"] >= 0) &
    (df["Recycled_KG"] <= df["Collected_KG"]) &
    (df["Vehicle_Trips"] >= 0)
]

if df.empty:
    st.error("No valid records found in the dataset.")
    st.stop()

# Dashboard filters
st.sidebar.header("Filters")

wards = sorted(df["Ward"].astype(str).unique())
selected_wards = st.sidebar.multiselect(
    "Select wards",
    wards,
    default=wards
)

view = df[df["Ward"].astype(str).isin(selected_wards)]

if view.empty:
    st.warning("Select at least one ward.")
    st.stop()

# Key performance indicators
total_waste = view["Collected_KG"].sum()
total_recycled = view["Recycled_KG"].sum()
recycling_rate = total_recycled / total_waste * 100 if total_waste else 0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Waste", f"{total_waste:,.0f} kg")
c2.metric("Recycled Waste", f"{total_recycled:,.0f} kg")
c3.metric("Recycling Rate", f"{recycling_rate:.1f}%")
c4.metric("Total Records", f"{len(view):,}")

st.divider()

# Ward analytics
st.subheader("1. Ward-wise Waste Generation")
ward_data = view.groupby("Ward")["Collected_KG"].sum().sort_values(ascending=False)
st.bar_chart(ward_data)

# Waste type analytics
st.subheader("2. Waste Type Distribution")
type_data = view.groupby("Waste_Type")["Collected_KG"].sum()
st.bar_chart(type_data)

# Recycling analytics
st.subheader("3. Recycling Analysis")
recycling_data = pd.DataFrame({
    "Quantity (kg)": [
        total_recycled,
        max(0, total_waste - total_recycled)
    ]
}, index=["Recycled", "Not yet recycled"])
st.bar_chart(recycling_data)

# Machine learning forecasting
st.subheader("4. Machine Learning: Seven-Day Forecast")

daily = df.groupby(["Date", "Ward"], as_index=False)["Collected_KG"].sum()

daily["Day"] = daily["Date"].dt.day
daily["Month"] = daily["Date"].dt.month
daily["Weekday"] = daily["Date"].dt.dayofweek
daily["DateNumber"] = (
    daily["Date"] - daily["Date"].min()
).dt.days

if daily["Date"].nunique() >= 10 and daily["Ward"].nunique() >= 1:
    split = int(daily["Date"].nunique() * 0.8)
    split_date = sorted(daily["Date"].unique())[split]

    train = daily[daily["Date"] < split_date]
    test = daily[daily["Date"] >= split_date]

    if len(train) > 0 and len(test) > 0:
        X_train = pd.get_dummies(
            train[["Ward", "Day", "Month", "Weekday", "DateNumber"]],
            columns=["Ward"]
        )

        X_test = pd.get_dummies(
            test[["Ward", "Day", "Month", "Weekday", "DateNumber"]],
            columns=["Ward"]
        ).reindex(columns=X_train.columns, fill_value=0)

        model = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        model.fit(X_train, train["Collected_KG"])

        predictions = model.predict(X_test)

        mae = mean_absolute_error(test["Collected_KG"], predictions)
        rmse = np.sqrt(mean_squared_error(test["Collected_KG"], predictions))
        r2 = r2_score(test["Collected_KG"], predictions)

        m1, m2, m3 = st.columns(3)
        m1.metric("MAE", f"{mae:.2f} kg")
        m2.metric("RMSE", f"{rmse:.2f} kg")
        m3.metric("R² Score", f"{r2:.3f}")

        future_dates = pd.date_range(
            daily["Date"].max() + pd.Timedelta(days=1),
            periods=7
        )

        future = pd.DataFrame([
            {
                "Date": date,
                "Ward": ward,
                "Day": date.day,
                "Month": date.month,
                "Weekday": date.dayofweek,
                "DateNumber": (date - daily["Date"].min()).days
            }
            for ward in selected_wards
            for date in future_dates
        ])

        X_future = pd.get_dummies(
            future[["Ward", "Day", "Month", "Weekday", "DateNumber"]],
            columns=["Ward"]
        ).reindex(columns=X_train.columns, fill_value=0)

        future["Predicted_Waste_KG"] = model.predict(X_future).round(2)

        st.dataframe(future, use_container_width=True)
        st.line_chart(
            future.pivot(
                index="Date",
                columns="Ward",
                values="Predicted_Waste_KG"
            )
        )

        st.download_button(
            "Download Forecast CSV",
            future.to_csv(index=False),
            "waste_forecast.csv",
            "text/csv"
        )
    else:
        st.warning("Not enough data for model evaluation.")
else:
    st.warning("More historical data is needed for forecasting.")

# Recommendations
st.subheader("5. Smart Recommendations")

ward_stats = view.groupby("Ward").agg(
    Waste=("Collected_KG", "sum"),
    Recycled=("Recycled_KG", "sum")
)

average = ward_stats["Waste"].mean()
ward_stats["Recycling_Rate"] = (
    ward_stats["Recycled"] / ward_stats["Waste"] * 100
)

for ward, row in ward_stats.iterrows():
    st.markdown(f"**{ward}**")
    if row["Waste"] > average:
        st.write("- Review collection frequency in this high-waste ward.")
    else:
        st.write("- Maintain the existing collection schedule.")

    if row["Recycling_Rate"] < 50:
        st.write("- Improve waste segregation and recycling awareness.")
    else:
        st.write("- Maintain and improve recycling practices.")

# Dataset
st.subheader("6. Processed Waste Dataset")
st.dataframe(view, use_container_width=True)

st.download_button(
    "Download Processed Dataset",
    view.to_csv(index=False),
    "processed_waste_data.csv",
    "text/csv"
)

st.success("Smart Waste Management Dashboard is ready!")


import streamlit as st
import pandas as pd
import os

# File used to store manually entered records
MANUAL_DATA_FILE = "manual_waste_data.csv"

# Initialize storage
if not os.path.exists(MANUAL_DATA_FILE):
    columns = [
        "Ward",
        "Date",
        "Waste Type",
        "Total Waste (kg)",
        "Recycled Waste (kg)"
    ]
    pd.DataFrame(columns=columns).to_csv(
        MANUAL_DATA_FILE, index=False
    )

st.sidebar.markdown("---")
st.sidebar.subheader("✍️ Manual Waste Data Entry")

with st.sidebar.form("manual_waste_form"):

    ward = st.selectbox(
        "Select Ward",
        ["Ward 1", "Ward 2", "Ward 3", "Ward 4", "Ward 5"]
    )

    collection_date = st.date_input(
        "Collection Date"
    )

    waste_type = st.selectbox(
        "Waste Type",
        ["Plastic", "Paper", "Organic", "Metal", "Glass", "Other"]
    )

    total_waste = st.number_input(
        "Total Waste (kg)",
        min_value=0.0,
        value=500.0,
        step=10.0
    )

    recycled_waste = st.number_input(
        "Recycled Waste (kg)",
        min_value=0.0,
        value=250.0,
        step=10.0
    )

    submitted = st.form_submit_button(
        "➕ Save Waste Record",
        use_container_width=True
    )

    if submitted:

        if total_waste <= 0:
            st.error("Total waste must be greater than zero.")

        elif recycled_waste > total_waste:
            st.error(
                "Recycled waste cannot exceed total waste."
            )

        else:
            new_record = pd.DataFrame([{
                "Ward": ward,
                "Date": str(collection_date),
                "Waste Type": waste_type,
                "Total Waste (kg)": total_waste,
                "Recycled Waste (kg)": recycled_waste
            }])

            new_record.to_csv(
                MANUAL_DATA_FILE,
                mode="a",
                header=False,
                index=False
            )

            st.success("Waste record saved successfully!")

            st.rerun()


# Load manually entered records
manual_df = pd.read_csv(MANUAL_DATA_FILE)

# Combine uploaded/demo data and manual records
# Your existing main dataset should be named df
if not manual_df.empty:
    df = pd.concat(
        [df, manual_df],
        ignore_index=True
    )

# Convert numeric columns
df["Total Waste (kg)"] = pd.to_numeric(
    df["Total Waste (kg)"], errors="coerce"
).fillna(0)

df["Recycled Waste (kg)"] = pd.to_numeric(
    df["Recycled Waste (kg)"], errors="coerce"
).fillna(0)

# Calculate analytics
total_waste = df["Total Waste (kg)"].sum()

recycled_waste = df["Recycled Waste (kg)"].sum()

recycling_rate = (
    recycled_waste / total_waste * 100
    if total_waste > 0 else 0
)

total_records = len(df)

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Waste",
    f"{total_waste:,.0f} kg"
)

col2.metric(
    "Recycled Waste",
    f"{recycled_waste:,.0f} kg"
)

col3.metric(
    "Recycling Rate",
    f"{recycling_rate:.1f}%"
)

col4.metric(
    "Total Records",
    f"{total_records:,}"
)


# Ward-wise waste generation
st.subheader("Ward-wise Waste Generation")

ward_data = df.groupby("Ward")[
    "Total Waste (kg)"
].sum()

st.bar_chart(ward_data)

# Waste type analysis
st.subheader("Waste Type Analysis")

waste_type_data = df.groupby("Waste Type")[
    "Total Waste (kg)"
].sum()

st.bar_chart(waste_type_data)

# Recycled vs non-recycled waste
st.subheader("Recycling Analysis")

df["Non-Recycled Waste (kg)"] = (
    df["Total Waste (kg)"] -
    df["Recycled Waste (kg)"]
)

recycling_data = df[
    ["Recycled Waste (kg)", "Non-Recycled Waste (kg)"]
].sum()

st.bar_chart(recycling_data)

# Display all records
st.subheader("Waste Collection Records")

st.dataframe(df, use_container_width=True)



