
import streamlit as st
import pandas as pd
import os
from datetime import date
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Smart Waste Management",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Smart Waste Management Dashboard")
st.caption("Manual Waste Monitoring and Recycling Analytics")

FILE = "manual_waste_data.csv"

COLUMNS = [
    "Date", "Ward", "Organic_KG",
    "Plastic_KG", "Paper_KG", "Metal_KG",
    "Recycled_KG", "Vehicle_Trips"
]

# ---------- LOAD DATA ----------

def load_data():
    if not os.path.exists(FILE):
        return pd.DataFrame(columns=COLUMNS)

    try:
        data = pd.read_csv(FILE)
        if list(data.columns) != COLUMNS:
            return pd.DataFrame(columns=COLUMNS)

        data["Date"] = pd.to_datetime(
            data["Date"], errors="coerce"
        )

        numeric_cols = COLUMNS[2:]
        for col in numeric_cols:
            data[col] = pd.to_numeric(
                data[col], errors="coerce"
            )

        data = data.dropna(subset=["Date"])
        data[numeric_cols] = data[numeric_cols].fillna(0)

        return data

    except Exception:
        st.error("Could not read the saved CSV file.")
        return pd.DataFrame(columns=COLUMNS)


df = load_data()

# ---------- MANUAL ENTRY ----------

st.sidebar.header("📝 Manual Waste Reading")

with st.sidebar.form("entry_form"):

    ward = st.selectbox(
        "Select Ward",
        [f"Ward {i}" for i in range(1, 11)]
    )

    reading_date = st.date_input(
        "Collection Date",
        value=date.today()
    )

    st.markdown("### Waste Readings (KG)")

    organic = st.number_input(
        "Organic Waste", min_value=0.0, value=0.0
    )

    plastic = st.number_input(
        "Plastic Waste", min_value=0.0, value=0.0
    )

    paper = st.number_input(
        "Paper Waste", min_value=0.0, value=0.0
    )

    metal = st.number_input(
        "Metal Waste", min_value=0.0, value=0.0
    )

    recycled = st.number_input(
        "Recycled Waste", min_value=0.0, value=0.0
    )

    trips = st.number_input(
        "Vehicle Trips", min_value=0, value=1
    )

    submit = st.form_submit_button("Save Waste Reading")

if submit:
    total = organic + plastic + paper + metal

    if total <= 0:
        st.sidebar.error("Enter waste quantity.")

    elif recycled > total:
        st.sidebar.error(
            "Recycled waste cannot exceed total waste."
        )

    elif (
        (df["Ward"] == ward) &
        (df["Date"].dt.date == reading_date)
    ).any():
        st.sidebar.error(
            "A reading already exists for this ward and date."
        )

    else:
        new_row = pd.DataFrame([{
            "Date": pd.Timestamp(reading_date),
            "Ward": ward,
            "Organic_KG": organic,
            "Plastic_KG": plastic,
            "Paper_KG": paper,
            "Metal_KG": metal,
            "Recycled_KG": recycled,
            "Vehicle_Trips": trips
        }])

        updated = pd.concat(
            [df, new_row], ignore_index=True
        )

        updated.to_csv(FILE, index=False)

        st.sidebar.success("Reading saved successfully!")
        st.rerun()

# ---------- DASHBOARD ----------

df = load_data()

if df.empty:
    st.info(
        "No waste readings available. Enter your first reading "
        "using the sidebar."
    )
else:

    df["Total_Waste_KG"] = (
        df["Organic_KG"] +
        df["Plastic_KG"] +
        df["Paper_KG"] +
        df["Metal_KG"]
    )

    df["Overall_Wastage_KG"] = (
        df["Total_Waste_KG"] - df["Recycled_KG"]
    )

    total_waste = df["Total_Waste_KG"].sum()
    recycled_total = df["Recycled_KG"].sum()
    organic_total = df["Organic_KG"].sum()
    remaining = df["Overall_Wastage_KG"].sum()

    recycling_rate = (
        recycled_total / total_waste * 100
        if total_waste > 0 else 0
    )

    st.header("📊 Overall Waste Statistics")

    a, b, c, d = st.columns(4)

    a.metric("Total Waste", f"{total_waste:.2f} KG")
    b.metric("Recycled Waste", f"{recycled_total:.2f} KG")
    c.metric("Organic Waste", f"{organic_total:.2f} KG")
    d.metric("Active Wards", df["Ward"].nunique())

    e, f = st.columns(2)

    e.metric("Overall Wastage", f"{remaining:.2f} KG")
    f.metric("Recycling Rate", f"{recycling_rate:.2f}%")

    st.divider()

    # WASTE COMPOSITION

    st.subheader("1. Waste Composition")

    composition = pd.Series({
        "Organic": organic_total,
        "Plastic": df["Plastic_KG"].sum(),
        "Paper": df["Paper_KG"].sum(),
        "Metal": df["Metal_KG"].sum()
    })

    st.bar_chart(composition)

    # WARD MONITORING

    st.subheader("2. Ward-wise Waste Monitoring")

    ward_data = df.groupby("Ward").agg(
        Total_Waste=("Total_Waste_KG", "sum"),
        Recycled=("Recycled_KG", "sum")
    )

    ward_data["Remaining"] = (
        ward_data["Total_Waste"] - ward_data["Recycled"]
    )

    st.bar_chart(ward_data[["Total_Waste"]])
    st.dataframe(ward_data, use_container_width=True)

    # OVERALL WASTE TREND

    st.subheader("3. Overall Waste Analysis")

    daily = (
        df.groupby("Date")["Total_Waste_KG"]
        .sum()
        .sort_index()
    )

    st.line_chart(daily)

    # FORECAST

    st.subheader("4. Seven-Day Waste Forecast")

    daily_df = daily.reset_index()
    daily_df.columns = ["Date", "Waste"]

    if len(daily_df) >= 7:

        daily_df["DayNumber"] = (
            daily_df["Date"] - daily_df["Date"].min()
        ).dt.days

        model = RandomForestRegressor(
            n_estimators=50,
            random_state=42
        )

        model.fit(
            daily_df[["DayNumber"]],
            daily_df["Waste"]
        )

        future_dates = pd.date_range(
            daily_df["Date"].max() + pd.Timedelta(days=1),
            periods=7
        )

        future_days = (
            future_dates - daily_df["Date"].min()
        ).days

        predictions = model.predict(
            pd.DataFrame({"DayNumber": future_days})
        )

        forecast = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Waste_KG": predictions.round(2)
        })

        st.dataframe(forecast, use_container_width=True)

        st.line_chart(
            forecast.set_index("Date")
        )

    else:
        st.warning(
            "Enter readings for at least 7 different dates "
            "to generate a forecast."
        )

    # RECORD TABLE

    st.subheader("5. Manual Waste Records")

    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download Waste Report",
        df.to_csv(index=False),
        "waste_report.csv",
        "text/csv"
    )

    # DELETE RECORDS BY DATE

    st.subheader("6. Delete Records by Date")

    delete_date = st.selectbox(
        "Select date",
        sorted(df["Date"].dt.date.unique()),
        key="delete_date"
    )

    if st.button("Delete Selected Date Records"):

        updated = df[
            df["Date"].dt.date != delete_date
        ][COLUMNS]

        updated.to_csv(FILE, index=False)

        st.success("Records deleted successfully!")
        st.rerun()

    # RECOMMENDATIONS

    st.subheader("7. Smart Recommendations")

    average = ward_data["Total_Waste"].mean()

    for ward_name, row in ward_data.iterrows():

        st.write(f"**{ward_name}**")

        if row["Total_Waste"] > average:
            st.write("Review waste collection frequency.")
        else:
            st.write("Maintain the current collection schedule.")

        if row["Total_Waste"] > 0:
            rate = row["Recycled"] / row["Total_Waste"] * 100

            if rate < 50:
                st.write("Improve waste segregation and recycling.")
            else:
                st.write("Maintain good recycling practices.")

st.success("Dashboard is ready!")

# Clear all saved waste records
if st.sidebar.button("🗑️ Clear All Waste Data"):
    empty_df = pd.DataFrame(columns=[
        "Ward",
        "Date",
        "Waste_Type",
        "Collected_KG",
        "Recycled_KG",
        "Vehicle_Trips"
    ])

    empty_df.to_csv(FILE, index=False)

    st.sidebar.success("All waste records deleted!")
    st.rerun()
