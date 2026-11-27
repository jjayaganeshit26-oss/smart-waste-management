
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Smart Waste Management",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Smart Waste Management System")
st.caption("Manual Waste Monitoring and Recycling Analytics")

FILE = "manual_waste_data.csv"

COLUMNS = [
    "Date", "Ward", "Organic_KG",
    "Plastic_KG", "Paper_KG", "Metal_KG",
    "Recycled_KG", "Vehicle_Trips"
]

# ---------------- DATA STORAGE ----------------

if not os.path.exists(FILE):
    pd.DataFrame(columns=COLUMNS).to_csv(FILE, index=False)


def load_data():
    data = pd.read_csv(FILE)
    if not data.empty:
        data["Date"] = pd.to_datetime(
            data["Date"], errors="coerce"
        )
        for col in COLUMNS[2:]:
            data[col] = pd.to_numeric(
                data[col], errors="coerce"
            ).fillna(0)
    return data


df = load_data()

# ---------------- MANUAL DATA ENTRY ----------------

st.sidebar.header("📝 Manual Waste Reading")

with st.sidebar.form("waste_form"):

    ward = st.selectbox(
        "Select Ward",
        [f"Ward {i}" for i in range(1, 11)]
    )

    date = st.date_input("Collection Date")

    st.markdown("### Waste Readings (KG)")

    organic = st.number_input(
        "Organic Waste",
        min_value=0.0,
        value=0.0
    )

    plastic = st.number_input(
        "Plastic Waste",
        min_value=0.0,
        value=0.0
    )

    paper = st.number_input(
        "Paper Waste",
        min_value=0.0,
        value=0.0
    )

    metal = st.number_input(
        "Metal Waste",
        min_value=0.0,
        value=0.0
    )

    recycled = st.number_input(
        "Total Recycled Waste (KG)",
        min_value=0.0,
        value=0.0
    )

    trips = st.number_input(
        "Vehicle Trips",
        min_value=0,
        value=1,
        step=1
    )

    submit = st.form_submit_button(
        "➕ Save Waste Reading",
        use_container_width=True
    )

    if submit:
        total_input = organic + plastic + paper + metal

        if total_input <= 0:
            st.error("Enter at least one waste reading.")

        elif recycled > total_input:
            st.error(
                "Recycled waste cannot exceed total waste."
            )

        elif df[
            (df["Date"].dt.date == date) &
            (df["Ward"] == ward)
        ].shape[0] > 0:
            st.error(
                "A reading already exists for this ward and date. "
                "Delete it before entering a replacement."
            )

        else:
            new_record = pd.DataFrame([{
                "Date": str(date),
                "Ward": ward,
                "Organic_KG": organic,
                "Plastic_KG": plastic,
                "Paper_KG": paper,
                "Metal_KG": metal,
                "Recycled_KG": recycled,
                "Vehicle_Trips": trips
            }])

            new_record.to_csv(
                FILE,
                mode="a",
                header=False,
                index=False
            )

            st.success("Waste reading saved!")
            st.rerun()

# ---------------- RELOAD DATA ----------------

df = load_data()

if not df.empty:
    df["Total_Waste_KG"] = (
        df["Organic_KG"] +
        df["Plastic_KG"] +
        df["Paper_KG"] +
        df["Metal_KG"]
    )

    df["Overall_Wastage_KG"] = (
        df["Total_Waste_KG"] -
        df["Recycled_KG"]
    )

# ---------------- DASHBOARD ----------------

st.header("📊 Overall Dashboard")

if df.empty:
    st.info(
        "No readings available. Enter waste readings "
        "using the sidebar to start the dashboard."
    )
else:
    total_waste = df["Total_Waste_KG"].sum()
    recycled_total = df["Recycled_KG"].sum()
    organic_total = df["Organic_KG"].sum()
    overall_wastage = df["Overall_Wastage_KG"].sum()

    active_wards = df["Ward"].nunique()

    recycling_rate = (
        recycled_total / total_waste * 100
        if total_waste > 0 else 0
    )

    # KPI CARDS

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Waste", f"{total_waste:,.2f} KG")
    c2.metric("Recycled Waste", f"{recycled_total:,.2f} KG")
    c3.metric("Organic Waste", f"{organic_total:,.2f} KG")
    c4.metric("Active Wards", active_wards)

    c5, c6 = st.columns(2)

    c5.metric("Overall Wastage", f"{overall_wastage:,.2f} KG")
    c6.metric("Recycling Rate", f"{recycling_rate:.2f}%")

    st.divider()

    # ---------------- WASTE COMPOSITION ----------------

    st.subheader("1. Waste Composition")

    composition = pd.Series({
        "Organic": df["Organic_KG"].sum(),
        "Plastic": df["Plastic_KG"].sum(),
        "Paper": df["Paper_KG"].sum(),
        "Metal": df["Metal_KG"].sum()
    })

    st.bar_chart(composition)

    st.subheader("Waste Composition Details")

    st.dataframe(
        composition.rename("Waste (KG)").reset_index()
        .rename(columns={"index": "Waste Type"}),
        use_container_width=True
    )

    # ---------------- WARD MONITORING ----------------

    st.subheader("2. Ward-wise Waste Monitoring")

    ward_data = df.groupby("Ward").agg(
        Total_Waste=("Total_Waste_KG", "sum"),
        Recycled=("Recycled_KG", "sum"),
        Organic=("Organic_KG", "sum")
    )

    ward_data["Remaining_Waste"] = (
        ward_data["Total_Waste"] -
        ward_data["Recycled"]
    )

    st.bar_chart(ward_data[["Total_Waste"]])

    st.dataframe(ward_data, use_container_width=True)

    # ---------------- DATE FILTER ----------------

    st.subheader("3. Overall Waste Analysis")

    selected_ward = st.selectbox(
        "Filter by Ward",
        ["All Wards"] + sorted(df["Ward"].unique())
    )

    filtered = df.copy()

    if selected_ward != "All Wards":
        filtered = filtered[
            filtered["Ward"] == selected_ward
        ]

    st.line_chart(
        filtered.groupby("Date")["Total_Waste_KG"].sum()
    )

    # ---------------- FORECAST ----------------

    st.subheader("4. Seven-Day Waste Forecast")

    daily = (
        df.groupby("Date")["Total_Waste_KG"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    if daily["Date"].nunique() >= 7:
        daily["DayNumber"] = (
            daily["Date"] - daily["Date"].min()
        ).dt.days

        X = daily[["DayNumber"]]
        y = daily["Total_Waste_KG"]

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X, y)

        future_dates = pd.date_range(
            daily["Date"].max() + pd.Timedelta(days=1),
            periods=7
        )

        future_days = (
            future_dates - daily["Date"].min()
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
            forecast.set_index("Date")["Predicted_Waste_KG"]
        )

        st.download_button(
            "Download Forecast",
            forecast.to_csv(index=False),
            "waste_forecast.csv",
            "text/csv"
        )

    else:
        st.warning(
            "Enter readings for at least 7 different dates "
            "to generate a forecast."
        )

    # ---------------- RECORDS ----------------

    st.subheader("5. Manual Waste Records")

    display_df = df.copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")

    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        "📥 Download Waste Report",
        display_df.to_csv(index=False),
        "waste_management_report.csv",
        "text/csv"
    )

    # ---------------- DELETE BY DATE ----------------

    st.subheader("6. Delete Records by Date")

    available_dates = sorted(
        df["Date"].dropna().dt.date.unique()
    )

    delete_date = st.selectbox(
        "Choose date to delete",
        available_dates,
        key="delete_date"
    )

    if st.button("🗑️ Delete All Records for This Date"):
        df = df[df["Date"].dt.date != delete_date]

        df.drop(
            columns=[
                "Total_Waste_KG",
                "Overall_Wastage_KG"
            ],
            errors="ignore"
        ).to_csv(FILE, index=False)

        st.success("Selected date records deleted.")
        st.rerun()

    # ---------------- RECOMMENDATIONS ----------------

    st.subheader("7. Smart Recommendations")

    for ward_name, row in ward_data.iterrows():
        st.markdown(f"**{ward_name}**")

        if row["Total_Waste"] > ward_data["Total_Waste"].mean():
            st.write(
                "High waste generation: review collection frequency."
            )
        else:
            st.write(
                "Waste generation is at or below the average."
            )

        if row["Total_Waste"] > 0:
            ward_rate = (
                row["Recycled"] / row["Total_Waste"] * 100
            )

            if ward_rate < 50:
                st.write(
                    "Improve segregation and recycling practices."
                )
            else:
                st.write(
                    "Maintain good recycling practices."
                )

st.success("Smart Waste Management Dashboard is ready!")


# Reload latest records
df = pd.read_csv(FILE)

st.header("📊 Waste Management Dashboard")

if df.empty:
    st.info("No records yet. Enter waste data using the sidebar.")
else:
    df["Collected_KG"] = pd.to_numeric(
        df["Collected_KG"], errors="coerce"
    ).fillna(0)

    df["Recycled_KG"] = pd.to_numeric(
        df["Recycled_KG"], errors="coerce"
    ).fillna(0)

    total = df["Collected_KG"].sum()
    recycled_total = df["Recycled_KG"].sum()
    remaining = total - recycled_total
    rate = recycled_total / total * 100 if total else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Waste", f"{total:,.2f} KG")
    c2.metric("Recycled Waste", f"{recycled_total:,.2f} KG")
    c3.metric("Remaining Waste", f"{remaining:,.2f} KG")
    c4.metric("Recycling Rate", f"{rate:.1f}%")

    st.subheader("1. Ward-wise Waste Generation")
    ward_data = df.groupby("Ward")["Collected_KG"].sum()
    st.bar_chart(ward_data)

    st.subheader("2. Waste Type Analysis")
    type_data = df.groupby("Waste_Type")["Collected_KG"].sum()
    st.bar_chart(type_data)

    st.subheader("3. Recycling Analysis")
    st.bar_chart(pd.DataFrame({
        "Waste (KG)": [recycled_total, remaining]
    }, index=["Recycled", "Not Recycled"]))

    st.subheader("4. Waste Collection Records")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        "Download Waste Records CSV",
        df.to_csv(index=False),
        "waste_records.csv",
        "text/csv"
    )

    st.subheader("5. Smart Recommendations")

    for ward_name, amount in ward_data.items():
        st.write(f"**{ward_name}:** {amount:.2f} KG collected")

        if amount > ward_data.mean():
            st.write("High waste generation. Review collection frequency.")
        else:
            st.write("Waste level is below the average.")

st.success("Smart Waste Management Dashboard is ready!")




# Delete records using collection date
st.sidebar.subheader("🗑️ Delete Records by Date")

df = pd.read_csv(FILE)

if not df.empty:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    available_dates = sorted(
        df["Date"].dropna().dt.date.unique()
    )

    selected_date = st.sidebar.selectbox(
        "Select date to delete",
        available_dates
    )

    if st.sidebar.button("Delete Records for Selected Date"):
        df = df[
            df["Date"].dt.date != selected_date
        ]

        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        df.to_csv(FILE, index=False)

        st.sidebar.success(
            f"Records for {selected_date} deleted!"
        )
        st.rerun()

else:
    st.sidebar.info("No records available.")

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
