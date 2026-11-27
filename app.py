
import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Smart Waste Management",
    page_icon="♻️",
    layout="wide"
)

st.title("♻️ Smart Waste Management & Recycling Analytics")
st.caption("Manual Waste Data Entry and Analytics Dashboard")

FILE = "manual_waste_data.csv"

COLUMNS = [
    "Ward", "Date", "Waste_Type",
    "Collected_KG", "Recycled_KG", "Vehicle_Trips"
]

if not os.path.exists(FILE):
    pd.DataFrame(columns=COLUMNS).to_csv(FILE, index=False)

# Load saved records
df = pd.read_csv(FILE)

# Manual data entry
st.sidebar.header("✍️ Enter Waste Data")

with st.sidebar.form("waste_form"):
    ward = st.selectbox(
        "Select Ward",
        ["Ward 1", "Ward 2", "Ward 3", "Ward 4", "Ward 5"]
    )

    date = st.date_input("Collection Date")

    waste_type = st.selectbox(
        "Waste Type",
        ["Organic", "Plastic", "Paper", "Metal", "Glass", "Other"]
    )

    collected = st.number_input(
        "Total Waste (KG)",
        min_value=0.0,
        value=100.0
    )

    recycled = st.number_input(
        "Recycled Waste (KG)",
        min_value=0.0,
        value=50.0
    )

    trips = st.number_input(
        "Vehicle Trips",
        min_value=0,
        value=1,
        step=1
    )

    save = st.form_submit_button("➕ Save Record")

    if save:
        if collected <= 0:
            st.error("Total waste must be greater than zero.")
        elif recycled > collected:
            st.error("Recycled waste cannot exceed total waste.")
        else:
            new_record = pd.DataFrame([{
                "Ward": ward,
                "Date": str(date),
                "Waste_Type": waste_type,
                "Collected_KG": collected,
                "Recycled_KG": recycled,
                "Vehicle_Trips": trips
            }])

            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(FILE, index=False)

            st.success("Record saved successfully!")
            st.rerun()

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

# Select a column to clear
st.sidebar.subheader("🧹 Clear Selected Column")

columns = [
    "Ward",
    "Date",
    "Waste_Type",
    "Collected_KG",
    "Recycled_KG",
    "Vehicle_Trips"
]

selected_column = st.sidebar.selectbox(
    "Choose column",
    columns
)

if st.sidebar.button("Clear Selected Column"):
    df = pd.read_csv(FILE)

    if not df.empty:
        df[selected_column] = ""

        df.to_csv(FILE, index=False)

        st.sidebar.success(
            f"{selected_column} data cleared!"
        )

        st.rerun()
    else:
        st.sidebar.warning("No records available.")
