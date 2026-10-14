
import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("data/customer_data.csv")

df = load_data()

# Dashboard title
st.title("Customer Churn & Revenue Intelligence")
st.write("Analyze customer retention, churn patterns, and revenue.")

# Key metrics
total_customers = len(df)
churned_customers = int(df["Churn"].sum())
churn_rate = df["Churn"].mean() * 100
total_revenue = df["TotalRevenue"].sum()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", f"{total_customers:,}")
col2.metric("Churned Customers", f"{churned_customers:,}")
col3.metric("Churn Rate", f"{churn_rate:.1f}%")
col4.metric("Total Revenue", f"{total_revenue:,.0f}")

st.divider()

# Churn by contract
st.subheader("Churn Rate by Contract Type")

contract_churn = (
    df.groupby("Contract", as_index=False)["Churn"]
    .mean()
)

contract_churn["Churn Rate (%)"] = contract_churn["Churn"] * 100

fig1 = px.bar(
    contract_churn,
    x="Contract",
    y="Churn Rate (%)",
    color="Contract",
    title="Customer Churn Rate by Contract"
)

st.plotly_chart(fig1, use_container_width=True)

# Revenue by churn status
st.subheader("Revenue Comparison")

revenue_data = (
    df.groupby("Churn", as_index=False)["TotalRevenue"]
    .mean()
)

revenue_data["Customer Status"] = revenue_data["Churn"].map({
    0: "Stayed",
    1: "Churned"
})

fig2 = px.bar(
    revenue_data,
    x="Customer Status",
    y="TotalRevenue",
    color="Customer Status",
    title="Average Revenue by Customer Status"
)

st.plotly_chart(fig2, use_container_width=True)

# Payment method analysis
st.subheader("Churn by Payment Method")

payment_data = (
    df.groupby("PaymentMethod", as_index=False)["Churn"]
    .mean()
)

payment_data["Churn Rate (%)"] = payment_data["Churn"] * 100

fig3 = px.bar(
    payment_data,
    x="PaymentMethod",
    y="Churn Rate (%)",
    color="PaymentMethod",
    title="Churn Rate by Payment Method"
)

st.plotly_chart(fig3, use_container_width=True)

# Customer data table
st.subheader("Customer Dataset")

status_filter = st.selectbox(
    "Filter customers by status",
    ["All", "Stayed", "Churned"]
)

if status_filter == "Stayed":
    display_df = df[df["Churn"] == 0]
elif status_filter == "Churned":
    display_df = df[df["Churn"] == 1]
else:
    display_df = df

st.dataframe(display_df, use_container_width=True)

st.caption(
    "Note: This project uses synthetic data generated for educational purposes."
)

