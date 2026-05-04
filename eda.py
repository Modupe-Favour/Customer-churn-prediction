import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_data(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def churn_distribution(df):
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]
    counts["Churn"] = counts["Churn"].map({1: "Churned", 0: "Retained"})
    fig = px.pie(counts, names="Churn", values="Count",
                 title="Overall Churn Distribution",
                 color_discrete_sequence=["#EF553B", "#00CC96"])
    return fig

def churn_by_contract(df):
    data = df.groupby("Contract")["Churn"].mean().reset_index()
    data["Churn"] = (data["Churn"] * 100).round(2)
    fig = px.bar(data, x="Contract", y="Churn",
                 title="Churn Rate by Contract Type (%)",
                 color="Churn", color_continuous_scale="Reds",
                 labels={"Churn": "Churn Rate (%)"})
    return fig

def churn_by_tenure(df):
    df = df.copy()
    df["TenureGroup"] = pd.cut(df["tenure"],
                                bins=[0, 12, 24, 48, 72],
                                labels=["0-12 months", "12-24 months",
                                        "24-48 months", "48-72 months"])
    data = df.groupby("TenureGroup")["Churn"].mean().reset_index()
    data["Churn"] = (data["Churn"] * 100).round(2)
    fig = px.bar(data, x="TenureGroup", y="Churn",
                 title="Churn Rate by Customer Tenure",
                 color="Churn", color_continuous_scale="Oranges",
                 labels={"Churn": "Churn Rate (%)"})
    return fig

def churn_by_monthly_charges(df):
    df = df.copy()
    df["ChurnLabel"] = df["Churn"].map({1: "Churned", 0: "Retained"})
    fig = px.histogram(df, x="MonthlyCharges", color="ChurnLabel",
                       barmode="overlay",
                       title="Monthly Charges by Churn Status",
                       color_discrete_sequence=["#EF553B", "#00CC96"])
    return fig

def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig