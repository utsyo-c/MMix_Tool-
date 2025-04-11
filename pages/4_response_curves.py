import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st
import plotly.express as px
import re

# Global constants - to change!! 
num_time = 12 #number of months 
num_geo = 2614  # number of bricks [geo_id.nunique()] 
# To fetch from dummy data 

# Set page config
st.set_page_config(page_title="Response Curves", layout="centered")
st.title("Response Curve Viewer")

# ----------------------------------------------------------------------------------------------------
# Function Definitions
# ----------------------------------------------------------------------------------------------------

def calc_calibration_factor(impactable_sales_nation, beta_coeff, spend_nation):   #to fetch choice of function  
    calib_factor = (impactable_sales_nation / (num_time * num_geo)) / (
        beta_coeff * np.log(1 + (spend_nation / (num_time * num_geo))))   
    return calib_factor


def create_response_curve(channel_name, impactable_sales_nation, beta_coeff, spend_nation, start, stop, step, price):
    calibration_factor = calc_calibration_factor(impactable_sales_nation, beta_coeff, spend_nation)
    spend_values = range(start, stop + 1, step)

    response_df = pd.DataFrame({
        'spend': spend_values,
        'impactable_geo_time': [None] * len(spend_values),
        'impactable_nation': [None] * len(spend_values),
        'impactable_nation_currency': [None] * len(spend_values),
        'roi': [None] * len(spend_values),
        'mroi': [None] * len(spend_values)
    })

    response_df['impactable_geo_time'] = calibration_factor * beta_coeff * np.log(
        1 + (response_df['spend'] / (num_time * num_geo)))
    response_df['impactable_nation'] = num_time * num_geo * response_df['impactable_geo_time']
    response_df['impactable_nation_currency'] = price * response_df['impactable_nation']
    response_df['roi'] = response_df['impactable_nation_currency'] / response_df['spend']
    response_df['mroi'] = (response_df['impactable_nation_currency'].shift(-1) -
                           response_df['impactable_nation_currency']) / (response_df['spend'].shift(-1) -
                                                                          response_df['spend'])

    channel_prefix = channel_name + '_'
    response_df = response_df.add_prefix(channel_prefix)

    return response_df


def create_final_merged_response_curve(model_result_df, start, stop, step, price):
    final_merged_response_curve = pd.DataFrame()

    for channel_name in model_result_df['channel']:
        impactable_sales_nation = float(model_result_df[model_result_df['channel'] == channel_name]['impactable_sensors'])
        beta_coeff = float(model_result_df[model_result_df['channel'] == channel_name]['coefficient'])
        spend_nation = float(model_result_df[model_result_df['channel'] == channel_name]['spend'])

        response_curve = create_response_curve(channel_name, impactable_sales_nation, beta_coeff, spend_nation,
                                               start, stop, step, price)

        if final_merged_response_curve.empty:
            final_merged_response_curve = response_curve
        else:
            final_spend_col = final_merged_response_curve.columns[0]
            new_spend_col = response_curve.columns[0]
            final_merged_response_curve = final_merged_response_curve.merge(response_curve,
                                                                            left_on=final_spend_col,
                                                                            right_on=new_spend_col,
                                                                            how='inner')

    return final_merged_response_curve


# ----------------------------------------------------------------------------------------------------
# Streamlit Interface
# ----------------------------------------------------------------------------------------------------

# File upload
uploaded_file = st.file_uploader("📤 Upload CSV file (Model Results)", type=["csv"])


# Response curve generation inputs
start = st.number_input("Start Spend Value", value=1000, step=1000)
stop = st.number_input("Stop Spend Value", value=40000000, step=1000000)
price = st.number_input("Unit Price of Product", value=49.6, format="%.2f")
step = 1000

if uploaded_file:
    try:
        model_result_df = pd.read_csv(uploaded_file)

        if {'channel', 'impactable_sensors', 'coefficient', 'spend'}.issubset(model_result_df.columns):
            merged_rc = create_final_merged_response_curve(model_result_df, start, stop, step, price)

            # Storing session state variables
            st.session_state['model_result_df'] = model_result_df
            st.session_state["merged_rc"] = merged_rc

            st.success("✅ Response curves generated successfully!")
            st.subheader("Merged Response Curve Data")
            st.dataframe(merged_rc)

            # ---- Interactive Viewer ----
            st.markdown("---")
            st.subheader("Plot Individual Channel Curves")

            # Extract channel prefixes
            prefix_pattern = re.compile(r"^((hcp|dtc)_[a-z]+)_")
            prefixes = sorted({match.group(1) for col in merged_rc.columns if (match := prefix_pattern.match(col))})

            if not prefixes:
                st.error("No valid channel prefixes found in column names.")
            else:
                selected_prefix = st.selectbox("Select a Channel to Plot", prefixes)
                matching_cols = [col for col in merged_rc.columns if col.startswith(selected_prefix + "_")]
                
                if len(matching_cols) < 2:
                    st.warning(f"Not enough columns starting with '{selected_prefix}_' to plot a response curve.")
                else:
                    x_col = st.selectbox("Select X-axis", matching_cols, index=0)
                    y_col = st.selectbox("Select Y-axis", matching_cols, index=1)

                    fig = px.scatter(merged_rc, x=x_col, y=y_col,
                                     title=f"Response Curve for {selected_prefix.replace('_', ' ').title()}",
                                     labels={
                                         x_col: x_col.replace(selected_prefix + '_', '').replace('_', ' ').title(),
                                         y_col: y_col.replace(selected_prefix + '_', '').replace('_', ' ').title()
                                     })
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📄 Filtered Channel Data")
                    st.dataframe(merged_rc[matching_cols])

            # ---- Multi-Channel Comparison ----
            st.markdown("---")
            st.subheader("Compare Multiple Channel Curves")

            multi_selected_prefixes = st.multiselect("Select Channels to Compare", prefixes)

            if multi_selected_prefixes:
                y_metric_options = ["roi","mroi","impactable_nation_currency"]
                selected_y_metric = st.selectbox("Select Y-axis Metric", y_metric_options, index=1)

                fig_multi = px.line()

                for prefix in multi_selected_prefixes:
                    spend_col = prefix + "_spend"
                    y_col = prefix + "_" + selected_y_metric

                    if spend_col in merged_rc.columns and y_col in merged_rc.columns:
                        fig_multi.add_scatter(
                            x=merged_rc[spend_col],
                            y=merged_rc[y_col],
                            mode="lines+markers",
                            name=prefix.replace("_", " ").title()
                        )

                fig_multi.update_layout(
                    title=f"Comparison of Channels on {selected_y_metric.replace('_', ' ').title()}",
                    xaxis_title="Spend",
                    yaxis_title=selected_y_metric.replace("_", " ").title(),
                    legend_title="Channel"
                )

                st.plotly_chart(fig_multi, use_container_width=True)

        else:
            st.error("Uploaded CSV does not contain required columns: 'channel', 'impactable_sensors', 'coefficient', 'spend'.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
