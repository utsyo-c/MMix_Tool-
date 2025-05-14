import streamlit as st
import pandas as pd
import numpy as np

# Create a carryover term
def apply_lag_to_dependent_variable(df, dependent_variable, geo_column, granularity_level_user_input):
    if granularity_level_user_input == "Weekly":
        lag_label = "weeks"
        max_lag = 52
    elif granularity_level_user_input == "Monthly":
        lag_label = "months"
        max_lag = 12
    else:
        st.warning("Granularity level not recognized. Lag cannot be applied.")
        return df, False

    lag_input = st.text_input(
        f"Enter how many {lag_label} to lag the dependent variable ({dependent_variable}):",
        value="1",
        key="dependent_lag_input_text"
    )

    if lag_input.isdigit():
        lag_value = int(lag_input)
        if 1 <= lag_value <= max_lag:
            df[f"{dependent_variable} Lagged"] = (
                df.groupby(geo_column, as_index = False)[dependent_variable].shift(lag_value, fill_value = 0).fillna(0))
            st.info(f"{dependent_variable} has been lagged by {lag_value} {lag_label} within each granularity entry.")
            return df, True
        else:
            st.error(f"Please enter a value between 1 and {max_lag} for {lag_label}.")
    else:
        st.error("Please enter a valid whole number.")
    
    return df, False

# Define Adstock function
def geometric_adstock(series, lags, adstock_coeff):
    """Applies geometric Adstock transformation within each region."""
    series = np.array(series, dtype=np.float64)  # Ensure it's numeric
    adstocked = np.zeros_like(series)

    for i in range(len(series)):
        for j in range(lags + 1):
            if i - j >= 0:
                adstocked[i] += (adstock_coeff ** j) * series[i - j]

    return adstocked

# Define Saturation function
def apply_saturation(series, method, power_k=0.5):
    """Applies saturation function on a Pandas Series or NumPy array."""
    series = np.array(series, dtype=np.float64)  # Ensure it's numeric
    
    if method.lower() == "log":
        return np.log1p(series)  # log(1 + x) to avoid log(0)
    elif method.lower() == "power":
        return np.power(series, power_k)
    
    return series  # Return unchanged if no valid method is given

# Function to apply transformations on df (original data), grouped by geo_column
def transform_edited_df(df, edited_df, geo_column):
    transformed_df = df.copy()

    for _, row in edited_df.iterrows():
        channel = row["Channel Name"]
        lags = int(row["Lags"])
        adstock_coeff = float(row["Adstock"])
        sat_function = row["Saturation Function"]
        power_k = float(row["Power (k)"]) if sat_function == "Power" else None

        if channel in transformed_df.columns:
            # Apply transformation separately for each region
            transformed_df[f"{channel}_transformed"] = (
                transformed_df.groupby(geo_column)[channel]
                .transform(lambda x: apply_saturation(geometric_adstock(x, lags, adstock_coeff), sat_function, power_k))
            )

    return transformed_df

# Function to handle user input and apply transformations
def user_input(date_column, geo_column, df):
    remove_cols = [date_column, geo_column, 'Sales']
    filtered_df = df.drop(columns=remove_cols)

    # Define initial data for user input
    channel_names = filtered_df.columns
    data = pd.DataFrame({
        "Channel Name": channel_names,
        "Saturation Function": ["Choose a function"] * len(channel_names),
        "Power (k)": [0.5] * len(channel_names),
        "Lags": [1] * len(channel_names),
        "Adstock": [0.5] * len(channel_names)
    })

    # Streamlit Data Editor
    edited_df = st.data_editor(
        data,
        column_config={
            "Channel Name": st.column_config.TextColumn("Channel Name", disabled=True),
            "Saturation Function": st.column_config.SelectboxColumn(
                "Saturation Function",
                options=["Power", "Log"],
                help="Select the saturation function for the channel"
            ),
            "Power (k)": st.column_config.NumberColumn("Power (k)", min_value=0.0, step=0.1, max_value=1),
            "Lags": st.column_config.NumberColumn("Lags", min_value=0, step=1, max_value=12),
            "Adstock": st.column_config.NumberColumn("Adstock", min_value=0.0, step=0.1, max_value=1)
        },
        hide_index=True,
        key="editable_table"
    )


    # Post-processing logic
    for i, row in edited_df.iterrows():
        if row["Saturation Function"] == "Log":
            edited_df.at[i, "Power (k)"] = None  # or some ignored flag like np.nan

    # Optionally display a note
    if "Log" in edited_df["Saturation Function"].values:
        st.warning("Note: For channels using the 'Log' saturation function, 'Power (k)' will be ignored.")




    if st.button("Process Data"):

        # # Append both summary and coefficients
        st.session_state['configuration_list'].append(edited_df)

        st.write("Final Inputs Received:")
        st.write(edited_df)

        # Apply transformations to df (actual spend data), grouped by geo_column
        transformed_df = transform_edited_df(df, edited_df, geo_column)

        # Display transformed DataFrame
        st.subheader("Transformed Data")
        st.dataframe(transformed_df)

        return transformed_df

# Run the Streamlit app
if __name__ == '__main__':


    if 'configuration_list' not in st.session_state:
        st.session_state['configuration_list'] = []    #Store regression outputs as a list in session state to preserve multiple versions


    st.title("User Input Interface")

    if 'date_column' in st.session_state and 'geo_column' in st.session_state and 'granular_df' in st.session_state and 'dependent_variable' in st.session_state:

        # Loading the session state variables
        date_column = st.session_state['date_column']
        geo_column = st.session_state['geo_column']
        df_raw = st.session_state['granular_df']
        df = st.session_state['granular_df']
        dependent_variable = st.session_state['dependent_variable']
        granularity_level_user_input = st.session_state['granularity_level_user_input']

        st.subheader("Original Granular Data")
        st.dataframe(df_raw)
        # st.write(df_raw[(df_raw['month_date'] > "2021-12-31")&(df_raw['month_date'] < "2023-01-01")]["Sales"].sum())
        # st.write(df_raw[df_raw['month_date'] > "2022-12-31"]["Sales"].sum())

        df, lag_applied = apply_lag_to_dependent_variable( df, dependent_variable, geo_column, granularity_level_user_input)
        if lag_applied:
            st.subheader(f"Granular Data with Lagged {dependent_variable}")
            st.dataframe(df)

        transformed_df = user_input(date_column, geo_column, df)

        st.session_state["transformed_df"] = transformed_df

    else:
        st.warning("No file has been uploaded")