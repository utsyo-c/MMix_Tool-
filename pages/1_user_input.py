import streamlit as st
import pandas as pd
import numpy as np

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

    if st.button("Process Data"):
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
    st.title("User Input Interface")

    if 'date_column' in st.session_state and 'geo_column' in st.session_state and 'granular_df' in st.session_state :

        # Loading the session state variables
        date_column = st.session_state['date_column']
        geo_column = st.session_state['geo_column']
        df = st.session_state['granular_df']
        
        transformed_df = user_input(date_column, geo_column, df)

        st.session_state["transformed_df"] = transformed_df

    else:
        st.warning("No file has been uploaded")
