import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm

def modeling_input(date_column, geo_column, df):
    # Columns to exclude from selection
    remove_cols = [date_column, geo_column, 'Sales']
    available_channels = [
        col for col in df.drop(columns=remove_cols).columns
        if col.endswith('_transformed')
    ]

    # User selects channels
    selected_channels = st.multiselect("Select channels to model on", available_channels)

    if selected_channels:
        selected_channels_df = pd.DataFrame({'Channel Name': selected_channels})
        # Shift index to start from 1
        selected_channels_df.index = np.arange(1, len(selected_channels_df)+1)

        if st.button("Enter"):
            st.success("Channels Received:")
            st.dataframe(selected_channels_df)

            # Filter df to selected channels + essential columns
            filtered_df = df[[date_column, geo_column, 'Sales'] + selected_channels]

            # Save to session state
            st.session_state['selected_channels_df'] = selected_channels_df
            st.session_state['filtered_df'] = filtered_df
            st.session_state['selected_channels'] = selected_channels  # Save names too

            return selected_channels_df

    return None


def run_regression(dependent_variable):
    # Retrieve filtered df and selected predictors
    filtered_df = st.session_state['filtered_df']
    selected_channels = st.session_state['selected_channels']

    X = filtered_df[selected_channels]
    y = filtered_df[dependent_variable]

    # Add a constant term for intercept
    X = sm.add_constant(X)

    # Fit the OLS model
    model = sm.OLS(y, X).fit()

    # Get the summary of the regression
    regression_summary = model.summary().as_text()

    # Display the OLS regression summary
    st.subheader("OLS Regression Results")
    st.text(regression_summary)

    # --- ADDITIONAL SUMMARY COMPUTATIONS ---
    st.subheader("Model Summary Table")

    # Extract coefficients into a DataFrame
    coefficients = pd.DataFrame({
        'Variable': model.params.index,
        'Coefficient': model.params.values
    })

    # Compute total sales from filtered_df
    sum_sales = filtered_df[dependent_variable].sum()
    filtered_df['const'] = 1
    coefficients['Modelled_activity'] = coefficients['Variable'].apply(lambda var: filtered_df[var].sum())
    coefficients['Impactable%'] = coefficients.apply(lambda row: (row['Coefficient'] * row ['Modelled_activity'] * 100) / sum_sales, axis = 1)
    
    st.dataframe(coefficients)

    # Save regression output to session state
    if 'regression_outputs' not in st.session_state:
        st.session_state['regression_outputs'] = []

    # Append both summary and coefficients
    st.session_state['regression_outputs'].append({
        'summary': regression_summary,
        'coefficients': coefficients
    })

# Run the Streamlit app
if __name__ == '__main__':
    st.title("Input Channels for Modeling")

    if 'date_column' in st.session_state and 'geo_column' in st.session_state and 'granular_df' in st.session_state and 'transformed_df' in st.session_state and 'dependent_variable' in st.session_state:
        # Load session state variables
        date_column = st.session_state['date_column']
        geo_column = st.session_state['geo_column']
        df = st.session_state['granular_df']
        transformed_df = st.session_state['transformed_df']
        dependent_variable = st.session_state['dependent_variable']

        selected_channels_df = modeling_input(date_column, geo_column, transformed_df)

        # Show filtered_df if it's available
        if 'filtered_df' in st.session_state:
            st.subheader("Filtered DataFrame Preview")
            st.dataframe(st.session_state['filtered_df'])

            # Add a button to run regression
            if st.button("Run OLS Regression"):
                run_regression(dependent_variable)
    else:
        st.warning("No file has been uploaded")
