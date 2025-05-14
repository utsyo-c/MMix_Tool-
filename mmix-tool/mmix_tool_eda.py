import streamlit as st
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from datetime import datetime, timedelta
import warnings


from helper import eda_sales_trend, modify_granularity


if __name__=='__main__':

    warnings.filterwarnings('ignore')
    
    # Streamlit App
    st.title("Exploratory Data Analysis")

    if "uploaded_file_df" not in st.session_state:  
        
        # Upload CSV File
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

        if uploaded_file:
            # Call the EDA function
            result = eda_sales_trend(uploaded_file)
            
            if result:
                geo_column, date_column, dependent_variable, granularity_level_df, granularity_level_user_input, df = result
                # Modifying granularity
                granular_df,date_column = modify_granularity(geo_column, date_column, granularity_level_df, granularity_level_user_input, df)

                if "dataframe" not in st.session_state:
                    st.session_state["dataframe"] = pd.DataFrame()
                
                st.session_state["dataframe"] = st.dataframe(granular_df)
                st.session_state["date_column"] = date_column
                st.session_state["geo_column"] = geo_column
                st.session_state['granular_df'] = granular_df
                st.session_state['dependent_variable'] = dependent_variable
                st.session_state['granularity_level_user_input'] = granularity_level_user_input
        else:
            st.warning("No file has been uploaded")
