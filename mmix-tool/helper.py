import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import plotly.express as px
import warnings



# Helper functions


# Last week apportioning 

# Step 1: Set the 'last Non Weekend day of every month' as the last working date
# create a dictionary for this date for each unique month in the data

def last_working_day(year, month, work_days):
    """""
    Sets the last non-weekend day of every month as the last working date

    Args:
        year (scalar): Year of the date
        month (scalar): Month of the date
        work_days (scalar): Number of working days in the week

    Returns
        last_day (scalar): The last working day of the month
    """


    if month==12:
        month=0
        year+=1

    last_day = datetime(year, month+1, 1) - timedelta(days=1)

    if work_days==5:
        while last_day.weekday() > 4:  # Friday is weekday 4
            last_day -= timedelta(days=1) #subtracting 1 day at a time

    return last_day

def rename_adjusted(kpi_name):
    new_name = "adjusted_" + kpi_name
    return new_name


def last_week_apportion(tactic_df,date_col_name,kpi_col_list,work_days):
    """""
    Proportionately allocates KPIs of last week in that month accurately to each month 
    based on number of working days in that week
    
    Args:
        tactic_df (dataframe): Dataframe containing geo-month and KPI information
        date_col_name (string): Column in tactic_df which corresponds to date
        kpi_col_list (list): List of KPI columns to be apportioned 
        work_days (scalar): Number of working days in the week

    Returns
        tactic_df (dataframe): Dataframe with KPI columns apportioned
    """

    # Step 1: Calculate last working date and create month level column

    tactic_df['month'] = tactic_df[date_col_name].dt.to_period('M')
    last_working_day_dict = {month: last_working_day(month.year, month.month,work_days) for month in tactic_df['month'].unique()}
    tactic_df['last_working_date'] = tactic_df['month'].map(last_working_day_dict)

    # Step 2: Calculate day difference from week_start_date to working_date
    tactic_df['day_diff'] = (tactic_df['last_working_date'] - tactic_df[date_col_name] + timedelta(days=1)).dt.days

    # Step 3: Filter weeks with day_diff < work_days and calculate adjusted calls
    adjusted_col_list = []
    for kpi_name in kpi_col_list:
        tactic_df[rename_adjusted(kpi_name)] = tactic_df.apply(lambda row: ((work_days-row['day_diff']) / work_days) * row[kpi_name] if row['day_diff'] < work_days else 0, axis=1)
        adjusted_col_list.append(rename_adjusted(kpi_name))

    # Step 4: Subtract adjusted calls from original calls and add new rows with adjusted calls for the next month
    # Original rows with calls subtracted


    for kpi_name in kpi_col_list:
        tactic_df[kpi_name] = tactic_df[kpi_name] - tactic_df[rename_adjusted(kpi_name)]

    # New rows with adjusted calls on the first day of the month
    new_rows = tactic_df[tactic_df[adjusted_col_list].gt(0).any(axis=1)].copy()
    new_rows[date_col_name] = new_rows[date_col_name] + pd.offsets.MonthBegin()

    # Add new rows
    for kpi_name in kpi_col_list:
        new_rows[kpi_name] = new_rows[rename_adjusted(kpi_name)]

    # Combine original and new rows
    tactic_df = pd.concat([tactic_df, new_rows], ignore_index=True)
    tactic_df.drop(['last_working_date','day_diff','month'], axis=1, inplace=True)

    #Removing the adjusted calls columns
    for adj_col in adjusted_col_list:
        tactic_df.drop(adj_col, axis=1, inplace=True)

    return tactic_df


# ------------------------------------------------------------------------------

# Detect time granularity present in dataframe input by user

def detect_date_granularity(df, date_column):
    # Convert to datetime
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

        df = df.drop_duplicates(subset = [date_column])

        # Sort dates
        df = df.sort_values(by=date_column)

        # Calculate differences between consecutive dates
        date_diffs = df[date_column].diff().dropna()

        # Get the most common time difference
        most_common_diff = date_diffs.mode()[0]  

        # Determine granularity based on the most common difference
        if most_common_diff.days == 1:
            return "Daily"
        elif most_common_diff.days == 7:
            return "Weekly"
        elif most_common_diff.days in [28, 29, 30, 31]:
            return "Monthly"
        elif most_common_diff.days >= 365:
            return "Yearly"
        else:
            st.warning(f"Irregular Date. Please check the column format.")
        
    except Exception as e:
        st.warning(f"The column '{date_column}' is not in a valid date format. Please check the column format.")


# ------------------------------------------------------------------------------


#   Convert time granularity from one to another (for creating integrated analytical database)    
def modify_granularity(geo_column,date_column,granularity_level_df,granularity_level_user_input, df):


    if granularity_level_df == granularity_level_user_input:
        remove_cols = [date_column, geo_column]
        filtered_df = df.drop(columns=remove_cols)
        df = df.groupby([geo_column, date_column])[filtered_df.columns].sum().reset_index()
        return df, date_column
    
    # Daily to week date
    elif granularity_level_user_input == 'Weekly':
        remove_cols = [date_column, geo_column]
        filtered_df = df.drop(columns=remove_cols)
        df["week_date"] = df[date_column] - pd.to_timedelta(df[date_column].dt.weekday, unit="D")
        df = df.groupby([geo_column, "week_date"])[filtered_df.columns].sum().reset_index()
        return df, 'week_date'
    
    # Daily to month date
    elif granularity_level_user_input == 'Monthly' and granularity_level_df=='Daily':
        # df["month_date"] = df[date_column].dt.to_period("M").apply(lambda r: r.start_time)
        remove_cols = [date_column,geo_column]
        filtered_df = df.drop(columns=remove_cols)
        df['month_date'] = df[date_column].dt.strftime('%Y-%m-01')
        df = df.groupby([geo_column,'month_date'])[filtered_df.columns].sum().reset_index()
        return df , 'month_date'

    # Weekly to month date
    elif granularity_level_user_input == 'Monthly' and granularity_level_df=='Weekly':
        remove_cols = [date_column,geo_column]
        filtered_df = df.drop(columns=remove_cols)
        
        # Allow user to select number of working days
        work_days = st.selectbox("Select number of working days per week", options=[5, 6, 7], index=2)

        df = last_week_apportion(df, date_column, filtered_df.columns, work_days= work_days)   

        df['month_date'] = df[date_column].dt.strftime('%Y-%m-01')
        df = df.groupby([geo_column,'month_date'])[filtered_df.columns].sum().reset_index()

        return df, 'month_date'


# ------------------------------------------------------------------------------


#   Perform EDA for channels
# def eda_sales_trend(uploaded_file):

#     if uploaded_file is not None:
#         # Read CSV File
#         df = pd.read_csv(uploaded_file)
#         st.write("### Preview of Uploaded CSV")
#         st.dataframe(df.head(100))

#         # Let user select columns
#         st.subheader("Select Columns")
#         dependent_variable = st.selectbox("Select Dependent Variable", [None] + list(df.columns), index=0 )
#         geo_column = st.selectbox("Select Modeling Granularity Column", [None] + list(df.columns), index=0)
#         date_column = st.selectbox("Select Date Column", [None] + list(df.columns), index=0)

#         if geo_column is not None:
#             num_geo = df[geo_column].nunique()
#             st.success(f"Number of unique entries in granularity column: {num_geo}")
        
#         # Time granularity detection
#         if date_column is not None:
#             time_granularity = detect_date_granularity(df, date_column )
#             st.write(f' The time granularity is : {time_granularity} ')
        
#             # Convert to datetime format
#             df[date_column] = pd.to_datetime(df[date_column])
#             df = df.dropna(subset=[date_column])

#             df_copy = df   #Original dataframe
           
#             # Set up date range filtering
#             min_date = df[date_column].min()
#             max_date = df[date_column].max()

#             # Convert to datetime.date if needed
#             min_date = pd.to_datetime(min_date).date()
#             max_date = pd.to_datetime(max_date).date()

#             # Side-by-side date selectors
#             col1, col2 = st.columns(2)
#             with col1:
#                 start_date = st.date_input(
#                     "Start Date",
#                     value=min_date,
#                     min_value=min_date,
#                     max_value=max_date,
#                     key="start_date"
#                 )
#             with col2:
#                 end_date = st.date_input(
#                     "End Date",
#                     value=max_date,
#                     min_value=min_date,
#                     max_value=max_date,
#                     key="end_date"
#                 )

#             # Validate selected dates
#             if start_date > end_date:
#                 st.warning("⚠️ Start date must be before or equal to end date.")
#                 st.stop()

#             # Filter the DataFrame based on selected range
#             df = df[(df[date_column] >= pd.to_datetime(start_date)) & (df[date_column] <= pd.to_datetime(end_date))]


#             # Handle case when only one date or empty data
#             if df.empty:
#                 st.warning("No data available for the selected date range. Please choose a broader range.")
#                 st.stop()

#             if (pd.to_datetime(start_date) == pd.to_datetime(end_date)):
#                 st.warning("Start and end dates are the same. Please select a wider date range for meaningful analysis.")
#                 st.stop()

#             # Control totals    
#             st.subheader("Control Totals")

#             remove_cols = [date_column, geo_column,'Month', 'Year']
#             filtered_df = df.drop(columns=remove_cols)

#             subtotal_df = pd.DataFrame({
#                 'Channel': filtered_df.columns,
#                 'Total': filtered_df.sum()
#             }
#             )
#             # subtotal_df['Total'] = subtotal_df['Total'].apply(lambda x: f"{x:,}")

#             styled_df = subtotal_df.style.format({"Total": "{:,}"})
#             st.dataframe(styled_df)

#             # Correlation Matrix
#             st.subheader("Correlation Matrix")
#             corr_matrix = filtered_df.corr()
#             fig = sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5).get_figure()
#             # Display in Streamlit
#             st.pyplot(fig)

#             #Metric to visualize
#             # Metric to visualize
#             st.subheader("Trend visualization")

#             # Allow only numeric columns for visualization
#             numeric_cols = df.select_dtypes(include='number').columns.tolist()
#             numeric_cols = [item for item in numeric_cols if item not in ['Month', 'Year']]

#             if not numeric_cols:
#                 st.warning("No numeric columns available for visualization.")
#                 return

#             # Allow multiple column selection
#             visualize_columns = st.multiselect("Select Data to visualize", numeric_cols)

#             if not visualize_columns:
#                 st.warning("Please select at least one metric to visualize.")
#                 return

#             # Aggregation Option
#             aggregation_level = st.selectbox("Select Aggregation Level", ["Weekly", "Monthly"])

#             # Check if the date_column and geo_column are valid
#             if date_column not in df.columns or geo_column not in df.columns:
#                 st.warning("Date or Geo column is not properly selected.")
#                 return

#             try:
#                 # Group by time and sum over selected columns
#                 if aggregation_level == "Weekly":
#                     visualize_trend_time = df.groupby(pd.Grouper(key=date_column, freq='W'))[visualize_columns].sum().reset_index()
#                 else:
#                     visualize_trend_time = df.groupby(pd.Grouper(key=date_column, freq='M'))[visualize_columns].sum().reset_index()

#                 # Keep date column as datetime and format x-axis in Plotly
#                 st.subheader(f"Trend of {', '.join(visualize_columns)} ({aggregation_level})")

#                 # Melt dataframe for Plotly (long format: one row per date/metric/value)
#                 trend_long = visualize_trend_time.melt(id_vars=date_column, value_vars=visualize_columns, 
#                                                     var_name="Metric", value_name="Value")

#                 # Create the Plotly line chart
#                 fig = px.line(
#                     trend_long,
#                     x=date_column,
#                     y="Value",
#                     color="Metric",
#                     markers=True,
#                     title=f"{' & '.join(visualize_columns)} Trend ({aggregation_level})",
#                     labels={
#                         date_column: "Time",
#                         "Value": "Value",
#                         "Metric": "Metric"
#                     }
#                 )

#                 fig.update_layout(
#                     xaxis_tickangle=45,
#                     xaxis=dict(tickformat="%b-%Y") if aggregation_level == "Monthly" else {}
#                 )

#                 st.plotly_chart(fig, use_container_width=True)

#             except Exception as e:
#                 st.error(f"An error occurred while plotting: {e}")

#             # Rolling up data granularity
#             st.subheader("Creating Integrated Analytics Database")

#             if time_granularity == 'Daily':
#                 options = ['Weekly','Monthly']
#             elif time_granularity == 'Weekly':
#                 options = ['Weekly','Monthly']
#             elif time_granularity == 'Monthly':
#                 options=['Monthly']

#             time_granularity_user_input = st.selectbox('Choose the time granularity level',options)

#             st.write('You selected time granularity: ',time_granularity_user_input)

#             return geo_column,date_column, dependent_variable, time_granularity,time_granularity_user_input,df_copy
            
def eda_sales_trend(uploaded_file):

    if uploaded_file is not None:
        # Read CSV File
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Uploaded CSV")
        st.dataframe(df.head(100))

        # Let user select columns
        st.subheader("Select Columns")
        dependent_variable = st.selectbox("Select Dependent Variable", [None] + list(df.columns), index=0 )
        geo_column = st.selectbox("Select Modeling Granularity Column", [None] + list(df.columns), index=0)
        date_column = st.selectbox("Select Date Column", [None] + list(df.columns), index=0)

        if geo_column is not None:
            num_geo = df[geo_column].nunique()
            st.success(f"Number of unique entries in granularity column: {num_geo}")
        
        # Time granularity detection
        if date_column is not None:
            time_granularity = detect_date_granularity(df, date_column )
            st.write(f' The time granularity is : {time_granularity} ')
        
            # Convert to datetime format
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.dropna(subset=[date_column])

            df_copy = df   # Original dataframe
           
            # Set up date range filtering
            min_date = df[date_column].min()
            max_date = df[date_column].max()

            # Convert to datetime.date if needed
            min_date = pd.to_datetime(min_date).date()
            max_date = pd.to_datetime(max_date).date()

            # Side-by-side date selectors
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="start_date"
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="end_date"
                )

            # Validate selected dates
            if start_date > end_date:
                st.warning("⚠️ Start date must be before or equal to end date.")
                st.stop()

            # Filter the DataFrame based on selected range
            df = df[(df[date_column] >= pd.to_datetime(start_date)) & (df[date_column] <= pd.to_datetime(end_date))]

            # Handle case when only one date or empty data
            if df.empty:
                st.warning("No data available for the selected date range. Please choose a broader range.")
                st.stop()

            if (pd.to_datetime(start_date) == pd.to_datetime(end_date)):
                st.warning("Start and end dates are the same. Please select a wider date range for meaningful analysis.")
                st.stop()

            # Control totals    
            st.subheader("Control Totals")

            remove_cols = [date_column, geo_column]
            filtered_df = df.drop(columns=remove_cols)

            subtotal_df = pd.DataFrame({
                'Channel': filtered_df.columns,
                'Total': filtered_df.sum()
            })

            styled_df = subtotal_df.style.format({"Total": "{:,}"})
            st.dataframe(styled_df)

            # Correlation Matrix
            st.subheader("Correlation Matrix")
            corr_matrix = filtered_df.corr()
            fig = sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5).get_figure()
            st.pyplot(fig)

            # Metric to visualize
            st.subheader("Trend visualization")

            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            numeric_cols = [item for item in numeric_cols]

            if not numeric_cols:
                st.warning("No numeric columns available for visualization.")
                return

            visualize_columns = st.multiselect("Select Data to visualize", numeric_cols)

            if not visualize_columns:
                st.warning("Please select at least one metric to visualize.")
                return

            aggregation_level = st.selectbox("Select Aggregation Level", ["Weekly", "Monthly"])

            if date_column not in df.columns or geo_column not in df.columns:
                st.warning("Date or Geo column is not properly selected.")
                return

            try:
                if aggregation_level == "Weekly":
                    visualize_trend_time = df.groupby(pd.Grouper(key=date_column, freq='W'))[visualize_columns].sum().reset_index()
                else:
                    visualize_trend_time = df.groupby(pd.Grouper(key=date_column, freq='M'))[visualize_columns].sum().reset_index()

                st.subheader(f"Trend of {', '.join(visualize_columns)} ({aggregation_level})")

                trend_long = visualize_trend_time.melt(id_vars=date_column, value_vars=visualize_columns, 
                                                    var_name="Metric", value_name="Value")

                fig = px.line(
                    trend_long,
                    x=date_column,
                    y="Value",
                    color="Metric",
                    markers=True,
                    title=f"{' & '.join(visualize_columns)} Trend ({aggregation_level})",
                    labels={date_column: "Time", "Value": "Value", "Metric": "Metric"}
                )

                fig.update_layout(
                    xaxis_tickangle=45,
                    xaxis=dict(tickformat="%b-%Y") if aggregation_level == "Monthly" else {}
                )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred while plotting: {e}")


            # Creation of Integrated Analytics Database:

            st.subheader("Creating Integrated Analytics Database")

            if time_granularity == 'Daily':
                options = ['Weekly','Monthly']
            elif time_granularity == 'Weekly':
                options = ['Weekly','Monthly']
            elif time_granularity == 'Monthly':
                options=['Monthly']

            time_granularity_user_input = st.selectbox('Choose the time granularity level', options)

            st.write('You selected time granularity: ', time_granularity_user_input)

            return geo_column, date_column, dependent_variable, time_granularity, time_granularity_user_input, df_copy

