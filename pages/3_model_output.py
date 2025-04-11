# import streamlit as st

# # Page for displaying the regression results
# def display_regression_results():
#     if 'regression_outputs' in st.session_state and st.session_state['regression_outputs']:
#         st.subheader("Previous Regression Results")

#         # Create a scrollable container for regression outputs
#         for idx, result in enumerate(st.session_state['regression_outputs']):
#             with st.expander(f"Run {idx+1}"):
#                 # Display the regression summary (OLS text)
#                 st.subheader("OLS Regression Summary")
#                 st.text(result['summary'])

#                 # Display the coefficient table as a dataframe
#                 st.subheader("Coefficients Table")
#                 st.dataframe(result['coefficients'])
#     else:
#         st.warning("No regression results found. Please run a regression first.")

# # Run the Streamlit app
# if __name__ == '__main__':
#     st.title("Model Output")
#     display_regression_results()

import streamlit as st

# Page for displaying and selecting regression results
def display_regression_results():
    if 'regression_outputs' in st.session_state and st.session_state['regression_outputs']:
        st.subheader("Previous Regression Results")

        # Dropdown to select model run
        model_labels = [f"Iteration {i+1}" for i in range(len(st.session_state['regression_outputs']))]
        selected_run_label = st.selectbox("Select the best model iteration", model_labels)

        # Get index of selected model
        selected_index = model_labels.index(selected_run_label)
        selected_result = st.session_state['regression_outputs'][selected_index]

        # Display selected model details
        st.subheader("Selected Model: " + selected_run_label)


        # Display configuration table
        st.subheader('Configuration Table')
        configuration_list = st.session_state['configuration_list']
        st.dataframe(configuration_list[selected_index])

        st.subheader("OLS Regression Summary")
        st.code(selected_result['summary'],language='text')

        st.subheader("Coefficients Table")
        st.dataframe(selected_result['coefficients'])

        

        # Option to save coefficients for response curve generation
        if st.button("Save this model's coefficients for response curve generation"):
            st.session_state['selected_model_coefficients'] = selected_result['coefficients']
            st.success("Coefficients saved for response curve generation!")

    else:
        st.warning("No regression results found. Please run a regression first.")

# Run the Streamlit app
if __name__ == '__main__':
    st.title("Model Output")
    display_regression_results()