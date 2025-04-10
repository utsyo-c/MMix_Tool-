import streamlit as st

# Page for displaying the regression results
def display_regression_results():
    if 'regression_outputs' in st.session_state and st.session_state['regression_outputs']:
        st.subheader("Previous Regression Results")

        # Create a scrollable container for regression outputs
        for idx, result in enumerate(st.session_state['regression_outputs']):
            with st.expander(f"Run {idx+1}"):
                # Display the regression summary (OLS text)
                st.subheader("OLS Regression Summary")
                st.text(result['summary'])

                # Display the coefficient table as a dataframe
                st.subheader("Coefficients Table")
                st.dataframe(result['coefficients'])
    else:
        st.warning("No regression results found. Please run a regression first.")

# Run the Streamlit app
if __name__ == '__main__':
    st.title("Model Output")
    display_regression_results()