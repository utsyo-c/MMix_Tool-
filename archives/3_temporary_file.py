import streamlit as st
st.title("Test File: Please incorporate the code into response curve file")


coeffs_df = st.session_state.get('selected_model_coefficients')
if coeffs_df is not None:
    # Proceed with using coeffs_df
    st.write("Loaded saved coefficients for response curve:")
    st.dataframe(coeffs_df)
else:
    st.warning("No saved coefficients found.")