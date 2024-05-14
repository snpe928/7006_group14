# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Vehicle Insurance Fraud Detection App",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display header and image
st.title("Vehicle Insurance Fraud Detection")
st.write("Insurer's reliable solution for real-time detection of potential fraudulent claims")
st.sidebar.title('Load Claim Information')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    def user_input_features():
        st.sidebar.title('Enter Claim Information')

        Gender = st.sidebar.radio('Gender', ('M', 'F'))
        HighEducation = st.sidebar.radio('High Education', ('Yes', 'No'))
        MaritalStatus = st.sidebar.selectbox('Marital Status', ('Single', 'Married'))
        AnnualIncome = st.sidebar.number_input('Annual Income', min_value=0)
        AgeofVehicle = st.sidebar.slider('Age of Vehicle', 0, 50, 0)
        ChangeofAddress = st.sidebar.radio('Change of Address', ('Yes', 'No'))
        Witness = st.sidebar.radio('Witness', ('Yes', 'No'))
        ClaimDay = st.sidebar.selectbox('Claim Day', ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
        ClaimAmount = st.sidebar.number_input('Claim Amount', min_value=0)
        Liability = st.sidebar.slider('Liability (%)', 0, 100, 0)
        LivingStatus = st.sidebar.radio('Living Status', ('Rent', 'Own'))
        PoilceReport = st.sidebar.radio('Police Report', ('Yes', 'No'))
        AccidentSite = st.sidebar.selectbox('Accident Site', ('Local','Highway', 'Parking Lot'))
        PastNumberofClaims = st.sidebar.number_input('Past Number of Claims', min_value=0)
        SafetyRating = st.sidebar.slider('Safety Rating', 0, 100, 0)

        # Creating a DataFrame with user input
        features = pd.DataFrame({
            'Gender': [Gender],
            'High Education': [HighEducation],
            'Marital Status': [MaritalStatus],
            'Annual Income': [AnnualIncome],
            'Age of Vehicle': [AgeofVehicle],
            'Change of Address': [ChangeofAddress],
            'Witness': [Witness],
            'Claim Day': [ClaimDay],
            'Claim Amount': [ClaimAmount],
            'Liability (%)': [Liability],
            'Living Status': [LivingStatus],
            'Police Report': [PoilceReport],
            'Accident Site': [AccidentSite],
            'Past Number of Claims': [PastNumberofClaims],
            'Safety Rating': [SafetyRating]
        })
        return features

    input_df = user_input_features()

# Displays the user input features
st.subheader('User Input features')
st.write(input_df)

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    return pickle.load(open('gbcfit.pkl', 'rb'))

model = load_model()

# Predict function
def predict(model, input_data):
    # Make predictions
    prediction = model.predict(input_data)
    # Predict probabilities
    predicted_proba = model.predict_proba(input_data)

    return prediction, predicted_proba

# Predict button
if st.sidebar.button('Predict'):
    input_data = pd.DataFrame({
        'gender': [1 if input_df['Gender'].iloc[0] == 'F' else 0],
        'marital_status': [1 if input_df['Marital Status'].iloc[0] == 'Married' else 0],
        'safty_rating': [input_df['Safety Rating'].iloc[0]],
        'annual_income': [input_df['Annual Income'].iloc[0]],
        'high_education_ind': [1 if input_df['High Education'].iloc[0] == 'Yes' else 0],
        'address_change_ind': [1 if input_df['Change of Address'].iloc[0] == 'Yes' else 0],
        'living_status': [1 if input_df['Living Status'].iloc[0] == 'Own' else 0],
        'claim_day_of_week': [st.sidebar.selectbox('Claim Day', list(range(7)))],
        'accident_site': [st.sidebar.selectbox('Accident Site', list(range(3)))],
        'past_num_of_claims': [input_df['Past Number of Claims'].iloc[0]],
        'witness_present_ind': [1 if input_df['Witness'].iloc[0] == 'Yes' else 0],
        'liab_prct': [input_df['Liability (%)'].iloc[0]],
        'policy_report_filed_ind':[1 if input_df['Police Report'].iloc[0] == 'Yes' else 0],
        'claim_est_payout': [input_df['Claim Amount'].iloc[0]],
        'age_of_vehicle': [input_df['Age of Vehicle'].iloc[0]]
    })

    # Make predictions
    prediction, predicted_proba = predict(model, input_data)

    st.write('### Prediction Probability:')
    st.dataframe(pd.DataFrame(predicted_proba, columns=['Non-Fraud', 'Fraud']), hide_index=True, width=2000)

    # Set threshold
    threshold = 0.5  # Can adjust if needed

    # Classify predictions based on the threshold
    if predicted_proba[0][1] > threshold + 0.4:
        st.error('**Fraudulent Claim Detected!** \n\n\n *Action Item*: \n\n\n 1. Immediately suspend processing of the fraudulent claim \n\n\n 2. Consult legal counsel and take appropriate legal action \n\n\n 3. Blacklist the customer from purchasing future insurance coverage')
    elif predicted_proba[0][1] > threshold:
        st.warning('**Suspicious Claim** \n\n\n *Action Item*: \n\n\n 1. Flag suspicious claims and escalate to fraud prevention team for review \n\n\n 2. Gather additional information and clarify any inconsistencies with customer  \n\n\n 3. Conduct investigation by gathering evidence and contacting relevant parties')
    else:
        st.success('**Non-Fraudulent Claim** \n\n\n *Action Item*: \n\n\n 1. Process the claim amount accordingly on FileaClaim portal\n\n\n 2. Notify customer the status of their claim once it is processed \n\n\n 3. Gather feedback from customer after their claim is resolved')
