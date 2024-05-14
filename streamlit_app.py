# Import libraries
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, classification_report
from imblearn.over_sampling import SMOTE
from scipy.stats import skew, kurtosis

# Set page configuration
st.set_page_config(
    page_title="Vehicle Insurance Fraud Detection App",
    page_icon=":shark:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display header and image
image = "vehi.png"
header_col, image_col = st.columns([15, 5])
with header_col:
    st.title("Vehicle Insurance Fraud Detection")
    st.write("Insurer's reliable solution for real-time detection of potential fraudulent claims")
with image_col:
    st.image("vehi.png", width=250)
with st.sidebar:
    st.title('Load Claim Information')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:
    def user_input_features():
        with st.sidebar:
            st.title('Enter Claim Information')

            c1, c2, c3 = st.columns(3)
            with c1:
                Gender = st.radio('Gender', ('M', 'F'))
                HighEducation = st.radio('High Education', ('Yes', 'No'))
                MaritalStatus = st.selectbox('Marital Status', ('Single', 'Married'))
                AnnualIncome = st.number_input('Annual Income', min_value=0)
                AgeofVehicle = st.slider('Age of Vehicle', 0, 50, 0)

            with c2:
                ChangeofAddress = st.radio('Change of Address', ('Yes', 'No'))
                Witness = st.radio('Witness', ('Yes', 'No'))
                ClaimDay = st.selectbox('Claim Day', ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
                ClaimAmount = st.number_input('Claim Amount', min_value=0)
                Liability = st.slider('Liability (%)', 0, 100, 0)

            with c3:
                LivingStatus = st.radio('Living Status', ('Rent', 'Own'))
                PoilceReport = st.radio('Police Report', ('Yes', 'No'))
                AccidentSite = st.selectbox('Accident Site', ('Local','Highway', 'Parking Lot'))
                PastNumberofClaims = st.number_input('Past Number of Claims', min_value=0)
                SafetyRating = st.slider('Safety Rating', 0, 100, 0)

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

if uploaded_file is not None:
    st.write(input_df.iloc[[0]])
else:
    st.write(input_df)

tab1, tab2 = st.tabs(["Prediction", "Exploration"])

with tab1:
    # Extract relevant values from input_df
    Gender = input_df['Gender'].iloc[0]
    ChangeofAddress = input_df['Change of Address'].iloc[0]
    LivingStatus = input_df['Living Status'].iloc[0]
    HighEducation = input_df['High Education'].iloc[0]
    Witness = input_df['Witness'].iloc[0]
    PoliceReport = input_df['Police Report'].iloc[0]
    MaritalStatus = input_df['Marital Status'].iloc[0]
    ClaimDay = input_df['Claim Day'].iloc[0]
    AccidentSite = input_df['Accident Site'].iloc[0]
    AnnualIncome = input_df['Annual Income'].iloc[0]
    ClaimAmount = input_df['Claim Amount'].iloc[0]
    PastNumberofClaims = input_df['Past Number of Claims'].iloc[0]
    AgeofVehicle = input_df['Age of Vehicle'].iloc[0]
    Liability = input_df['Liability (%)'].iloc[0]
    SafetyRating = input_df['Safety Rating'].iloc[0]

    # Convert categorical variables to numerical
    a = 1 if Gender == 'F' else 0
    b = 1 if ChangeofAddress == 'Yes' else 0
    c = 1 if LivingStatus == 'Own' else 0
    d = 1 if HighEducation == 'Yes' else 0
    e = 1 if Witness == 'Yes' else 0
    f = 1 if PoliceReport == 'Yes' else 0
    g = 1 if MaritalStatus == 'Married' else 0
    if ClaimDay == 'Monday': h = 0
    elif ClaimDay == 'Tuesday': h = 1
    elif ClaimDay == 'Wednesday': h = 2
    elif ClaimDay == 'Thursday': h = 3
    elif ClaimDay == 'Friday': h = 4
    elif ClaimDay == 'Saturday': h = 5
    else: h = 6
    if AccidentSite == 'Local': i = 0
    elif AccidentSite == 'Highway': i = 1
    else: i = 2

        # Predict button
    if st.sidebar.button('Predict'):
        input_data = pd.DataFrame({
            'gender': [a],
            'marital_status': [g],
            'safty_rating': [SafetyRating],
            'annual_income': [AnnualIncome],
            'high_education_ind': [d],
            'address_change_ind': [b],
            'living_status': [c],
            'claim_day_of_week': [h],
            'accident_site': [i],
            'past_num_of_claims': [PastNumberofClaims],
            'witness_present_ind': [e],
            'liab_prct': [Liability],
            'policy_report_filed_ind':[f],
            'claim_est_payout': [ClaimAmount],
            'age_of_vehicle': [AgeofVehicle]
        })

        # Make predictions
        try:
            # Load the classification model
            load_gbc = pickle.load(open('gbcfit.pkl', 'rb'))

            # Make predictions
            prediction = load_gbc.predict(input_data)

            # Predict probabilities
            predicted_proba = load_gbc.predict_proba(input_data)

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

        except Exception as e:
            st.error('Welcome! Please enter your claim information correctly on the left.')

with tab2:
    st.title('Exploratory Data Analysis')

    # Import data
    data = pd.read_csv('training data.csv')
    #data = input_df

    # Count the frequency of each category in the chosen variable
    pie_colors = ['#1F77B4', '#FFA500']
    category_counts = data['fraud'].value_counts()
    labels = ["No Fraud", "Fraud"]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(category_counts, labels=labels, autopct=lambda p: '{:.0f} ({:.1f}%)'.format(p * sum(category_counts) / 100, p), startangle=140, explode=[0.05,0],colors=pie_colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

    # Display the plot in Streamlit
    st.subheader('1. No. of Fraudulent Car Claim Data')
    st.pyplot(fig)

    # Select numerical columns excluding label encoded columns
    numerical_features = ['annual_income', 'safty_rating','past_num_of_claims', 'liab_prct', 'claim_est_payout', 'age_of_vehicle']

    # Create a Streamlit figure
    st.subheader("2. Box Plots of Numerical Features")

    # Create separate box plots for each numerical feature
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, feature in enumerate(numerical_features):
        sns.boxplot(x=data[feature], ax=axes[i//3, i%3])
        axes[i//3, i%3].set_title(f'Box Plot of {feature}')
        axes[i//3, i%3].set_xlabel('')
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Filter the columns based on the numerical features
    numerical_data = data[numerical_features]

    # Generate the descriptive statistics for the filtered numerical data
    descriptive_stats = numerical_data.describe().T

    # Display the descriptive statistics on Streamlit
    st.subheader("3. Descriptive Statistics for Numerical Features:")
    st.dataframe(descriptive_stats)

    # Calculate skewness and kurtosis values for numerical features
    skewness_values = data[numerical_features].apply(skew)
    kurtosis_values = data[numerical_features].apply(kurtosis)

    # Display skewness and kurtosis values in a table
    st.subheader("4. Skewness and Kurtosis values for Numerical Features:")
    table_data = {"Feature": numerical_features, "Skewness": skewness_values, "Kurtosis": kurtosis_values}
    st.table(pd.DataFrame(table_data))

    # Visualisation for numerical features
    num_rows = (len(numerical_features) + 1) // 2  # Determine the number of rows for subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 4 * num_rows))
    for i, feature in enumerate(numerical_features):
        row = i // 2
        col = i % 2
        sns.histplot(data=data, x=feature, kde=True, ax=axes[row, col])
        axes[row, col].set_title(f'Distribution of {feature}')
    plt.tight_layout()
    st.pyplot(fig)


    st.subheader("5. Descriptive Statistics for Categorical Features:")
    categorical_features = ['gender', 'marital_status', 'high_education_ind', 'address_change_ind', 'living_status',
                        'claim_day_of_week', 'accident_site', 'witness_present_ind',
                        'policy_report_filed_ind']

    # Filter the columns based on the numerical features
    categorical_data = data[categorical_features].astype('object')

    # Generate the descriptive statistics for the filtered numerical data
    categorical_data.describe(include='object').T

    # Determine the number of rows and columns for subplots
    # Categorical Features
    st.subheader("6. Bar Chart for Categorical Features:")
    num_features = len(categorical_features)
    num_cols = 3
    num_rows = (num_features + num_cols - 1) // num_cols

    # Create the figure and axes
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))

    # Flatten the axes if necessary
    if num_rows == 1:
        axs = [axs]

    # Iterate over categorical features and plot each one
    for i, feature in enumerate(categorical_features):
        row = i // num_cols
        col = i % num_cols
        sns.countplot(data=data, x=feature, ax=axs[row][col])
        axs[row][col].set_title(f'Distribution of {feature}')
        axs[row][col].tick_params(axis='x', rotation=45)

    # Hide any unused subplots
    for i in range(num_features, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axs[row][col].axis('off')

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Bivariate Analysis
    # Numerical Features (vs Fraud)
    st.subheader("7. Bivariate Analysis (Numerical Features vs Fraud)")

    # Determine the number of rows for subplots
    num_features = len(numerical_features)
    num_cols = 2
    num_rows = (num_features + num_cols - 1) // num_cols

    # Create the figure and axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 5 * num_rows))

    # Create separate box plots for each numerical feature with 'fraud' and 'non-fraud' categories
    for i, feature in enumerate(numerical_features):
        row = i // num_cols
        col = i % num_cols
        sns.boxplot(x='fraud', y=feature, data=data, ax=axes[row][col])
        axes[row][col].set_title(f'Box Plot of {feature}')
        axes[row][col].set_xlabel('Fraud')
        axes[row][col].set_ylabel(feature)
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Categorical Features (vs Fraud)
    st.subheader("8. Bivariate Analysis (Categorical Features vs Fraud)")
    # Determine the number of rows for subplots
    num_rows = (len(categorical_features) + 2) // 3

    # Create a figure
    fig, axes = plt.subplots(num_rows, 3, figsize=(18, 5 * num_rows))

    # Flatten the axes if necessary
    if num_rows == 1:
        axes = [axes]

    # Iterate over categorical features and plot each one
    for i, feature in enumerate(categorical_features, 1):
        row = (i - 1) // 3
        col = (i - 1) % 3
        sns.countplot(data=data, x=feature, hue='fraud', ax=axes[row, col])
        axes[row, col].set_title(f'Distribution of {feature} by Fraud')
        axes[row, col].tick_params(axis='x', rotation=45)

    # Hide any unused subplots
    for i in range(len(categorical_features), num_rows * 3):
        row = i // 3
        col = i % 3
        axes[row, col].axis('off')

    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    # Calculate the correlation matrix
    st.subheader("9. Correlation Matrix")
    # Drop variables that are not having significant correlation with DV
    data.drop(['claim_date','vehicle_price', 'vehicle_weight','channel', 'vehicle_category','vehicle_color','age_of_driver', 'claim_number', 'zip_code'], inplace = True, axis = 1)
    # Label encoding
    data.replace({'gender': {'M': '0', 'F': '1'},
              'living_status': {'Rent': '0', 'Own': '1'},
              'channel': {'Broker': '0', 'Online': '1', 'Phone': '2'},
              'vehicle_category': {'Compact': '0', 'Large': '1', 'Medium': '2'},
              'accident_site': {'Local': '0', 'Highway': '1', 'Parking Lot': '2'},
              'vehicle_color': {'white': '0', 'silver': '1', 'gray': '2', 'black': '3', 'red': '4', 'blue': '5', 'other': '6'},
              'claim_day_of_week': {'Monday': '0', 'Tuesday': '1', 'Wednesday': '2', 'Thursday': '3', 'Friday': '4', 'Saturday': '5', 'Sunday': '6'}}, inplace=True)

    # Calculate the correlation matrix
    corr_matrix = data.corr()

    # Create the heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='flare')

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
