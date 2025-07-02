import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
import os

# Set up the Streamlit app
st.set_page_config(page_title="Machine Learning Model for Cancer Data", layout="wide")

st.title("🩺 Machine Learning Model for Cancer Data")

# Sidebar for application information
st.sidebar.title("About")
st.sidebar.info("""
This application uses an ensemble of machine learning models 
to predict cancer-related data using various 
clinical and biological features.
""")

@st.cache_data
def load_data():
    """Load data with multiple options"""
    # Try to load from repository first
    try:
        X = pd.read_csv("X.csv")
        y = pd.read_csv("y.csv")
        return X, y, "repository"
    except FileNotFoundError:
        try:
            X = pd.read_csv("data/X.csv")
            y = pd.read_csv("data/y.csv")
            return X, y, "repository"
        except FileNotFoundError:
            return None, None, "upload_needed"
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None, None, "error"

@st.cache_resource
def train_model(X_train, y_train):
    """Train the ensemble model"""
    # Define the models
    xgb_best = XGBClassifier(
        learning_rate=0.2, 
        max_depth=3, 
        n_estimators=100, 
        subsample=0.8,
        random_state=2090
    )
    
    cat_best = CatBoostClassifier(
        verbose=0,
        depth=4, 
        iterations=200, 
        learning_rate=0.1,
        random_state=2090
    )
    
    lgbm_best = LGBMClassifier(
        learning_rate=0.2, 
        max_depth=3, 
        n_estimators=200,
        random_state=2090
    )
    
    rf = RandomForestClassifier(
        random_state=2025,
        bootstrap=True, 
        max_depth=None, 
        min_samples_leaf=1, 
        min_samples_split=5, 
        n_estimators=100
    )
    
    # Create a voting classifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_best),
            ('cat', cat_best),
            ('lgbm', lgbm_best),
            ('rf', rf)
        ],
        voting='soft',
        weights=(1, 1, 4, 1)
    )
    
    with st.spinner("🔄 Training model in progress..."):
        ensemble_model.fit(X_train, y_train.values.ravel())
    
    return ensemble_model

# Try to load data
X, y, load_status = load_data()

# If data loading failed, show file upload option
if load_status == "upload_needed":
    st.warning("📁 Data files not found in repository. Please upload your CSV files.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload X.csv")
        uploaded_X = st.file_uploader("Choose X.csv file", type="csv", key="X")
        
    with col2:
        st.subheader("Upload y.csv")
        uploaded_y = st.file_uploader("Choose y.csv file", type="csv", key="y")
    
    if uploaded_X is not None and uploaded_y is not None:
        try:
            X = pd.read_csv(uploaded_X)
            y = pd.read_csv(uploaded_y)
            st.success("✅ Files uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading uploaded files: {str(e)}")
            st.stop()
    else:
        st.info("👆 Please upload both X.csv and y.csv files to continue.")
        st.markdown("""
        **Instructions:**
        1. Upload your X.csv file (features)
        2. Upload your y.csv file (target variable)
        3. The application will automatically process the data
        """)
        st.stop()

elif load_status == "error":
    st.stop()

# Define features
features = ['age_at_diagnosis', 'taille', 'CA19-9', 'ACE', 'volume_colon',
           'nombre_metastaes', 'capécitabine', 'oxaliplatine', 'irinotécan',
           'bévacizumab', 'panitumumab', 'cetuximab', 'chimiothérapie_exclusive',
           'chirurgie', 'tumor_size_total', 'RAS_0.0', 'RAS_2.0']

# Check that all columns exist
missing_features = [f for f in features if f not in X.columns]
if missing_features:
    st.error(f"❌ Missing columns in data: {missing_features}")
    st.info("Available columns: " + ", ".join(X.columns.tolist()))
    st.stop()

X = X[features]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2090, stratify=y
)

# Train model
ensemble_model = train_model(X_train, y_train)

# User interface for data input
st.header("📊 Enter Patient Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographic and Clinical Data")
    age_at_diagnosis = st.slider("Age at diagnosis", min_value=0, max_value=5, value=2, help="Patient's age at time of diagnosis")
    taille = st.slider("Height", min_value=0, max_value=4, value=2, help="Patient's height")
    
    st.subheader("Biological Markers")
    CA19_9 = st.slider("CA19-9", min_value=0, max_value=4, value=2, help="CA19-9 tumor marker")
    ACE = st.slider("CEA", min_value=0, max_value=4, value=2, help="Carcinoembryonic antigen")
    
    st.subheader("Tumor Characteristics")
    volume_colon = st.slider("Colon volume", min_value=0, max_value=4, value=2)
    nombre_metastaes = st.slider("Number of metastases", min_value=0, max_value=5, value=2)
    tumor_size_total = st.slider("Total tumor size", min_value=0.0, max_value=4.0, value=2.0)

with col2:
    st.subheader("Drug Treatments")
    capécitabine = st.selectbox("Capecitabine", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oxaliplatine = st.selectbox("Oxaliplatin", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    irinotécan = st.selectbox("Irinotecan", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    bévacizumab = st.selectbox("Bevacizumab", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    panitumumab = st.selectbox("Panitumumab", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cetuximab = st.selectbox("Cetuximab", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    st.subheader("Treatment Types")
    chimiothérapie_exclusive = st.selectbox("Chemotherapy only", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    chirurgie = st.selectbox("Surgery", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    st.subheader("Genetic Markers")
    RAS_0_0 = st.selectbox("RAS 0.0", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    RAS_2_0 = st.selectbox("RAS 2.0", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Create input DataFrame
test_app = pd.DataFrame({
    'age_at_diagnosis': [age_at_diagnosis],
    'taille': [taille],
    'CA19-9': [CA19_9],
    'ACE': [ACE],
    'volume_colon': [volume_colon],
    'nombre_metastaes': [nombre_metastaes],
    'capécitabine': [capécitabine],
    'oxaliplatine': [oxaliplatine],
    'irinotécan': [irinotécan],
    'bévacizumab': [bévacizumab],
    'panitumumab': [panitumumab],
    'cetuximab': [cetuximab],
    'chimiothérapie_exclusive': [chimiothérapie_exclusive],
    'chirurgie': [chirurgie],
    'tumor_size_total': [tumor_size_total],
    'RAS_0.0': [RAS_0_0],
    'RAS_2.0': [RAS_2_0]
})

# Display entered data
st.subheader("📋 Summary of Entered Data")
st.dataframe(test_app, use_container_width=True)

# Prediction button
if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
    try:
        with st.spinner("⏳ Computing..."):
            prediction = ensemble_model.predict(test_app)
            prediction_proba = ensemble_model.predict_proba(test_app)
        
        # Display results
        st.header("🎯 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Predicted Class")
            st.success(f"**Result: {prediction[0]}**")
        
        with col2:
            st.subheader("Probabilities by Class")
            proba_df = pd.DataFrame(
                prediction_proba, 
                columns=[f"Class {cls}" for cls in ensemble_model.classes_]
            ).round(4)
            st.dataframe(proba_df, use_container_width=True)
        
        # Probability chart
        st.subheader("📊 Probability Visualization")
        chart_data = pd.DataFrame({
            'Class': [f"Class {cls}" for cls in ensemble_model.classes_],
            'Probability': prediction_proba[0]
        })
        st.bar_chart(chart_data.set_index('Class'))
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")


# Footer
st.markdown("---")
st.markdown("*Application developed with Streamlit for predictive analysis in oncology*")
