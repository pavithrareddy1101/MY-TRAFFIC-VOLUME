import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime

# Page configuration
st.set_page_config(page_title="TrafficTelligence - Dark Mode", layout="wide")

# Inject custom dark theme CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #BB86FC;
    }
    .css-1v3fvcr, .css-10trblm, .st-bw {
        color: #e0e0e0;
    }
    .stButton>button {
        background-color: #3700B3;
        color: white;
    }
    .stSelectbox, .stSlider, .stRadio, .stTextInput, .stNumberInput {
        background-color: #1F1F1F;
        color: white;
    }
    .stDataFrame {
        background-color: #1E1E1E;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🚦 TrafficTelligence - Andhra Pradesh Trip Predictor (Dark Mode)")

st.sidebar.header("🔧 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload 'cleaned_green_tripdata_2025_03.csv'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='ascii')
    st.success("✅ Dataset Loaded Successfully")

    with st.expander("📊 Dataset Preview"):
        st.dataframe(df, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["📈 Visuals", "🧠 Model & Metrics", "🚗 Predict"])

    with tab1:
        st.header("📊 Visual Insights")

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⏱️ Trip Duration Distribution")
            fig1, ax1 = plt.subplots()
            sns.set_theme(style="darkgrid")
            sns.histplot(df['trip_duration'], bins=30, kde=True, color="#BB86FC", ax=ax1)
            ax1.set_xlabel("Trip Duration (minutes)")
            st.pyplot(fig1)

        with col2:
            st.subheader("🛣️ Distance vs Duration (Rush Hour)")
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=df, x='trip_distance', y='trip_duration', hue='is_rush_hour', palette='coolwarm', ax=ax2)
            ax2.set_xlabel("Trip Distance (km)")
            st.pyplot(fig2)

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("🕘 Average Duration by Hour")
            avg_duration_by_hour = df.groupby('pickup_hour')['trip_duration'].mean().reset_index()
            fig3, ax3 = plt.subplots()
            sns.barplot(data=avg_duration_by_hour, x='pickup_hour', y='trip_duration', palette="mako", ax=ax3)
            ax3.set_xlabel("Pickup Hour")
            st.pyplot(fig3)

        with col4:
            st.subheader("📅 Weekend vs Weekday Duration")
            fig4, ax4 = plt.subplots()
            sns.boxplot(data=df, x='is_weekend', y='trip_duration', palette='pastel', ax=ax4)
            ax4.set_xticklabels(['Weekday', 'Weekend'])
            st.pyplot(fig4)

    with tab2:
        st.header("🧠 Model Training & Results")

        features = ['trip_distance', 'pickup_hour', 'is_weekend', 'is_rush_hour']
        target = 'trip_duration'

        if all(col in df.columns for col in features + [target]):
            X = df[features]
            Y = df[target]

            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            model = LinearRegression()
            model.fit(X_train, Y_train)

            Y_pred = model.predict(X_test)
            mse = mean_squared_error(Y_test, Y_pred)
            r2 = r2_score(Y_test, Y_pred)

            st.metric("Mean Squared Error", f"{mse:.2f}")
            st.metric("R-squared (R²)", f"{r2:.4f}")

        else:
            st.error("Required columns not found in dataset.")

    with tab3:
        st.header("🚗 Predict Trip Duration")

        if 'model' not in locals():
            st.warning("Upload dataset and train model first.")
        else:
            with st.form("predict_form"):
                source = st.text_input("From", value="Vijayawada")
                destination = st.text_input("To", value="Guntur")
                trip_distance = st.number_input("Distance (km)", min_value=0.0, value=20.0)
                pickup_hour = st.slider("Pickup Hour", 0, 23, 9)
                is_weekend = st.radio("Weekend?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                is_rush_hour = st.radio("Rush Hour?", [0, 1], format_func=lambda x: "Yes" if x else "No")
                vehicle = st.selectbox("Vehicle Type", ["Car", "Bike", "Bus", "Truck"])
                
                submitted = st.form_submit_button("🔍 Predict Duration")

            if submitted:
                input_data = pd.DataFrame([[trip_distance, pickup_hour, is_weekend, is_rush_hour]], columns=features)
                predicted_duration = model.predict(input_data)[0]

                if vehicle.lower() == "bike":
                    predicted_duration *= 0.8
                elif vehicle.lower() == "bus":
                    predicted_duration *= 1.2
                elif vehicle.lower() == "truck":
                    predicted_duration *= 1.5

                traffic_level = "🔴 High" if predicted_duration > 90 else "🟠 Moderate" if predicted_duration > 45 else "🟢 Low"
                traffic_volume = int(50 + predicted_duration * 2 + 30)

                st.success("✅ Prediction Complete")
                st.metric("Estimated Duration (minutes)", f"{predicted_duration:.1f}")
                st.metric("Traffic Level", traffic_level)
                st.metric("Vehicle Volume", f"{traffic_volume} vehicles")

                with st.expander("Trip Summary"):
                    st.write(f"**Date & Time:** {datetime.now().strftime('%A, %d %B %Y, %I:%M %p')}")
                    st.write(f"**From:** {source} → **To:** {destination}")
                    st.write(f"**Distance:** {trip_distance} km")

else:
    st.info("Upload your dataset to begin.")