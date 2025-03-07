import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

# Title and description
st.title("Personal Fitness Tracker")
st.markdown("""
    In this WebApp, you will be able to predict the number of kilocalories burned based on your input parameters such as:
    - Age
    - Gender
    - BMI
    - Heart Rate
    - Exercise Duration
    - Body Temperature
    """)

# Sidebar with user input form
st.sidebar.header("User Input Parameters")

def user_input_features():
    # Collect user input for prediction
    
    # Age input with range instruction
    age = st.sidebar.text_input("Age", "30")
    age = int(age) if age.isdigit() else 30  # Default to 30 if input is invalid
    st.sidebar.markdown("**Age**: Enter your age (10-100 years). Default is 30.")
    
    # BMI input with range instruction
    bmi = st.sidebar.text_input("BMI", "20")
    bmi = float(bmi) if bmi.replace('.', '', 1).isdigit() else 20.0  # Default to 20 if input is invalid
    st.sidebar.markdown("**BMI**: Enter your BMI (15-40). Default is 20.")
    
    # Exercise Duration selection with range instruction
    duration = st.sidebar.selectbox("Exercise Duration (minutes)", options=[15, 30, 45, 60, 90], index=1)
    st.sidebar.markdown("**Exercise Duration**: Select how long you exercise (15-90 minutes). Default is 30 minutes.")
    
    # Heart Rate input with range instruction
    heart_rate = st.sidebar.text_input("Heart Rate (bpm)", "80")
    heart_rate = int(heart_rate) if heart_rate.isdigit() else 80  # Default to 80 if input is invalid
    st.sidebar.markdown("**Heart Rate**: Enter your heart rate (60-200 bpm). Default is 80 bpm.")
    
    # Body Temperature input with range instruction
    body_temp = st.sidebar.text_input("Body Temperature (°C)", "37")
    body_temp = float(body_temp) if body_temp.replace('.', '', 1).isdigit() else 37.0  # Default to 37°C if invalid
    st.sidebar.markdown("**Body Temperature**: Enter your body temperature (36-42°C). Default is 37°C.")
    
    # Gender as radio button
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))
    st.sidebar.markdown("**Gender**: Select your gender (Male/Female).")
    
    # Encode gender (Male=1, Female=0)
    gender = 1 if gender_button == "Male" else 0

    # Create a dataframe with user inputs
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }
    
    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

# Tabs for different sections of the app
tabs = st.selectbox("Choose the section", ["Input Parameters", "Prediction", "Results"])

# Load the dataset and train the model
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Calculate BMI
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

# Prepare the training and testing sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

# One-hot encoding
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the RandomForestRegressor model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)

# Display input parameters
if tabs == "Input Parameters":
    st.write("### Your Input Parameters")
    st.write(df)

# Display prediction
if tabs == "Prediction":
    st.write("### Prediction")
    
    # Show progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.02)
    
    st.write(f"Predicted Calories Burned: {round(prediction[0], 2)} kilocalories")

# Display similar results based on predicted calories
if tabs == "Results":
    st.write("### Similar Results")
    
    # Show progress bar
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.02)
    
    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
    
    st.write("Showing a random sample of similar results:")
    st.write(similar_data.sample(5))

    # Show general statistics based on input comparison
    st.write("### General Information Comparison:")
    boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
    boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
    boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
    boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()
    
    st.write(f"You are older than {round(sum(boolean_age) / len(boolean_age), 2) * 100}% of people.")
    st.write(f"Your exercise duration is higher than {round(sum(boolean_duration) / len(boolean_duration), 2) * 100}% of people.")
    st.write(f"You have a higher heart rate than {round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100}% of people.")
    st.write(f"You have a higher body temperature than {round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100}% of people.")
