import streamlit as st
import mysql.connector
import hashlib
import tensorflow as tf
import numpy as np
from PIL import Image
import io


# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="project"
    )


# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Authentication functions
def login_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, hash_password(password)))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return user


def signup_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hash_password(password)))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        st.error(f"Error: {err}")
        return False
    finally:
        cursor.close()
        conn.close()


# Load the model globally to avoid reloading it on every prediction
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.keras")


model = load_model()


# Define pages
def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state['authenticated'] = True
            st.session_state['username'] = username

        else:
            st.error("Incorrect username or password")


def signup_page():
    st.title("Signup Page")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")

    if st.button("Sign Up"):
        if signup_user(new_username, new_password):
            st.success("User registered successfully. Please login.")



def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


def main_page():
    st.title("Main Page")
    st.markdown(f"<h1 style='font-size: 24px;'>welcome {st.session_state['username']}! 👋</h1>", unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.session_state['username'] = ""
    # st.experimental_rerun()

    # Main Page
    if app_mode == "Home":
        st.header("PLANT DISEASE RECOGNITION SYSTEM")
        image_path = "home_page.jpeg"
        st.image(image_path, use_column_width=True)
        st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

    elif app_mode == "About":
        st.header("About")
        st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

    elif app_mode == "Disease Recognition":
        st.header("Disease Recognition")
        test_image = st.file_uploader("Choose an Image:")
        if test_image:
            st.image(test_image, use_column_width=True)
            if st.button("Predict"):
                result_index = model_prediction(test_image)
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                              'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
                              'Corn_(maize)___healthy',
                              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                              'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                              'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                              'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                              'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                              'Tomato___Tomato_mosaic_virus',
                              'Tomato___healthy']
                st.success("Model is Predicting it's a {}".format(class_name[result_index]))

               

            # Main application logic
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if not st.session_state['authenticated']:
    option = st.sidebar.selectbox("Choose Page", ["Login", "Signup"])
    if option == "Login":
        login_page()
    else:
        signup_page()
else:
    main_page()
