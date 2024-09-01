import streamlit as st
import mysql.connector
import hashlib
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Constants
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE_KB = 100  

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


@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("trained_model.keras")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size(file):
    file.seek(0, io.SEEK_END)  
    size = file.tell() 
    file.seek(0)  
    return size

def model_prediction(test_image, model):
    try:
        image = Image.open(test_image).convert('RGB')
        image = image.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  
        predictions = model.predict(input_arr)
        confidence = np.max(predictions) 
        result_index = np.argmax(predictions) 
        return result_index, confidence
    except Exception as e:
        st.error(f"Error in model prediction: {e}")
        return None, None


disease_info = {
    'Disease: Apple scab': {
        'description': 'Apple scab is a fungal disease that causes dark, sunken lesions on apples and other fruit crops, leading to reduced yield and quality.',
        'supplement': 'Fungicides such as captan or copper-based solutions. Additionally, consider using systemic fungicides for better control.'
    },
    'Apple___Black_rot': {
        'description': 'Black rot is a fungal disease that causes dark, sunken lesions on apple fruit and leaves, leading to reduced yield and fruit quality.',
        'supplement': 'Fungicides like thiophanate-methyl or pyraclostrobin. Pruning affected branches and ensuring proper air circulation can also help.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Cedar apple rust is a fungal disease that causes yellowish spots on apple leaves and fruit, leading to premature leaf drop and reduced fruit quality.',
        'supplement': 'Fungicides such as chlorothalonil or myclobutanil. Removing nearby cedar trees (alternate hosts) can reduce disease pressure.'
    },
    'Apple___healthy': {
        'description': 'The apple plant is healthy with no visible signs of disease.',
        'supplement': 'Continue with regular care and maintenance. No additional supplements are necessary unless disease symptoms appear.'
    },
    'Blueberry___healthy': {
        'description': 'The blueberry plant is healthy with no visible signs of disease.',
        'supplement': 'Maintain good cultural practices and regular monitoring for any signs of disease or pests.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'description': 'Powdery mildew is a fungal disease that causes a white, powdery substance to cover cherry leaves and fruit, leading to reduced growth and fruit quality.',
        'supplement': 'Apply fungicides like sulfur or potassium bicarbonate. Ensure proper spacing between plants to improve air circulation.'
    },
    'Cherry_(including_sour)___healthy': {
        'description': 'The cherry plant is healthy with no visible signs of disease.',
        'supplement': 'Regular maintenance and monitoring are advised to prevent any potential issues.'
    },
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot': {
        'description': 'Cercospora leaf spot, or gray leaf spot, is a fungal disease that causes grayish-brown lesions on corn leaves, leading to reduced photosynthesis and yield.',
        'supplement': 'Use fungicides like propiconazole or azoxystrobin. Implement crop rotation and avoid overhead irrigation to reduce disease spread.'
    },
    'Corn_(maize)___Common_rust_': {
        'description': 'Common rust is a fungal disease that causes reddish-brown pustules on corn leaves, reducing photosynthesis and overall plant health.',
        'supplement': 'Fungicides such as tebucanozole or chlorothalonil can be effective. Selecting resistant corn varieties can also help.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Northern leaf blight is a fungal disease that causes large, grayish-green lesions on corn leaves, leading to reduced photosynthesis and yield.',
        'supplement': 'Apply fungicides like pyraclostrobin or tebuconazole. Rotate crops and manage residue to minimize the risk.'
    },
    'Corn_(maize)___healthy': {
        'description': 'The corn plant is healthy with no visible signs of disease.',
        'supplement': 'Continue regular care and monitoring. No additional supplements are necessary unless disease symptoms appear.'
    },
    'Grape___Black_rot': {
        'description': 'Black rot is a fungal disease that causes dark, sunken lesions on grape berries and leaves, reducing fruit quality and yield.',
        'supplement': 'Apply fungicides like myclobutanil or copper-based products. Ensure proper pruning and spacing to improve airflow.'
    },
    'Grape___Esca_(Black_Measles)': {
        'description': 'Esca, or black measles, is a fungal disease that causes dark lesions and wood rot in grapevines, leading to reduced fruit quality and vine health.',
        'supplement': 'Use fungicides like fenhexamid or sulfur. Implement good vineyard management practices and remove affected vines.'
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'description': 'Leaf blight is a fungal disease that causes angular, brown lesions on grape leaves, leading to premature leaf drop and reduced vine health.',
        'supplement': 'Apply fungicides such as copper-based products or chlorothalonil. Maintain proper vineyard sanitation and pruning.'
    },
    'Grape___healthy': {
        'description': 'The grape vine is healthy with no visible signs of disease.',
        'supplement': 'Continue regular vineyard management practices and monitor for any potential issues.'
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'description': 'Huanglongbing (citrus greening) is a bacterial disease that causes yellowing and misshapen fruit on orange trees, leading to severe yield loss and tree decline.',
        'supplement': 'There is no effective cure for HLB. Use preventative measures such as insect control and removing infected trees to manage spread.'
    },
    'Peach___Bacterial_spot': {
        'description': 'Bacterial spot is a bacterial disease that causes dark, sunken spots on peach fruit and leaves, leading to reduced fruit quality and yield.',
        'supplement': 'Apply copper-based bactericides or antibiotics. Ensure good air circulation and avoid overhead irrigation.'
    },
    'Peach___healthy': {
        'description': 'The peach tree is healthy with no visible signs of disease.',
        'supplement': 'Regular care and monitoring are advised to maintain plant health and prevent diseases.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'description': 'Bacterial spot is a bacterial disease that causes dark, sunken spots on bell pepper fruit and leaves, reducing fruit quality and yield.',
        'supplement': 'Use copper-based bactericides or antibiotics. Implement crop rotation and avoid splashing water on plants.'
    },
    'Pepper,_bell___healthy': {
        'description': 'The bell pepper plant is healthy with no visible signs of disease.',
        'supplement': 'Continue with regular care and monitoring to prevent potential issues.'
    },
    'Potato___Early_blight': {
        'description': 'Early blight is a fungal disease that causes dark, concentric lesions on potato leaves, reducing photosynthesis and yield.',
        'supplement': 'Apply fungicides like chlorothalonil or azoxystrobin. Rotate crops and manage irrigation to reduce disease spread.'
    },
    'Potato___healthy': {
        'description': 'The potato plant is healthy with no visible signs of disease.',
        'supplement': 'Regular care and monitoring are recommended to maintain plant health.'
    },
    'Squash___Powdery_mildew': {
        'description': 'Powdery mildew is a fungal disease that causes a white, powdery coating on squash leaves and fruit, leading to reduced plant health and yield.',
        'supplement': 'Use fungicides like sulfur or potassium bicarbonate. Improve air circulation and avoid overhead watering.'
    },
    'Squash___healthy': {
        'description': 'The squash plant is healthy with no visible signs of disease.',
        'supplement': 'Continue regular care and monitoring to prevent disease.'
    },
    'Tomato___Bacterial_spot': {
        'description': 'Bacterial spot is a bacterial disease that causes dark, sunken spots on tomato leaves and fruit, reducing quality and yield.',
        'supplement': 'Apply copper-based bactericides or antibiotics. Implement crop rotation and avoid splashing water on plants.'
    },
    'Tomato___Early_blight': {
        'description': 'Early blight is a fungal disease that causes dark, concentric lesions on tomato leaves, reducing photosynthesis and yield.',
        'supplement': 'Apply fungicides like chlorothalonil or tebuconazole. Ensure proper air circulation and avoid overhead irrigation.'
    },
    'Tomato___Late_blight': {
        'description': 'Late blight is a fungal disease that causes dark, water-soaked lesions on tomato leaves and fruit, leading to severe plant decline and yield loss.',
        'supplement': 'Use fungicides like mefenoxam or copper-based products. Remove and destroy infected plant debris and practice crop rotation.'
    },
    'Tomato___healthy': {
        'description': 'The tomato plant is healthy with no visible signs of disease.',
        'supplement': 'Regular care and monitoring are advised to prevent potential issues.'
    }
}

def login_page():
    st.title("Login Page")
    username = st.text_input("Username")
    DOB=st.text_input('dd/mm/yyyy')
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
           

def main_page():
    st.title("Main Page")
    st.markdown(f"<h1 style='font-size: 24px;'>Welcome {st.session_state['username']}! 👋</h1>", unsafe_allow_html=True)

    
    st.sidebar.title("Dashboard")
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

    if st.sidebar.button("Logout"):
        st.session_state['authenticated'] = False
        st.session_state['username'] = ""
        

    
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
        <div class="container">
        <h2>About the Dataset</h2>
        <p>This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on <a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset" target="_blank" style="color: #61dafb;">Kaggle</a>.</p>
        <p>It consists of approximately 87,000 RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure. A separate directory contains 33 test images for prediction purposes.</p>
        <h2>Content</h2>
        <ul>
            <li><strong>Training Set:</strong> 70,295 images</li>
            <li><strong>Validation Set:</strong> 17,572 images</li>
            <li><strong>Test Set:</strong> 33 images</li>
         </ul>
        </div>
        """, unsafe_allow_html=True)

    elif app_mode == "Disease Recognition":
        st.title("Disease Recognition")
        st.write("Upload an image of the plant leaf to get predictions.")

        test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
        
        if test_image is not None:
            file_size_kb = get_file_size(test_image) / 1024  
            if not allowed_file(test_image.name):
                st.error("Unsupported file type. Please upload a JPG or PNG image.")
            elif file_size_kb > MAX_FILE_SIZE_KB:
                st.error(f"File size exceeds {MAX_FILE_SIZE_KB} KB. Please upload a smaller image.")
            else:
                st.image(test_image, use_column_width=True)
                
                if st.button("Predict"):
                    with st.spinner("Analyzing..."):
                        try:
                            result_index, confidence = model_prediction(test_image, model)
                            if result_index is not None:
                                class_names = list(disease_info.keys())
                                prediction = class_names[result_index]
                                description = disease_info.get(prediction, {}).get('description', "Description not available.")
                                supplement = disease_info.get(prediction, {}).get('supplement', "Supplement information not available.")
                                st.success(f"**Prediction:** {prediction}")
                                st.markdown(f"**Description:** {description}")
                                st.markdown(f"**Supplements:** {supplement}")
                                st.markdown(f"**Confidence:** {confidence:.2f}")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")


if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if 'username' not in st.session_state:
    st.session_state['username'] = ""

if st.session_state['authenticated']:
    main_page()
else:
    st.sidebar.title("Login/Signup")
    app_mode = st.sidebar.selectbox("Select Page", ["Login", "Signup"])
    
    if app_mode == "Login":
        login_page()
    elif app_mode == "Signup":
        signup_page()
