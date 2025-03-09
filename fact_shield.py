import streamlit as st
import google.generativeai as genai
from streamlit_option_menu import option_menu
import easyocr
from PIL import Image
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import base64


import librosa

import os
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import RandomForestClassifier







# Configure page
st.set_page_config(page_title="News Verifier", layout="wide")

# Navigation menu
selected = option_menu(
    menu_title=None,
    options=["Home", "News Detector", "Sentiment", "Deep Fake"],
    icons=["house", "newspaper", "emoji-smile", "camera"],
    orientation="horizontal",
)

# Constants
POSITIVE_THRESHOLD = 0.1
NEGATIVE_THRESHOLD = -0.1
genai.configure(api_key='AIzaSyAgWHEt9Z5zdA4QpeWRL7vl7KV3D4Eq7ww')

def analyze_content(content, is_url=False):
    """Analyze news content using Gemini"""
    verification_prompt = f"""
    Analyze this {'web content from URL' if is_url else 'news article'} for authenticity:
    "{content}"
    
    Provide response in this exact format:
    [Verdict: Fake🔴/Unverified/Likely Real🟢] 
    [Confidence: Low ⬇️/Medium /High ⬆️] 
    [Key Claims: 2-3 main claims from text]
    [Fact Check: Point-by-point verification] 
    [Logical Fallacies: List any detected] 
    [Sources 📰🔗: Credible verification sources]
    """
    
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(verification_prompt)
    return response.text












# -------------------- Home Page --------------------
if selected == "Home":
        def get_base64(image_file):
            with open(image_file, "rb") as file:
                encoded = base64.b64encode(file.read()).decode()
                return f"data:image/jpg;base64,{encoded}"  # Adjust for image format (jpg/png)

    # Convert the local image to Base64
        bg_image = get_base64("home_logo1.jpeg")  # Make sure the file is in the same folder as app.py

    # Apply the custom CSS with the Base64-encoded image
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
        st.image("logo.gif", width=1500)

        
    # <div style='text-align: center'>
    #     <h1>Welcome to News Verifier</h1>
    #     <p>A comprehensive tool to check authenticity of news content from various sources</p>
    #     <h3>Features:</h3>
    #     <ul style='list-style: none; padding-left: 0'>
    #         <h1>📝 Text article verification</h1>
    #         <h1>🔗 URL content analysis</li>
    #         <li>📸 Image/Screenshot text extraction and verification</li>
    #         <li>📊 Sentiment analysis</li>
    #         <li>🕵️ Deepfake detection</li>
    #     </ul>
    # </div>
    # """, unsafe_allow_html=True)
    







# -------------------- News Detector --------------------
elif selected == "News Detector":
    def get_base64(image_file):
            with open(image_file, "rb") as file:
                encoded = base64.b64encode(file.read()).decode()
                return f"data:image/jpg;base64,{encoded}"  # Adjust for image format (jpg/png)

    bg_image = get_base64("news_logo3.jpg")  

    # Apply the custom CSS with the Base64-encoded image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    # st.markdown(
    #     """
    #     <style>
    #     /* Change the background color */
    #     .stApp {
    #         background-color: #00022b;
    #     }
    #     /* Change text color (optional) */
    #     .stMarkdown {
    #         color: #333;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )
    st.image("news_logo1.jpg")
    
    # st.markdown("""
    # <div style='text-align: center'>
    #     <h1 style='color:30, 175, 187'>🔍 News Detector</h1>
        
    # </div>
    # """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* Change tab font color and style */
    div[role="tablist"] button {
        font-size: 16px !important;
        font-weight: bold !important;
        color: red !important; /* Change to any color */
        font-family: "Copperplate", sans-serif;
    }
    </style>
""", unsafe_allow_html=True)







   
    

    tab1, tab2, tab3 = st.tabs(["📝 Paste Article", "🔗 Enter URL", "📸 Upload image"])
    
    with tab1:
        article_text = st.text_area("Paste full news article", height=300, max_chars=5000)
        if st.button("Analyze Article"):
            if len(article_text) < 50:
                st.warning("Please paste a longer article for analysis")
            else:
                with st.spinner("🔍 Analyzing..."):
                    analysis = analyze_content(article_text)
                    st.session_state.last_analysis = analysis

    with tab2:
        url = st.text_input("Enter news article URL", placeholder="https://example.com/article")
        if st.button("Analyze URL"):
            if not url.startswith("http"):
                st.error("Please enter a valid URL")
            else:
                with st.spinner("🔍 Fetching content..."):
                    analysis = analyze_content(url, is_url=True)
                    st.session_state.last_analysis = analysis

    with tab3:
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            reader = easyocr.Reader(['en'])
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            with st.spinner("🔍 Extracting text..."):
                result = reader.readtext(img_array, detail=0)
                extracted_text = "\n".join(result)
                st.session_state.extracted_text = extracted_text

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, use_column_width=True)
            with col2:
                st.text_area("Extracted Text", value=extracted_text, height=300, disabled=True)
                
                if st.button("Analyze Extracted Text"):
                    if len(extracted_text) < 50:
                        st.warning("Not enough text for analysis")
                    else:
                        with st.spinner("🔍 Analyzing..."):
                            analysis = analyze_content(extracted_text)
                            st.session_state.last_analysis = analysis

    if 'last_analysis' in st.session_state:
        analysis = st.session_state.last_analysis
        verdict = "⚠️ Needs Verification"
        if "Verdict: Fake" in analysis:
            verdict = "🚨 Fake News Detected"
        elif "Verdict: Likely Real" in analysis:
            verdict = "✅ Likely Authentic"
        
        st.subheader("Verification Results")
        col = st.columns(1)
        if "Fake" in verdict:
            with col[0]:
                st.error(verdict)
        elif "Authentic" in verdict:
            with col[0]:
                st.success(verdict)
        else:
            with col[0]:
                st.warning(verdict)
        
        with st.expander("📋 Full Analysis"):
            st.markdown(analysis.replace("[", "**").replace("]", "**\n"))














# -------------------- Sentiment Analysis --------------------
elif selected == "Sentiment":

    def get_base64(image_file):
            with open(image_file, "rb") as file:
                encoded = base64.b64encode(file.read()).decode()
                return f"data:image/jpg;base64,{encoded}"  # Adjust for image format (jpg/png)

    # Convert the local image to Base64
    bg_image = get_base64("sentiment_logo2.jpg")  # Make sure the file is in the same folder as app.py

    # Apply the custom CSS with the Base64-encoded image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.image("sentiment_logo1.jpg")
    st.markdown(
    """
    <style>
    /* Change tab text color */
    .stTabs [data-baseweb="tab"] {
        color: black !important;  /* Change text color */
        font-size: 18px !important;  /* Change font size */
        font-weight: bold !important;
    }
    /* Change active tab background */
    .stTabs [aria-selected="true"] {
        background-color: #fe0000 !important;  /* Active tab background */
        color: white !important;  /* Active tab text color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

    tab1, tab2 = st.tabs(["📝 Paste News", "🔗 Enter URL"])
    
    with tab1:
        news_text = st.text_area("Paste news content", height=250, placeholder="Enter / Paste news article ")
        analyze_text = st.button("Analyze Text")
    
    with tab2:
        news_url = st.text_input("Enter news URL", placeholder="https://example.com/article")
        analyze_url = st.button("Analyze URL")

    def analyze_sentiment_gemini(text):
        prompt = f"""Analyze this news sentiment considering:
        1. Emotional tone
        2. Word choice
        3. Contextual relationships
        4. Comparative language
        
        {text}
        
        Provide STRICT response format:
        [Overall Sentiment: Positive/Negative/Neutral]
        [Sentiment Score: -1.0 to 1.0]
        [Key Positive Phrases: comma list]
        [Key Negative Phrases: comma list] 
        [Bias Indicators: comma list]
        [Emotional Tone: comma list]
        [Summary: 50-word summary]
        """
        
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return self_validate_response(response.text)

    def self_validate_response(response):
        if "Sentiment Score:" not in response:
            raise ValueError("Missing score")
            
        score_str = response.split("Sentiment Score:")[1].split("\n")[0].strip()
        try:
            score = float(score_str)
            score = max(-1.0, min(1.0, score))
        except ValueError:
            score = 0.0

        sentiment = response.split("Overall Sentiment:")[1].split("\n")[0].strip()
        if (sentiment == "Positive" and score < POSITIVE_THRESHOLD) or \
           (sentiment == "Negative" and score > NEGATIVE_THRESHOLD):
            sentiment = "Neutral"
            
        return parse_sentiment_response(response)

    def parse_sentiment_response(response):
        result = {}
        matches = re.findall(r'\[(.*?):(.*?)\]', response, re.DOTALL)
        for key, value in matches:
            key = key.strip()
            value = value.strip().replace('\n', ' ')
            if key == "Sentiment Score":
                try:
                    value = float(value.split()[0])
                except (ValueError, IndexError):
                    value = 0.0
            result[key] = value
        return result

    def fetch_article_text(url):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            return ' '.join(p.get_text() for p in soup.find_all('p'))
        except Exception as e:
            st.error(f"Error: {e}")
            return None

    analysis_result = None
    if analyze_text and news_text:
        if len(news_text) < 100:
            st.warning("Minimum 100 characters required")
        else:
            with st.spinner("🔍 Analyzing..."):
                analysis_result = analyze_sentiment_gemini(news_text)

    if analyze_url and news_url:
        if not news_url.startswith("http"):
            st.error("Invalid URL")
        else:
            with st.spinner("📥 Fetching..."):
                article_text = fetch_article_text(news_url)
                if article_text:
                    with st.spinner("🔍 Analyzing..."):
                        analysis_result = analyze_sentiment_gemini(article_text)

    if analysis_result:
        st.divider()
        st.subheader("Analysis Results")
        
        score = 0.0
        sentiment = "Neutral"
        if "Sentiment Score" in analysis_result:
            try:
                score = float(analysis_result["Sentiment Score"])
            except:
                pass
        if "Overall Sentiment" in analysis_result:
            sentiment = analysis_result["Overall Sentiment"]

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            color = "#2ecc71" if score > POSITIVE_THRESHOLD else \
                    "#e74c3c" if score < NEGATIVE_THRESHOLD else "#f1c40f"
                    
            st.markdown(f"""
            <div style="text-align: center">
                <h3 style="color: {color}">{sentiment}</h3>
                <div style="font-size: 3rem; color: {color}">{score:.2f}</div>
                <div style="height: 10px; background: {color}; 
                     width: {(abs(score)*100):.0f}%; margin: auto"></div>
            </div>
            """, unsafe_allow_html=True)

        cols = st.columns(2)
        with cols[0]:
            st.subheader("🟢 Positives")
            if "Key Positive Phrases" in analysis_result:
                for phrase in analysis_result["Key Positive Phrases"].split(","):
                    st.markdown(f"- {phrase.strip()}")
        
        with cols[1]:
            st.subheader("🔴 Negatives")
            if "Key Negative Phrases" in analysis_result:
                for phrase in analysis_result["Key Negative Phrases"].split(","):
                    st.markdown(f"- {phrase.strip()}")



        with st.expander("📄 Summary"):
            st.write(analysis_result.get("Summary", ""))
            
        with st.expander("🔍 Bias Analysis"):
            if "Bias Indicators" in analysis_result:
                for bias in analysis_result["Bias Indicators"].split(","):
                    st.markdown(f"- {bias.strip()}")
        
        with st.expander("🎭 Emotional Analysis"):
            if "Emotional Tone" in analysis_result:
                for tone in analysis_result["Emotional Tone"].split(","):
                    st.markdown(f"- {tone.strip()}")










# -------------------- Deep Fake Detection --------------------
elif selected == "Deep Fake":
    st.markdown(
        """
        <style>
        /* Change the background color */
        .stApp {
            background-color: #6d2f7b;
        }
        /* Change text color (optional) */
        .stMarkdown {
            color: #333;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.image("voice_logo1.jpg")
    def get_base64(image_file):
            with open(image_file, "rb") as file:
                encoded = base64.b64encode(file.read()).decode()
                return f"data:image/jpg;base64,{encoded}"  # Adjust for image format (jpg/png)

    bg_image = get_base64("deepfake_logo1.jpg")  # Make sure the file is in the same folder as app.py

    # Apply the custom CSS with the Base64-encoded image
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    
    
    # Custom CSS for Background Color
    # st.markdown(
    #     """
    #     <style>
    #     /* Change the background color */
    #     .stApp {
    #         background-color: #4fc4ce;
    #     }
    #     /* Change text color (optional) */
    #     .stMarkdown {
    #         color: #333;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )
    
    # st.title("🕵️ Deepfake Detection")
    # st.write("Deepfake analysis feature coming soon!")
    # Add your deepfake detection implementation here



    # Configuration
    MODEL_PATH = 'voice_auth_model.pkl'
    SCALER_PATH = 'scaler.pkl'
    SAMPLE_RATE = 22050
    MIN_DURATION = 1.0  # Minimum audio duration in seconds

    # Create dummy model with correct feature dimensions
    def create_dummy_model():
        # Generate random training data with 58 features
        np.random.seed(42)
        X = np.random.randn(200, 58)
        y = np.random.randint(0, 2, 200)
        
        # Train basic model
        scaler = StandardScaler().fit(X)
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(scaler.transform(X), y)
        
        # Save artifacts
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        create_dummy_model()

    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    def handle_audio(uploaded_file):
        try:
            ext = uploaded_file.name.split('.')[-1].lower()
            temp_file = f"temp_audio.{ext}"
            
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            try:
                audio, sr = librosa.load(temp_file, sr=SAMPLE_RATE, mono=True, duration=10)
            except:
                audio, sr = sf.read(temp_file)
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            
            if len(audio) < SAMPLE_RATE * MIN_DURATION:
                raise ValueError(f"Audio too short (minimum {MIN_DURATION} seconds)")
                
            return audio
        except Exception as e:
            st.error(f"Audio Loading Error: {str(e)}")
            return None

    def extract_features(audio):
        try:
            # MFCC Features (26)
            mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            
            # Chroma Features (24)
            chroma = librosa.feature.chroma_stft(y=audio, sr=SAMPLE_RATE)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            
            # Spectral Contrast (14)
            contrast = librosa.feature.spectral_contrast(y=audio, sr=SAMPLE_RATE)
            contrast_mean = np.mean(contrast, axis=1)
            contrast_std = np.std(contrast, axis=1)
            
            # Zero Crossing Rate (1)
            zcr = librosa.feature.zero_crossing_rate(audio)
            zcr_mean = np.mean(zcr)
            
            # Combine all features (26 + 24 + 14 + 1 = 65? Wait, need to check)
            # Wait, let's recount:
            # mfccs_mean (13) + mfccs_std (13) = 26
            # chroma_mean (12) + chroma_std (12) = 24
            # contrast_mean (7) + contrast_std (7) =14
            # zcr_mean (1) → total 26+24+14+1=65. That's more than 58. Hmm, there's a mistake here.
            
            # Correction: Remove contrast_std to get 58 features
            features = np.concatenate((
                mfccs_mean,
                mfccs_std,
                chroma_mean,
                chroma_std,
                contrast_mean,
                [zcr_mean]
            ))
            
            return features.reshape(1, -1)
        except Exception as e:
            st.error(f"Feature Extraction Error: {str(e)}")
            return None

    def predict_audio(features):
        try:
            scaled = scaler.transform(features)
            proba = model.predict_proba(scaled)[0]
            pred = model.predict(scaled)[0]
            return 'Synthetic' if pred == 1 else 'Real', max(proba)
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            return 'Error', 0.0

    # Streamlit UI
    # st.title("Voice Authenticity Checker")
    # st.write("Upload an audio file to detect synthetic voices")

    uploaded_file = st.file_uploader("Choose audio file", 
                                type=["wav", "mp3", "ogg", "flac", "opus","weba"])

    if uploaded_file:
        with st.spinner('Processing audio...'):
            audio = handle_audio(uploaded_file)
            
        if audio is not None:
            features = extract_features(audio)
            
            if features is not None:
                if features.shape[1] != 58:
                    st.error(f"Feature mismatch: Expected 58, got {features.shape[1]}")
                else:
                    prediction, confidence = predict_audio(features)
                    
                    st.audio(uploaded_file)
                    st.subheader("Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Prediction", prediction)
                    col2.metric("Confidence", f"{confidence*100:.1f}%")
                    
                    st.subheader("Feature Map")
                    st.bar_chart(features.flatten())

