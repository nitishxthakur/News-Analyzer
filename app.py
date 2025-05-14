import os
import pickle
import gdown
import streamlit as st

# Define a function to download models from Google Drive
def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

# Paths and file IDs for models
model_paths = {
    "bias_detection": ("1-02Xk7Rt6yBrzDg2aEdXfZgzD2pSN8xC", "bias_detection.pkl"),
    "biased_word_detection": ("1FF691PcXaoOhBSxyjMOdybnbPxwEFxHD", "biased_word_detection.pkl"),
    "topic_detection": ("19vx4C4ZtqSG0GW-WnyfBG43QTQfMTE8l", "topic_detection.pkl"),
    "leaning_detection": ("1V7FUl4Y6DKOyHeSquVCuI0eqMuNGOsmJ", "leaning_detection.pkl"),
}

# Download and load models
models = {}
for model_name, (file_id, file_name) in model_paths.items():
    download_model(file_id, file_name)
    with open(file_name, "rb") as f:
        models[model_name] = pickle.load(f)

# Define prediction functions
def predict_bias(text):
    model = models["bias_detection"]
    return model.predict([text])[0]

def detect_biased_words(text):
    model = models["biased_word_detection"]
    return model.predict([text])[0]  # Assumes the model returns a list of biased words

def detect_topic(text):
    model = models["topic_detection"]
    return model.predict([text])[0]

def detect_leaning(text):
    model = models["leaning_detection"]
    return model.predict([text])[0]

# Streamlit interface
st.title("Media Bias Detection Tool")

# User input
text_input = st.text_area("Enter text for analysis:")
if st.button("Analyze"):
    if text_input.strip():
        with st.spinner("Analyzing text..."):
            # Predictions
            bias_result = predict_bias(text_input)
            biased_words = detect_biased_words(text_input)
            topic_result = detect_topic(text_input)
            leaning_result = detect_leaning(text_input)

        # Display results
        st.subheader("Analysis Results")
        st.write(f"**Bias Detection:** {bias_result}")
        st.write(f"**Biased Words:** {', '.join(biased_words)}")
        st.write(f"**Topic Detection:** {topic_result}")
        st.write(f"**Leaning Detection:** {leaning_result}")
    else:
        st.error("Please enter some text for analysis!")

# Footer
st.markdown("---")
st.markdown("Developed for media bias detection.")
