import os
import pickle
import gdown
import streamlit as st
import sys  # Import sys for environment check

# Print Python version for debugging
print(f"Python version: {sys.version}")

# Try setting torch.classes.__path__ at the very beginning
try:
    import torch
    if hasattr(torch.classes, '__path__'):
        torch.classes.__path__ = [os.path.join(torch.__path__[0], 'classes')]
    else:
        torch.classes.__path__ = []
    print("Successfully attempted to set torch.classes.__path__")
except ImportError:
    print("PyTorch not found, skipping torch.classes.__path__ modification.")
except Exception as e:
    print(f"An error occurred while trying to set torch.classes.__path__: {e}")

# Define a function to download models from Google Drive
def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=True)  # Set quiet=True for cleaner output

# Paths and file IDs for models
model_paths = {
    "bias_detection": ("1-02Xk7Rt6yBrzDg2aEdXfZgzD2pSN8xC", "bias_model.pkl"),
    "biased_word_detection": ("1FF691PcXaoOhBSxyjMOdybnbPxwEFxHD", "biased_word_model.pkl"),
    "topic_detection": ("19vx4C4ZtqSG0GW-WnyfBG43QTQfMTE8l", "topic_model.pkl"),
    "leaning_detection": ("1V7FUl4Y6DKOyHeSquVCuI0eqMuNGOsmJ", "leaning_model.pkl"),
}

# Download and load models
models = {}
for model_name, (file_id, file_name) in model_paths.items():
    download_model(file_id, file_name)
    try:
        with open(file_name, "rb") as f:
            models[model_name] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model '{file_name}': {e}")
        st.stop()

# Define prediction functions
def predict_bias(text):
    model = models["bias_detection"]
    return model.predict([text])[0]

def detect_biased_words(text):
    model = models["biased_word_detection"]
    # Assuming the model returns a list of biased words.
    # If it returns a single string, you might need to adjust this.
    prediction = model.predict([text])[0]
    if isinstance(prediction, str):
        return [word.strip() for word in prediction.split(',')] # Split string into a list
    elif isinstance(prediction, list):
        return prediction
    else:
        return [str(prediction)] # Handle other potential outputs

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
