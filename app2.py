#pip install -r requirements.txt

import streamlit as st
import inference
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import tempfile
import os
import cv2
import tempfile
import shutil


# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="hOzJCGXzxtVYkT4mJ7n9"
    
)

st.title('Multimodal Human Activity Recognition App')

input_type = st.radio("Choose Input Type:", ('Image', 'Video', 'Live'))

threshold = st.slider('Set Confidence Threshold:', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

uploaded_file = None
if input_type == 'Image':
    uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'], key="image_uploader")
elif input_type == 'Video':
    uploaded_file = st.file_uploader("Upload a Video", type=['mp4', 'avi'], key="video_uploader")
elif input_type == 'Live' and st.button('Open Webcam'):
    st.session_state.webcam_active = True

# Image Processing
if input_type == 'Image' and uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    if st.button('Process Image'):
        image_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tfile:
            tfile.write(uploaded_file.getvalue())
            tfile.flush()  # Ensure data is written to disk
            os.fsync(tfile.fileno())  # Force write to disk
            image_path = tfile.name
        if image_path:
            result = CLIENT.infer(image_path, model_id="human-action-recognition-2-swqmx/2")
            if 'predictions' in result and len(result['predictions']) > 0:
                class_name = result['predictions'][0].get('class', 'No class found')
                st.write(f"Class: {class_name}")
            else:
                st.write("No classifications found")
       

# Video Processing
if input_type == 'Video' and uploaded_file is not None:
    with st.spinner("Processing video..."):
        try:
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_file.getvalue())
                tfile.flush()  # Ensure data is written to disk
                video_path = tfile.name

            # Extract frames from the video
            cap = cv2.VideoCapture(video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Process frame
                    image_path = tempfile.mktemp(suffix=".jpg")
                    cv2.imwrite(image_path, frame)
                    result = CLIENT.infer(image_path, model_id="human-action-recognition-2-swqmx/2")
                    if 'predictions' in result and len(result['predictions']) > 0:
                        for prediction in result['predictions']:
                            # Draw predictions on the frame
                            text = f"{prediction.get('class', 'No class')} ({prediction.get('confidence', 0):.2f})"
                            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                    
                    # Display the frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame)
                    os.remove(image_path)
            finally:
                cap.release()
            shutil.rmtree(video_path, ignore_errors=True)
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

# Handling webcam capture
# Initialize your variables and session state as necessary
if input_type == 'Live':
    if 'webcam_active' not in st.session_state:
        st.session_state['webcam_active'] = False

    if st.session_state.get('webcam_active', False):
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)
        frame_count = 0  # Initialize a frame counter
        try:
            while st.session_state.webcam_active:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to grab frame")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
                
                # Use the frame count as a unique key for the button
                if st.button('Capture and Analyze', key=f"capture_{frame_count}"):
                    ret, buffer = cv2.imencode('.jpg', frame)
                    result = CLIENT.infer(buffer.tobytes(), model_id="human-action-recognition-2-swqmx/2")
                    if 'predictions' in result and len(result['predictions']) > 0:
                        class_name = result['predictions'][0].get('class', 'No class found')
                        st.write(f"Class: {class_name}")
                    else:
                        st.write("No classifications found")
                    break  # or consider what to do next, e.g., continue capturing
                
                frame_count += 1  # Increment frame counter after each loop iteration

        finally:
            camera.release()
            st.session_state['webcam_active'] = False  # Ensure webcam is turned off when done

# Clear all button
if st.button('Clear Everything'):
    st.session_state.clear()
