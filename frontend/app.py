import streamlit as st
import requests
import base64
import io
from PIL import Image
import cv2
import numpy as np
import tempfile
import os

# Configure Streamlit page
st.set_page_config(
    page_title="SS-Suite:Pipe Counting Model",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = "http://localhost:8000"

def check_api_connection():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        return response.status_code == 200
    except:
        return False

def load_model(model_file):
    """Upload and load YOLO model"""
    files = {"model_file": model_file}
    try:
        response = requests.post(f"{API_BASE_URL}/load_model", files=files)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def get_model_status():
    """Check model loading status"""
    try:
        response = requests.get(f"{API_BASE_URL}/model_status")
        if response.status_code == 200:
            return response.json()["model_loaded"]
    except:
        pass
    return False

def process_image_api(image_file):
    """Send image to API for processing"""
    files = {"image_file": image_file}
    try:
        response = requests.post(f"{API_BASE_URL}/process_image", files=files)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def process_video_api(video_file):
    """Send video to API for processing"""
    files = {"video_file": video_file}
    try:
        response = requests.post(f"{API_BASE_URL}/process_video", files=files)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def main():
    st.title("SS-Suite:Pipe Counting Model")
    st.markdown("---")
    
    # Check API connection
    if not check_api_connection():
        st.error("❌ Cannot connect to the API server. Please make sure the FastAPI server is running on localhost:8000")
        st.info("Run: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`")
        return
    
    st.markdown("---")
    
    # Sidebar for model management
    with st.sidebar:
        st.success("✅ Connected to Server")
        
        st.header("🤖 Model Management")
        
        # Check model status
        model_loaded = get_model_status()
        
        if model_loaded:
            st.success("✅ Model is loaded and ready")
        else:
            st.warning("⚠️ No model loaded")
        
        # Model upload section
        st.subheader("Upload YOLO Model")
        model_file = st.file_uploader(
            "Choose your trained .pt file",
            type=['pt'],
            help="Upload your trained YOLOv8 model file for pipe detection"
        )
        
        if model_file is not None:
            if st.button("🔄 Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    success, result = load_model(model_file)
                
                if success:
                    st.success("✅ Model loaded successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to load model: {result.get('detail', 'Unknown error')}")
        
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.markdown("""
        - **Image Processing**: Single image pipe detection
        - **Video Processing**: Video with object tracking
        - **Smart Tracking**: Prevents duplicate counting
        - **Real-time Results**: Instant processing feedback
        """)
    
    # Main content area
    if not model_loaded:
        st.info("Please load a trained model first using the sidebar")
        return
    
    # Processing mode selection
    processing_mode = st.radio(
        "Select Processing Mode:",
        ["📷 Image Processing", "🎥 Video Processing"],
        horizontal=True
    )
    
    if processing_mode == "📷 Image Processing":
        st.header("📷 Image Processing")
        st.markdown("Upload an image to detect and count pipes")
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_image is not None:
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_image)
                st.image(image, use_column_width=True)
                
                # Image info
                st.info(f"""
                **Image Details:**
                - Size: {image.size[0]} x {image.size[1]} pixels
                - Format: {image.format}
                - Mode: {image.mode}
                """)
            
            # Process button
            if st.button("🔍 Detect Pipes", type="primary", use_container_width=True):
                with st.spinner("Processing image... Please wait"):
                    # Reset file pointer
                    uploaded_image.seek(0)
                    success, result = process_image_api(uploaded_image)
                
                if success:
                    with col2:
                        st.subheader("Detection Results")
                        
                        # Decode and display annotated image
                        annotated_image_data = base64.b64decode(result["annotated_image"])
                        annotated_image = Image.open(io.BytesIO(annotated_image_data))
                        st.image(annotated_image, use_column_width=True)
                        
                        # Results summary
                        st.success(f"**Pipes Detected: {result['pipe_count']}**")
                        st.info(result["message"])
                        
                        # Download button for processed image
                        img_buffer = io.BytesIO()
                        annotated_image.save(img_buffer, format='PNG')
                        img_buffer.seek(0)
                        
                        st.download_button(
                            label="📥 Download Processed Image",
                            data=img_buffer.getvalue(),
                            file_name="pipe_detection_result.png",
                            mime="image/png"
                        )
                else:
                    st.error(f"❌ Processing failed: {result.get('detail', 'Unknown error')}")
    
    else:  # Video Processing
        st.header("🎥 Video Processing with Object Tracking")
        st.markdown("Upload a video to detect and count unique pipes with advanced tracking")
        
        # Video upload
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_video is not None:
            # Video info
            st.subheader("📹 Original Video")
            
            # Save video temporarily to get info
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                tmp_file_path = tmp_file.name
            
            # Get video properties
            cap = cv2.VideoCapture(tmp_file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Display original video
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.video(uploaded_video, format="video/mp4", start_time=0)
            
            # Video details
            st.info(f"""
            **Video Details:**
            - Duration: {duration:.2f} seconds
            - FPS: {fps:.2f}
            - Total Frames: {frame_count}
            - Resolution: {width} x {height}
            - File Size: {len(uploaded_video.getvalue()) / (1024*1024):.2f} MB
            """)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            # Processing warning
            st.warning("""
            ⚠️ **Video Processing Notice:**
            - Video processing may take several minutes depending on length and resolution
            - The system uses advanced object tracking to prevent duplicate counting
            - Each pipe is assigned a unique ID and tracked across frames
            """)
            
            # Process button
            if st.button("🎬 Process Video", type="primary", use_container_width=True):
                # Reset file pointer
                uploaded_video.seek(0)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Uploading video and initializing processing...")
                progress_bar.progress(10)
                
                try:
                    status_text.text("🤖 AI model processing video frames...")
                    progress_bar.progress(30)
                    
                    success, result = process_video_api(uploaded_video)
                    
                    if success:
                        progress_bar.progress(90)
                        status_text.text("✅ Processing complete! Preparing results...")
                        
                        # Display results
                        st.success("🎉 Video processing completed successfully!")
                        
                        # Results summary
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "🔢 Total Unique Pipes",
                                result["total_unique_pipes"],
                                help="Number of unique pipes detected using object tracking"
                            )
                        
                        with col2:
                            st.metric(
                                "🎞️ Frames Processed",
                                result["total_frames_processed"],
                                help="Total number of video frames analyzed"
                            )
                        
                        with col3:
                            st.metric(
                                "🏷️ Tracking Objects",
                                result["tracking_summary"]["total_tracks"],
                                help="Number of individual object tracks created"
                            )
                        
                        # Processed video
                        st.subheader("🎬 Processed Video with Tracking")
                        
                        # Decode processed video
                        processed_video_data = base64.b64decode(result["processed_video"])
                        
                        # Download button only
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=processed_video_data,
                            file_name="pipe_counting_result.mp4",
                            mime="video/mp4"
                        )
                        
                        # Detailed tracking info
                        # with st.expander("📊 Detailed Tracking Information"):
                        #     st.json(result["tracking_summary"])
                        #     st.markdown("""
                        #     **Tracking Explanation:**
                        #     - Each detected pipe gets a unique tracking ID
                        #     - Objects are tracked across frames using IoU (Intersection over Union)
                        #     - Tracks that persist for 5+ frames are counted as valid pipes
                        #     - This prevents counting the same pipe multiple times
                        #     """)
                        
                        progress_bar.progress(100)
                        status_text.text("🎉 All done!")
                        
                    else:
                        st.error(f"❌ Processing failed: {result.get('detail', 'Unknown error')}")
                        progress_bar.progress(0)
                        status_text.text("❌ Processing failed")
                
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    progress_bar.progress(0)
                    status_text.text("❌ Error occurred")

if __name__ == "__main__":
    main()



