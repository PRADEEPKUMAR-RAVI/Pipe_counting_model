# 🔧 Pipe Counting System

A simple yet powerful system for detecting and counting pipes in **images** and **videos** using YOLO, FastAPI, and Streamlit.

## Google Colab Link
- https://colab.research.google.com/drive/1ZVthjNAaHzpvu-gdrCIdZ_sS_W48rBFv#scrollTo=zoEHmYnrHeTu

## 🌟 Features
- Detect and count pipes in images  
- Process videos with object tracking (no duplicate counts)  
- Real-time detection with a clean Streamlit UI  
- Easy YOLO model loading  


## 📋 Usage Guide 
### 1. Load Your Model 
1. Open the Streamlit interface 
2. Use the sidebar to upload your trained .pt YOLO model 
3. Click "Load Model" and wait for confirmation 
### 2. Image Processing 
1. Select "Image Processing" mode 
2. Upload an image (JPG, PNG, BMP) 
3. Click "Detect Pipes" 
4. View results and download the annotated image 
### 3. Video Processing 
1. Select "Video Processing" mode 
2. Upload a video (MP4, AVI, MOV, MKV) 
3. Click "Process Video" 
4. Wait for processing (may take several minutes) 
5. View the processed video with tracking 
6. Download the result

## 🎯 Key Technical Features

### Object Tracking Algorithm 
- **IoU-based Matching**: Uses Intersection over Union for object association 
- **Track Persistence**: Objects must be tracked for 5+ frames to be counted 
- **Duplicate Prevention**: Each pipe gets a unique ID across video frames 
- **Track History**: Maintains movement history for better tracking

### Video Processing Features 
- **Frame-by-frame Analysis**: Processes each video frame individually 
- **Track Visualization**: Shows tracking paths and IDs 
- **Count Accuracy**: Prevents recounting moving objects 
- **Progress Tracking**: Real-time processing updates

### Performance Optimizations 
- **Efficient Tracking**: Lightweight IoU calculation 
- **Memory Management**: Limited track history (30 frames) 
- **Frame Processing**: Optimized OpenCV operations 
- **API Streaming**: Efficient file handling

## 📊 Expected Input/Output

### Image Input 
- **Formats**: JPG, JPEG, PNG, BMP 
- **Size**: Any resolution (will be processed by YOLO) 
- **Output**: Annotated image with bounding boxes and count 
### Video Input 
- **Formats**: MP4, AVI, MOV, MKV 
- **Size**: Any resolution and duration 
- **Output**: Processed video with tracking visualization 
### Model Requirements 
- **Format**: PyTorch (.pt) file 
- **Type**: YOLOv8 trained for pipe detection 
- **Classes**: Should detect pipes/similar cylindrical objects

## 🚀 Quick Start

```bash
# Clone project
git clone <repo_url>
cd pipe-counting-system

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt


# Start FastAPI backend
uvicorn main:app --reload --port 8000

# Start Streamlit frontend
streamlit run app.py
