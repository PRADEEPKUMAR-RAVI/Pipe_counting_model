
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from typing import Dict, List, Tuple
import json
from collections import defaultdict
import io
import base64
from PIL import Image

# Try to import config, fall back to defaults if not available
try:
    from config import *
except ImportError:
    # Default configuration if config.py doesn't exist
    MODEL_PATH = "best.pt"
    AUTO_LOAD_MODEL = True
    CONFIDENCE_THRESHOLD = 0.2
    IOU_THRESHOLD = 0.3
    MAX_DISAPPEARED_FRAMES = 30
    MIN_TRACK_LENGTH = 5
    MAX_TRACK_HISTORY = 30
    ENABLE_TRACKING_VISUALIZATION = True
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    ENABLE_CORS = True
    ENABLE_DEBUG_LOGGING = True
    LOG_MODEL_LOADING = True

app = FastAPI(title="Pipe Counting API")

# Add CORS middleware
if ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Global variables
model = None
tracker_data = defaultdict(lambda: {"center_history": [], "counted": False})

def log_message(message):
    """Print message if debug logging is enabled"""
    if ENABLE_DEBUG_LOGGING:
        print(message)

def load_model(model_path: str):
    """Load YOLO model with enhanced PyTorch 2.6+ compatibility"""
    global model
    
    if not isinstance(model_path, str):
        print(f"❌ Error: model_path is not a string, got {type(model_path)}: {model_path}")
        return False
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file does not exist at {model_path}")
        return False
    
    try:
        import torch
        import ultralytics.nn.tasks as tasks
        from ultralytics import YOLO
        
        print(f"🔄 Attempting to load model from: {model_path}")
        print(f"📦 PyTorch version: {torch.__version__}")
        
        # Check file size
        file_size = os.path.getsize(model_path)
        print(f"📁 File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        
        if file_size < 1000:  # Less than 1KB is suspicious
            print("⚠️ Warning: File size is very small, might be corrupted")
            return False
        
        # Add comprehensive safe globals for YOLOv8 models
        safe_globals_list = [
            # Ultralytics specific classes
            tasks.DetectionModel,
            'ultralytics.nn.tasks.DetectionModel',
            'ultralytics.nn.modules.head.Detect',
            'ultralytics.nn.modules.conv.Conv',
            'ultralytics.nn.modules.block.C2f',
            'ultralytics.nn.modules.block.Bottleneck',
            'ultralytics.nn.modules.block.SPPF',
            'ultralytics.nn.backbone.RegNet',
            'ultralytics.nn.backbone.EfficientNet',
            
            # PyTorch standard modules
            'torch.nn.modules.container.Sequential',
            'torch.nn.modules.container.ModuleList',
            'torch.nn.modules.conv.Conv2d',
            'torch.nn.modules.batchnorm.BatchNorm2d',
            'torch.nn.modules.activation.SiLU',
            'torch.nn.modules.activation.ReLU',
            'torch.nn.modules.activation.LeakyReLU',
            'torch.nn.modules.pooling.AdaptiveAvgPool2d',
            'torch.nn.modules.pooling.MaxPool2d',
            'torch.nn.modules.linear.Linear',
            'torch.nn.modules.upsampling.Upsample',
            'torch.nn.modules.dropout.Dropout',
            'torch.nn.modules.normalization.GroupNorm',
            'torch.nn.modules.normalization.LayerNorm',
            
            # Collections and other Python modules
            'collections.OrderedDict',
            'builtins.dict',
            'builtins.list',
            'builtins.tuple',
        ]
        
        # Add all safe globals
        if LOG_MODEL_LOADING:
            print("🔐 Adding safe globals for PyTorch 2.6+ compatibility...")
        for global_item in safe_globals_list:
            try:
                if isinstance(global_item, str):
                    torch.serialization.add_safe_globals([global_item])
                else:
                    torch.serialization.add_safe_globals([global_item])
            except Exception as e:
                pass  # Silently continue if global already exists
        
        # Method 1: Try standard YOLO loading
        try:
            if LOG_MODEL_LOADING:
                print("🔄 Attempting standard YOLO loading...")
            model = YOLO(model_path)
            if LOG_MODEL_LOADING:
                print("✅ Model loaded successfully with standard method")
            return True
            
        except Exception as e1:
            if LOG_MODEL_LOADING:
                print(f"❌ Standard loading failed: {e1}")
            
            # Method 2: Try with explicit torch.load using weights_only=False
            try:
                if LOG_MODEL_LOADING:
                    print("🔄 Trying alternative loading with weights_only=False...")
                
                # Temporarily monkey patch the torch_safe_load function
                import ultralytics.nn.tasks as tasks_module
                original_torch_safe_load = tasks_module.torch_safe_load
                
                def patched_torch_safe_load(file, *args, **kwargs):
                    """Patched version that uses weights_only=False"""
                    try:
                        return torch.load(file, map_location='cpu', weights_only=False), file
                    except Exception as e:
                        if LOG_MODEL_LOADING:
                            print(f"Patched load failed: {e}")
                        # Fall back to original method
                        return original_torch_safe_load(file, *args, **kwargs)
                
                # Apply the patch
                tasks_module.torch_safe_load = patched_torch_safe_load
                
                # Try loading again
                model = YOLO(model_path)
                
                # Restore original function
                tasks_module.torch_safe_load = original_torch_safe_load
                
                if LOG_MODEL_LOADING:
                    print("✅ Model loaded successfully with patched method")
                return True
                
            except Exception as e2:
                if LOG_MODEL_LOADING:
                    print(f"❌ Patched loading failed: {e2}")
                    print(f"💥 All loading methods failed. Error: {e2}")
                return False
        
    except Exception as e:
        print(f"💥 Critical error in load_model: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    if AUTO_LOAD_MODEL:
        print(f"🚀 Starting up... Auto-loading model from: {MODEL_PATH}")
        success = load_model(MODEL_PATH)
        if success:
            print("✅ Model loaded successfully on startup!")
        else:
            print("❌ Failed to load model on startup. You can try loading manually via API.")

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def simple_tracker(detections, frame_number, iou_threshold=None, max_disappeared=None):
    """Simple object tracker based on IoU matching"""
    global tracker_data
    
    # Use config values if not provided
    if iou_threshold is None:
        iou_threshold = IOU_THRESHOLD
    if max_disappeared is None:
        max_disappeared = MAX_DISAPPEARED_FRAMES
    
    current_detections = []
    
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        
        # Skip low confidence detections
        if conf < CONFIDENCE_THRESHOLD:
            continue
            
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Find best matching existing track
        best_match_id = None
        best_iou = 0
        
        for track_id, track_info in tracker_data.items():
            if len(track_info["center_history"]) > 0:
                last_detection = track_info["center_history"][-1]
                last_box = last_detection["box"]
                current_box = [x1, y1, x2, y2]
                
                iou = calculate_iou(last_box, current_box)
                
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_id = track_id
        
        # Create new track or update existing one
        if best_match_id is None:
            # Create new track
            track_id = f"pipe_{len(tracker_data)}_{frame_number}"
            tracker_data[track_id] = {
                "center_history": [],
                "counted": False,
                "last_seen": frame_number
            }
        else:
            track_id = best_match_id
            tracker_data[track_id]["last_seen"] = frame_number
        
        # Update track history
        tracker_data[track_id]["center_history"].append({
            "frame": frame_number,
            "center": (center_x, center_y),
            "box": [x1, y1, x2, y2],
            "confidence": conf
        })
        
        # Keep only recent history
        if len(tracker_data[track_id]["center_history"]) > MAX_TRACK_HISTORY:
            tracker_data[track_id]["center_history"] = tracker_data[track_id]["center_history"][-MAX_TRACK_HISTORY:]
        
        current_detections.append({
            "track_id": track_id,
            "box": [x1, y1, x2, y2],
            "confidence": conf,
            "center": (center_x, center_y)
        })
    
    # Remove tracks that haven't been seen for too long
    tracks_to_remove = []
    for track_id, track_info in tracker_data.items():
        if frame_number - track_info.get("last_seen", 0) > max_disappeared:
            tracks_to_remove.append(track_id)
    
    for track_id in tracks_to_remove:
        del tracker_data[track_id]
    
    return current_detections

def count_unique_pipes():
    """Count unique pipes that have been tracked sufficiently"""
    unique_count = 0
    for track_id, track_info in tracker_data.items():
        # Count pipes that have been tracked for at least MIN_TRACK_LENGTH frames
        if len(track_info["center_history"]) >= MIN_TRACK_LENGTH:
            unique_count += 1
    return unique_count

def process_image(image_array):
    """Process single image for pipe detection"""
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded")
    
    # Run detection
    results = model(image_array)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf, cls])
    
    # Draw simple bounding boxes
    annotated_image = image_array.copy()
    
    # Filter detections by confidence
    high_conf_detections = [det for det in detections if det[4] >= CONFIDENCE_THRESHOLD]
    pipe_count = len(high_conf_detections)
    
    for detection in high_conf_detections:
        x1, y1, x2, y2, conf, cls = detection
        # Draw plain green rectangle
        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Add only count text at bottom
    cv2.putText(annotated_image, f'Total Pipes: {pipe_count}', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return annotated_image, pipe_count

def process_video_frame(frame, frame_number):
    """Process single video frame with tracking"""
    if model is None:
        return frame, 0
    
    # Run detection
    results = model(frame)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, conf, cls])
    
    # Apply tracking
    tracked_detections = simple_tracker(detections, frame_number)
    
    # Draw simple tracked detections
    annotated_frame = frame.copy()
    
    for detection in tracked_detections:
        x1, y1, x2, y2 = detection["box"]
        
        # Draw plain green bounding box
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    # Get unique pipe count
    unique_count = count_unique_pipes()
    
    # Add only count text at bottom
    cv2.putText(annotated_frame, f'Unique Pipes Detected: {unique_count}', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f'Current Frame Detections: {len(tracked_detections)}', 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return annotated_frame, unique_count

@app.post("/load_model")
async def load_model_endpoint(model_file: UploadFile = File(...)):
    """Load YOLO model from uploaded file (optional - you can also set MODEL_PATH)"""
    if not model_file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="Model file must be a .pt file")
    
    # Save uploaded model temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        content = await model_file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Verify file exists and has content
        if not os.path.exists(tmp_file_path):
            raise Exception("Temporary file was not created properly")
        
        file_size = os.path.getsize(tmp_file_path)
        if file_size == 0:
            raise Exception("Uploaded file is empty")
        
        print(f"Temporary file created: {tmp_file_path}")
        print(f"File size: {file_size} bytes")
        
        # Load model
        success = load_model(tmp_file_path)
        
        if success:
            return {"message": "Model loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing model file: {str(e)}")
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        except:
            pass

@app.post("/load_model_from_path")
async def load_model_from_path_endpoint(model_path: str):
    """Load YOLO model from specified path"""
    success = load_model(model_path)
    
    if success:
        return {"message": f"Model loaded successfully from {model_path}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load model from {model_path}")

@app.post("/reload_model")
async def reload_model_endpoint():
    """Reload model from the configured MODEL_PATH"""
    success = load_model(MODEL_PATH)
    
    if success:
        return {"message": f"Model reloaded successfully from {MODEL_PATH}"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to reload model from {MODEL_PATH}")

@app.post("/process_image")
async def process_image_endpoint(image_file: UploadFile = File(...)):
    """Process uploaded image for pipe detection"""
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please load a model first.")
    
    # Read and decode image
    content = await image_file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Process image
    try:
        annotated_image, pipe_count = process_image(image)
        
        # Encode result image
        _, buffer = cv2.imencode('.jpg', annotated_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "pipe_count": pipe_count,
            "annotated_image": image_base64,
            "message": f"Detected {pipe_count} pipes in the image"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/process_video")
async def process_video_endpoint(video_file: UploadFile = File(...)):
    """Process uploaded video for pipe counting with tracking"""
    if model is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Please load a model first.")
    
    # Clear previous tracking data
    global tracker_data
    tracker_data.clear()
    
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        content = await video_file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Open video
        cap = cv2.VideoCapture(tmp_file_path)
        
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Create output video writer with better codec compatibility
        output_path = tmp_file_path.replace('.mp4', '_processed.mp4')
        # Use H.264 codec for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # If H264 fails, try MP4V as fallback
        if not out.isOpened():
            print("H264 codec failed, trying MP4V...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # If still fails, try XVID
        if not out.isOpened():
            print("MP4V codec failed, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_path = tmp_file_path.replace('.mp4', '_processed.avi')  # Change extension for XVID
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise HTTPException(status_code=500, detail="Failed to create output video writer with any codec")
        
        frame_number = 0
        max_pipe_count = 0
        
        print("Starting video processing...")
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Finished processing {frame_number} frames")
                break
            
            # Process frame
            annotated_frame, current_count = process_video_frame(frame, frame_number)
            max_pipe_count = max(max_pipe_count, current_count)
            
            # Write frame to output video
            out.write(annotated_frame)
            frame_number += 1
            
            # Progress logging every 100 frames
            if frame_number % 100 == 0:
                print(f"Processed {frame_number}/{total_frames} frames, current max count: {max_pipe_count}")
        
        # Release resources
        cap.release()
        out.release()
        
        print("Video processing completed, reading processed video...")
        
        # Check if output file was created
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Output video file was not created")
        
        output_size = os.path.getsize(output_path)
        if output_size == 0:
            raise HTTPException(status_code=500, detail="Output video file is empty")
        
        print(f"Output video size: {output_size} bytes")
        
        # Read processed video
        with open(output_path, 'rb') as f:
            video_data = f.read()
        
        video_base64 = base64.b64encode(video_data).decode('utf-8')
        
        # Clean up temporary files
        os.unlink(tmp_file_path)
        os.unlink(output_path)
        
        print("Video processing and encoding completed successfully")
        print(f"Video codec used: {fourcc}")
        
        return {
            "total_unique_pipes": max_pipe_count,
            "total_frames_processed": frame_number,
            "processed_video": video_base64,
            "video_info": {
                "codec": str(fourcc),
                "fps": fps,
                "resolution": f"{width}x{height}",
                "total_frames": frame_number
            },
            "tracking_summary": {
                "total_tracks": len(tracker_data),
                "tracks_info": {k: len(v["center_history"]) for k, v in tracker_data.items()}
            },
            "message": f"Video processed successfully. Detected {max_pipe_count} unique pipes."
        }
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        output_path = tmp_file_path.replace('.mp4', '_processed.mp4')
        if os.path.exists(output_path):
            os.unlink(output_path)
        # Also check for .avi file in case XVID codec was used
        output_path_avi = tmp_file_path.replace('.mp4', '_processed.avi')
        if os.path.exists(output_path_avi):
            os.unlink(output_path_avi)
        print(f"Video processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/model_status")
async def model_status():
    """Check if model is loaded and show configuration"""
    return {
        "model_loaded": model is not None,
        "configured_model_path": MODEL_PATH,
        "auto_load_enabled": AUTO_LOAD_MODEL,
        "configuration": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "max_disappeared_frames": MAX_DISAPPEARED_FRAMES,
            "min_track_length": MIN_TRACK_LENGTH,
            "max_track_history": MAX_TRACK_HISTORY,
            "tracking_visualization": ENABLE_TRACKING_VISUALIZATION
        },
        "model_info": {
            "model_type": str(type(model)) if model else None,
            "model_file_exists": os.path.exists(MODEL_PATH) if MODEL_PATH else False
        }
    }

@app.get("/")
async def root():
    return {"message": "Pipe Counting API is running"}

if __name__ == "__main__":
    import uvicorn
    log_message(f"🚀 Starting Pipe Counting API on {API_HOST}:{API_PORT}")
    log_message(f"📋 Model Path: {MODEL_PATH}")
    log_message(f"⚙️ Auto Load: {AUTO_LOAD_MODEL}")
    uvicorn.run(app, host=API_HOST, port=API_PORT)