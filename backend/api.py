# FastAPI imports
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse

# Standard library imports
import asyncio
import base64
import io
import logging
import os
import sys
import av
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import aiofiles
import cv2
import numpy as np
import torch
import torch.nn as nn
import uvicorn
from pydantic import BaseModel
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Model Inference API", 
             description="API for performing inference with various deep learning models and YOLOv10",
             version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model cache
model_cache = {}



# Configure video output directory
VIDEO_OUTPUT_DIR = Path("yolo_video_output")
VIDEO_OUTPUT_DIR.mkdir(exist_ok=True)

class ModelName(str, Enum):
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V2_TRASHBOX = "mobilenet_v2_trashbox"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    RESNET152 = "resnet152"

class Prediction(BaseModel):
    label: str
    probability: float

class InferenceResponse(BaseModel):
    model: str
    predictions: List[Prediction]

class ModelInfo(BaseModel):
    name: str
    description: str
    input_size: tuple[int, int]
    input_channels: int
    framework: str

class DetectionResult(BaseModel):
    class_counts: Dict[str, int]
    confidence_scores: List[float]
    boxes: List[List[float]]
    processed_image: str  # Base64 encoded processed image

def get_class_names(model_name: str = None) -> list:
    """Get fixed class names for the trash classification task based on the model type."""
    if model_name == "mobilenet_v2_trashbox":
        # Special case for trashbox model with 7 classes
        return ["glass", "cardboard", "metal", "paper", "e-waste", "plastic", "medical"]
    # Default case with 6 classes
    return ["cardboard", "metal", "plastic", "trash", "paper", "glass"]


def get_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """Function to return the specified model."""
    base_model_name = "mobilenet_v2" if model_name == "mobilenet_v2_trashbox" else model_name
    model = getattr(models, base_model_name)(pretrained=True)

    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'fc'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Conv2d):
        model.classifier = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        raise ValueError(f"Model {model_name} is not supported or needs custom handling.")

    return model

class ModelRegistry:
    def __init__(self):
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Initialize models config
            self.models_config = {
                ModelName.MOBILENET_V3_SMALL: {
                    'path': "models/mobilenet_v3_small_data_aug_imagenet_pretrained/model-best.pth",
                    'description': 'MobileNetV3-Small model with data augmentation pretrained on ImageNet',
                    'num_classes': 6
                },
                ModelName.MOBILENET_V2: {
                    'path': "models/mobilenet_v2_data_aug_imagenet_pretrained/model-best.pth",
                    'description': 'MobileNetV2 model with data augmentation pretrained on ImageNet',
                    'num_classes': 6
                },
                ModelName.MOBILENET_V2_TRASHBOX: {
                    'path': "models/mobilenetv2_data_aug_imagenet_trashbox/model-best.pth",
                    'description': 'MobileNetV2 model with data augmentation trained on trashbox dataset',
                    'num_classes': 7  # Special case for trashbox model
                },
                ModelName.MOBILENET_V3_LARGE: {
                    'path': "models/mobilenet_v3_large_data_aug_imagenet_pretrained/model-best.pth",
                    'description': 'MobileNetV3-Large model with data augmentation pretrained on ImageNet',
                    'num_classes': 6
                },
                ModelName.RESNET152: {
                    'path': "models/resnet152_data_aug_imagenet_pretrained/model-best.pth",
                    'description': 'ResNet152 model with data augmentation pretrained on ImageNet',
                    'num_classes': 6
                }
            }
            
            # Log available model paths
            for model_name, config in self.models_config.items():
                logger.info(f"Checking model path for {model_name}: {config['path']}")
                if not config['path']:
                    logger.warning(f"Model weights not found for {model_name} at {config['path']}")
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        except Exception as e:
            logger.error(f"Error in ModelRegistry initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def get_model(self, model_name: ModelName) -> torch.nn.Module:
        try:
            if model_name not in model_cache:
                if model_name not in self.models_config:
                    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
                
                model_path = self.models_config[model_name]['path']
                num_classes = self.models_config[model_name]['num_classes']
                logger.info(f"Loading model from path: {model_path} with {num_classes} classes")
                
                if not model_path:
                    raise HTTPException(status_code=404, detail=f"Model weights not found at {model_path}")
                
                model = get_model(model_name.value, num_classes)
                logger.info(f"Created model architecture for {model_name}")
                
                # Load weights
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded weights for {model_name}")
                
                model.eval()
                model = model.to(self.device)
                model_cache[model_name] = model
            
            return model_cache[model_name]
        except Exception as e:
            logger.error(f"Error in get_model: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_class_names(self, model_name: ModelName) -> list:
        """Get class names for the specified model."""
        return get_class_names(model_name.value)

    def get_model_info(self, model_name: ModelName) -> ModelInfo:
        try:
            if model_name not in self.models_config:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
            return ModelInfo(
                name=model_name,
                description=self.models_config[model_name]['description'],
                input_size=(224, 224),
                input_channels=3,
                framework="PyTorch"
            )
        except Exception as e:
            logger.error(f"Error in get_model_info: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Initialize model registry
try:
    model_registry = ModelRegistry()
except Exception as e:
    logger.error(f"Failed to initialize ModelRegistry: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

app.mount("/static", StaticFiles(directory=VIDEO_OUTPUT_DIR), name="static")

@app.post("/api/inference", response_model=InferenceResponse)
async def perform_inference(
    model: ModelName = Form(...),
    file: UploadFile = File(...),
):
    try:
        logger.info(f"Received inference request for model: {model}")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = model_registry.transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(model_registry.device)
        
        # Get model and perform inference
        model_instance = await model_registry.get_model(model)
        class_names = model_registry.get_class_names(model)
        
        with torch.no_grad():
            outputs = model_instance(image_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze()
        
        # Get top 5 predictions
        num_classes = len(class_names)
        top_k = min(1, num_classes)
        top_prob, top_catid = torch.topk(probabilities, top_k)        
        # Format results
        predictions = [
            Prediction(
                # label=class_names[idx],
                # probability=round(float(prob),2)
                label = f"Predicted: {class_names[idx]}",
                # probability = f"Probability: {prob*100:.2f}%"
                probability = round(prob*100,2)
            )
            for idx, prob in zip(top_catid.tolist(), top_prob.tolist())
        ]
        
        response = InferenceResponse(
            model=model,
            predictions=predictions
        )
        logger.info(f"Successful inference for model {model}")
        return response
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/{model_name}", response_model=ModelInfo)
async def get_model_info(model_name: ModelName):
    try:
        return model_registry.get_model_info(model_name)
    except Exception as e:
        logger.error(f"Error in get_model_info endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@contextmanager
def temporary_file(suffix=None):
    """Context manager for handling temporary files with proper cleanup."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        yield temp_file
    finally:
        temp_file.close()
        try:
            os.unlink(temp_file.name)
        except Exception:
            pass

# Initialize YOLO model
@torch.no_grad()
def get_model_yolo():
    model = YOLO('models/best.pt')
    return model

model = get_model_yolo()

@app.post("/detect/yolo/image", response_model=DetectionResult)
async def process_image(
    file: UploadFile = File(...),
    confidence: float = 0.2
):
    """
    Process an image file and return both detection results and processed image.
    
    Args:
        file: Image file to process
        confidence: Confidence threshold for detection (0-1)
    
    Returns:
        JSON containing:
        - Detection statistics (class counts, confidence scores, bounding boxes)
        - Base64 encoded processed image with detection boxes
    """
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Perform detection
        results = model(
            source=image,
            conf=confidence,
            device="cpu"
        )[0]
        
        # Process detection results
        class_counts = {}
        confidence_scores = []
        boxes = []
        
        for box in results.boxes:
            class_name = model.names[int(box.cls)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidence_scores.append(float(box.conf))
            boxes.append(box.xyxy[0].tolist())
        
        # Get processed image with annotations
        processed_image = results.plot()
        
        # Convert processed image to base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return DetectionResult(
            class_counts=class_counts,
            confidence_scores=confidence_scores,
            boxes=boxes,
            processed_image=processed_image_base64
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Thread pool for handling video processing
thread_pool = ThreadPoolExecutor(max_workers=3)  # Adjust based on server capacity

#YOLO video processing

class VideoProcessor:
    def __init__(self, model_path: str, output_dir: Path):
        self.model = YOLO(model_path)
        self.output_dir = output_dir

    def process_video_frame(self, frame, confidence: float):
        """Process a single video frame with the YOLO model"""
        results = self.model(frame, conf=confidence)[0]
        return results.plot(), results

    async def process_video(self, file: Path, output_path: Path, confidence: float) -> Dict:
        """Process video file and save to output directory using PyAV for H.264 encoding"""
        try:
            # Create a temporary buffer to store video data
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                # Read uploaded file into temporary buffer
                content = await file.read()
                temp_file.write(content)
                temp_file.flush()

                # Open video using OpenCV
                cap = cv2.VideoCapture(temp_file.name)
                if not cap.isOpened():
                    raise ValueError("Could not open video file")

                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                # Initialize PyAV for video writing
                container = av.open(str(output_path), mode="w")
                stream = container.add_stream("h264", rate=fps)
                stream.width = width
                stream.height = height
                stream.pix_fmt = "yuv420p"
                stream.bit_rate = 5_000_000  # Set bitrate to 5 Mbps (adjust as needed)
                stream.options = {
                    "preset": "slow",       # Use slow preset for better compression quality
                    "crf": "20",            # Constant rate factor: lower means better quality
                    "tune": "film"          # Optimize for high-quality content
                }

                # Initialize counters
                total_class_counts = {}
                frame_count = 0

                # Process video frames
                loop = asyncio.get_event_loop()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process frame in thread pool to avoid blocking
                    processed_frame, results = await loop.run_in_executor(
                        thread_pool,
                        self.process_video_frame,
                        frame,
                        confidence,
                    )

                    # Update detection counts
                    for box in results.boxes:
                        class_name = self.model.names[int(box.cls)]
                        total_class_counts[class_name] = total_class_counts.get(class_name, 0) + 1

                    # Convert processed frame for PyAV
                    av_frame = av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
                    packet = stream.encode(av_frame)
                    if packet:
                        container.mux(packet)

                    frame_count += 1

                # Flush remaining packets
                container.mux(stream.encode(None))

                # Release resources
                cap.release()
                container.close()

            return {
                "detection_stats": total_class_counts,
                "frames_processed": frame_count,
                "output_path": str(output_path),
            }

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")

# Initialize video processor with YOLO model
video_processor = VideoProcessor("models/best.pt", VIDEO_OUTPUT_DIR)

@app.post("/detect/yolo/video")
async def process_video_endpoint(
    file: UploadFile = File(...),
    confidence: float = 0.2
):
    """
    Process video file and save to output directory with timestamp
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported video format")

    try:
        # Create timestamp and processed filename
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        processed_filename = f"{timestamp}_{file.filename}"
        output_path = VIDEO_OUTPUT_DIR / processed_filename

        # Process video
        result = await video_processor.process_video(
            file,
            output_path,
            confidence
        )
        return processed_filename

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/yolo_video/{filename}")
async def get_processed_video(filename: str):
    """
    Retrieve a processed video file from the output directory.
    
    Args:
        filename: Name of the processed video file
        
    Returns:
        FileResponse: Video file stream
    """
    try:
        # Construct the full path to the video file
        video_path = VIDEO_OUTPUT_DIR / filename
        
        # Check if the file exists
        if not video_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Video file {filename} not found"
            )
            
        # Return the video file as a streaming response
        return FileResponse(
            path=video_path,
            media_type="video/mp4",
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"Error retrieving video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8002, reload=True)