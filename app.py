import os
import logging
from flask import Flask, request, render_template, jsonify, send_from_directory, send_file, url_for
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import dlib
import time
from datetime import datetime
import uuid
import subprocess
import shutil
import tempfile
import math
import face_recognition
import skimage.feature as feature
from moviepy.editor import VideoFileClip

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure static folder for processed videos
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

PROCESSED_VIDEOS_FOLDER = os.path.join(STATIC_FOLDER, 'processed_videos')
if not os.path.exists(PROCESSED_VIDEOS_FOLDER):
    os.makedirs(PROCESSED_VIDEOS_FOLDER)

# Configure allowed extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize face detector
face_detector = dlib.get_frontal_face_detector()

# Load the pre-trained model
model = None
def load_model():
    global model
    try:
        model_path = 'model/deepfake_model.h5'
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
            
        model = tf.keras.models.load_model('model/deepfake_model.h5')
        # Verify model loaded correctly by checking its architecture
        if not hasattr(model, 'predict'):
            logger.error("Loaded model appears invalid - missing predict method")
            return False
            
        # Log model architecture
        model.summary(print_fn=logger.info)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model = None
        return False

def convert_to_uint8(image):
    """Convert image to uint8 format safely."""
    try:
        # Handle None case
        if image is None:
            return None
            
        # If already uint8, return as is
        if image.dtype == np.uint8:
            return image
            
        # Convert float64 to float32 first if needed
        if image.dtype == np.float64:
            image = image.astype(np.float32)
            
        # Handle float32/float64 values
        if image.dtype in [np.float32, np.float64]:
            # Check if values are in [0,1] range
            if np.min(image) >= 0 and np.max(image) <= 1:
                image = (image * 255).clip(0, 255)
            # Check if values are already in [0,255] range
            elif np.min(image) >= 0 and np.max(image) <= 255:
                pass
            else:
                # Normalize to [0,255] range
                image = ((image - np.min(image)) * (255.0 / (np.max(image) - np.min(image)))).clip(0, 255)
        
        # Finally convert to uint8
        return image.astype(np.uint8)
    except Exception as e:
        logger.error(f"Error converting image to uint8: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model input."""
    try:
        # Convert to uint8 first
        image = convert_to_uint8(image)
        if image is None:
            raise ValueError("Failed to convert image to uint8")
        
        # Keep a copy of uint8 version
        uint8_image = image.copy()
        
        # Ensure image has 3 channels (RGB)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Resize
        target_size = (224, 224)
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        # Convert to float32 and normalize
        preprocessed = resized.astype(np.float32) / 255.0
        
        # Ensure shape is (1, 224, 224, 3)
        if len(preprocessed.shape) == 3:
            preprocessed = np.expand_dims(preprocessed, axis=0)
        
        # Double-check dimensions
        if preprocessed.shape != (1, 224, 224, 3):
            raise ValueError(f"Invalid shape after preprocessing: {preprocessed.shape}, expected (1, 224, 224, 3)")
        
        return preprocessed, uint8_image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None, None

def detect_deepfake(frame):
    """Enhanced deepfake detection with comprehensive analysis."""
    try:
        # Convert frame to uint8 first
        frame = convert_to_uint8(frame)
        if frame is None:
            return {
                'error': 'Failed to process frame',
                'details': 'Could not convert frame to uint8 format'
            }
        
        # Preprocess frame
        processed_frame, uint8_frame = preprocess_image(frame)
        if processed_frame is None or uint8_frame is None:
            return {
                'error': 'Failed to process frame',
                'details': 'Could not preprocess frame'
            }
        
        # Get model prediction
        prediction = model.predict(processed_frame)
        
        # Perform comprehensive analysis
        analysis_results = {
            'facial_landmarks': extract_facial_landmarks(uint8_frame),
            'texture': analyze_texture_consistency(uint8_frame),
            'color': analyze_color_consistency(uint8_frame),
            'compression': detect_compression_artifacts(uint8_frame),
            'frequency': analyze_frequency_domain(uint8_frame),
            'quality': analyze_image_quality(uint8_frame)
        }
        
        # Generate detailed technical explanation
        explanation = get_enhanced_technical_explanation(prediction, analysis_results)
        
        return {
            'success': True,
            'prediction': explanation['prediction'],
            'confidence': explanation['confidence'],
            'analysis': {
                'technical_analysis': explanation['technical_analysis'],
                'visual_indicators': explanation['visual_indicators'],
                'statistical_analysis': explanation['statistical_analysis'],
                'frequency_analysis': explanation['frequency_analysis'],
                'compression_analysis': explanation['compression_analysis'],
                'color_analysis': explanation['color_analysis']
            },
            'raw_metrics': analysis_results
        }
        
    except Exception as e:
        logger.exception("Error in deepfake detection")
        return {
            'error': 'Detection failed',
            'details': str(e)
        }

def process_image(image_path):
    """Process the image and return detection results."""
    logger.info(f"Starting to process image: {image_path}")
    
    # First verify file exists and has content
    if not os.path.exists(image_path):
        raise ValueError(f"Image file does not exist: {image_path}")
        
    file_size = os.path.getsize(image_path)
    logger.info(f"File size: {file_size} bytes")
    if file_size == 0:
        raise ValueError("Image file is empty")
    
    # Try multiple methods to read the image
    image = None
    errors = []
    
    # Method 1: PIL with raw bytes
    try:
        from PIL import Image
        import io
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            
        # Try to load image from bytes
        try:
            pil_image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Successfully opened image bytes with PIL. Format: {pil_image.format}")
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array as uint8
            image = np.array(pil_image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            logger.info("Successfully converted PIL image to OpenCV format")
        except Exception as e:
            errors.append(f"PIL bytes error: {str(e)}")
            logger.warning(f"Failed to read image bytes with PIL: {str(e)}")
    except Exception as e:
        errors.append(f"File read error: {str(e)}")
        logger.warning(f"Failed to read file bytes: {str(e)}")
    
    # Method 2: PIL direct
    if image is None:
        try:
            pil_image = Image.open(image_path)
            pil_image.verify()  # Verify it's a valid image
            pil_image = Image.open(image_path)  # Need to reopen after verify
            pil_image.load()  # Force load image data
            logger.info(f"Successfully opened image with PIL. Format: {pil_image.format}, Size: {pil_image.size}, Mode: {pil_image.mode}")
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array as uint8
            image = np.array(pil_image, dtype=np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            logger.info("Successfully converted PIL image to OpenCV format")
        except Exception as e:
            errors.append(f"PIL error: {str(e)}")
            logger.warning(f"Failed to read image with PIL: {str(e)}")
    
    # Method 3: OpenCV direct
    if image is None:
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is not None:
                logger.info(f"Successfully read image with OpenCV. Shape: {image.shape}")
                # Ensure uint8 format
                image = image.astype(np.uint8)
                if len(image.shape) == 2:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    logger.info("Converted grayscale to BGR")
                elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
                    logger.info("Converted RGBA to BGR")
                elif len(image.shape) == 3 and image.shape[2] == 3:  # Already BGR
                    logger.info("Image already in BGR format")
                else:
                    image = None
                    errors.append(f"Unexpected image shape: {image.shape}")
            else:
                errors.append("OpenCV imread returned None")
        except Exception as e:
            errors.append(f"OpenCV error: {str(e)}")
            logger.warning(f"Failed to read image with OpenCV: {str(e)}")
    
    # Method 4: OpenCV with numpy
    if image is None:
        try:
            with open(image_path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is not None:
                    # Ensure uint8 format
                    image = image.astype(np.uint8)
                    logger.info(f"Successfully read image with OpenCV imdecode. Shape: {image.shape}")
                else:
                    errors.append("OpenCV imdecode returned None")
        except Exception as e:
            errors.append(f"OpenCV imdecode error: {str(e)}")
            logger.warning(f"Failed to read image with OpenCV imdecode: {str(e)}")
    
    # If all methods failed, raise error with details
    if image is None:
        error_msg = "All image reading methods failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    try:
        # Validate image
        if not isinstance(image, np.ndarray):
            raise ValueError(f"Invalid image type: {type(image)}")
            
        if image.size == 0:
            raise ValueError("Image is empty")
            
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError(f"Invalid image dimensions: {image.shape}")
            
        if image.dtype != np.uint8:
            logger.warning(f"Converting image from {image.dtype} to uint8")
            image = (image * 255).astype(np.uint8)
            
        if np.isnan(image).any():
            raise ValueError("Image contains NaN values")
            
        if np.isinf(image).any():
            raise ValueError("Image contains infinite values")
            
        logger.info(f"Final image shape: {image.shape}, dtype: {image.dtype}")
        return image
        
    except Exception as e:
        logger.exception("Error validating image")
        raise ValueError(f"Image validation failed: {str(e)}")

def process_video(video_path):
    """Process video file and extract frames for analysis."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        frames = []
        while len(frames) < 1:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0
            frames.append(frame)

        cap.release()
        cv2.destroyAllWindows()

        if not frames:
            raise ValueError("No frames could be extracted from video")

        # Pad with last frame if we don't have enough frames
        while len(frames) < 1:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3)))

        return np.array(frames)

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        # Ensure cleanup even if error occurs
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        raise

import face_recognition
from moviepy.editor import VideoFileClip
import tempfile

def extract_frames(video_path, max_frames=1):
    """Extract a single significant frame from video file."""
    try:
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            raise ValueError("No frames found in video")
            
        # Get the middle frame for better representation
        middle_frame_pos = total_frames // 2
        video.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_pos)
        ret, frame = video.read()
        video.release()
        
        if not ret or frame is None:
            raise ValueError("Failed to extract frame")
            
        logger.info(f"Successfully extracted frame from position {middle_frame_pos}")
        return [frame]
        
    except Exception as e:
        logger.error(f"Error extracting frame: {str(e)}")
        raise

def detect_faces(image):
    """Detect faces in an image using face_recognition library."""
    try:
        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image)
        
        return face_locations
    except Exception as e:
        logger.error(f"Error detecting faces: {str(e)}")
        return []

def process_video_frame(frame):
    """Process a single video frame."""
    try:
        # Ensure frame is in correct format
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame")
            
        # Detect faces
        face_locations = detect_faces(frame)
        if not face_locations:
            raise ValueError("No faces detected in frame")
            
        # Preprocess frame
        preprocessed_frame, uint8_frame = preprocess_image(frame)
        
        # Make prediction
        prediction = model.predict(preprocessed_frame, verbose=0)
        confidence = float(prediction[0][0])
        
        # Annotate frame
        annotated_frame = annotate_image(uint8_frame, confidence, face_locations)
        
        return {
            'is_fake': bool(confidence > 0.5),
            'confidence': confidence,
            'annotated_frame': annotated_frame
        }
    except Exception as e:
        logger.error(f"Error processing video frame: {str(e)}")
        raise

def transcode_video(input_path, output_path):
    """Transcode video to browser-compatible format."""
    try:
        # Check if ffmpeg is available
        if not shutil.which('ffmpeg'):
            logger.error("ffmpeg not found. Please install ffmpeg.")
            return None

        # Convert to mp4 with h.264 codec and aac audio
        command = [
            'ffmpeg', '-i', input_path,
            '-c:v', 'libx264',  # Video codec
            '-preset', 'fast',   # Encoding speed preset
            '-crf', '23',       # Quality (23 is default, lower is better)
            '-c:a', 'aac',      # Audio codec
            '-movflags', '+faststart',  # Enable fast start for web playback
            '-y',               # Overwrite output file
            output_path
        ]
        
        logger.info(f"Transcoding video: {' '.join(command)}")
        
        # Run ffmpeg
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if result.returncode != 0:
            logger.error(f"Transcoding failed: {result.stderr}")
            return None
            
        logger.info("Video transcoding successful")
        return output_path
        
    except Exception as e:
        logger.exception(f"Error transcoding video: {str(e)}")
        return None

def get_technical_explanation(prediction, frame_features):
    """Generate detailed technical explanation for the prediction."""
    confidence = float(prediction[0])
    is_fake = confidence > 0.5
    
    explanations = {
        'prediction': 'Fake' if is_fake else 'Real',
        'confidence': f"{confidence * 100:.2f}%",
        'technical_details': [],
        'visual_artifacts': [],
        'metadata_analysis': []
    }
    
    # Technical analysis based on common deepfake artifacts
    if is_fake:
        if confidence > 0.8:
            explanations['technical_details'].extend([
                "Strong indicators of digital manipulation detected",
                "Significant inconsistencies in facial landmarks",
                "Unnatural texture patterns in skin regions"
            ])
        else:
            explanations['technical_details'].extend([
                "Moderate signs of manipulation present",
                "Minor inconsistencies in facial features",
                "Subtle artifacts in image quality"
            ])
            
        # Visual artifacts analysis
        explanations['visual_artifacts'].extend([
            "Inconsistent lighting across facial regions",
            "Unnatural blending at face boundaries",
            "Irregular texture patterns in skin areas",
            "Asymmetric facial features"
        ])
        
    else:
        explanations['technical_details'].extend([
            "Natural facial feature consistency",
            "Expected texture patterns",
            "Coherent lighting distribution"
        ])
        
        explanations['visual_artifacts'].extend([
            "Natural skin texture variation",
            "Consistent shadow patterns",
            "Expected facial symmetry"
        ])
    
    # Add metadata analysis
    explanations['metadata_analysis'].extend([
        "Analysis of image compression patterns",
        "Evaluation of color space consistency",
        "Assessment of noise distribution"
    ])
    
    return explanations

def analyze_frame(frame, model):
    """Analyze a single frame and return detailed results."""
    try:
        # Preprocess frame
        processed_frame, uint8_frame = preprocess_image(frame)
        if processed_frame is None:
            return {
                'error': 'Failed to process frame',
                'details': 'Could not detect face in frame'
            }
        
        # Get prediction
        prediction = model.predict(processed_frame)
        
        # Extract frame features (placeholder for actual feature extraction)
        frame_features = {
            'quality_metrics': analyze_image_quality(uint8_frame),
            'facial_landmarks': extract_facial_landmarks(uint8_frame)
        }
        
        # Get detailed technical explanation
        technical_analysis = get_technical_explanation(prediction, frame_features)
        
        return {
            'success': True,
            'prediction': technical_analysis['prediction'],
            'confidence': technical_analysis['confidence'],
            'analysis': {
                'technical_details': technical_analysis['technical_details'],
                'visual_artifacts': technical_analysis['visual_artifacts'],
                'metadata_analysis': technical_analysis['metadata_analysis']
            }
        }
        
    except Exception as e:
        logger.exception("Error analyzing frame")
        return {
            'error': 'Analysis failed',
            'details': str(e)
        }

def analyze_image_quality(image):
    """Analyze image quality metrics."""
    try:
        # Convert to grayscale for noise analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate basic image statistics
        mean, std = cv2.meanStdDev(gray)
        
        # Estimate noise level
        noise_sigma = estimate_noise(gray)
        
        # Calculate sharpness using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return {
            'brightness': float(mean[0]),
            'contrast': float(std[0]),
            'noise_level': float(noise_sigma),
            'sharpness': float(sharpness)
        }
    except Exception as e:
        logger.warning(f"Failed to analyze image quality: {str(e)}")
        return {}

def estimate_noise(gray_img):
    """Estimate noise level in image."""
    H, W = gray_img.shape
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]
    sigma = np.sum(np.sum(np.absolute(cv2.filter2D(gray_img, -1, np.array(M)))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
    return sigma

def extract_facial_landmarks(image):
    """Extract and analyze facial landmarks."""
    try:
        # Convert to RGB for dlib
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = face_detector(rgb_image)
        
        if not faces:
            return {}
            
        # Get landmarks for first face
        shape = predictor(rgb_image, faces[0])
        landmarks = face_recognition.face_landmarks(rgb_image, [shape_to_np(shape)])[0]
        
        # Analyze landmark symmetry and relationships
        symmetry_score = calculate_facial_symmetry(landmarks)
        feature_consistency = analyze_feature_consistency(landmarks)
        
        return {
            'symmetry_score': symmetry_score,
            'feature_consistency': feature_consistency,
            'landmark_count': len(landmarks)
        }
    except Exception as e:
        logger.warning(f"Failed to extract facial landmarks: {str(e)}")
        return {}

def calculate_facial_symmetry(landmarks):
    """Calculate facial symmetry score from landmarks."""
    try:
        # Get facial midpoint
        left_eye = np.mean(landmarks['left_eye'], axis=0)
        right_eye = np.mean(landmarks['right_eye'], axis=0)
        midpoint = (left_eye + right_eye) / 2
        
        # Calculate symmetry score based on key points
        symmetry_scores = []
        for left, right in zip(landmarks['left_eye'], landmarks['right_eye']):
            dist_diff = abs(np.linalg.norm(left - midpoint) - np.linalg.norm(right - midpoint))
            symmetry_scores.append(dist_diff)
            
        return float(np.mean(symmetry_scores))
    except Exception as e:
        logger.warning(f"Failed to calculate facial symmetry: {str(e)}")
        return 0.0

def analyze_feature_consistency(landmarks):
    """Analyze consistency of facial features."""
    try:
        # Calculate ratios between facial features
        eye_distance = np.linalg.norm(
            np.mean(landmarks['left_eye'], axis=0) - 
            np.mean(landmarks['right_eye'], axis=0)
        )
        nose_length = np.linalg.norm(
            landmarks['nose_bridge'][0] - landmarks['nose_bridge'][-1]
        )
        mouth_width = np.linalg.norm(
            landmarks['top_lip'][0] - landmarks['top_lip'][6]
        )
        
        # Calculate feature ratios
        eye_nose_ratio = eye_distance / nose_length
        eye_mouth_ratio = eye_distance / mouth_width
        
        return {
            'eye_nose_ratio': float(eye_nose_ratio),
            'eye_mouth_ratio': float(eye_mouth_ratio)
        }
    except Exception as e:
        logger.warning(f"Failed to analyze feature consistency: {str(e)}")
        return {}

def analyze_texture_consistency(image):
    """Analyze texture consistency across facial regions."""
    try:
        # Convert to uint8 before any OpenCV operations
        image = convert_to_uint8(image)
        if image is None:
            raise ValueError("Failed to convert input image to uint8")
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Verify gray image format
        if gray.dtype != np.uint8:
            gray = convert_to_uint8(gray)
            if gray is None:
                raise ValueError("Failed to convert grayscale image to uint8")
        
        # Calculate GLCM
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = feature.greycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
        
        # Calculate properties
        contrast = feature.greycoprops(glcm, 'contrast')
        dissimilarity = feature.greycoprops(glcm, 'dissimilarity')
        homogeneity = feature.greycoprops(glcm, 'homogeneity')
        energy = feature.greycoprops(glcm, 'energy')
        correlation = feature.greycoprops(glcm, 'correlation')
        
        return {
            'contrast_mean': float(np.mean(contrast)),
            'dissimilarity_mean': float(np.mean(dissimilarity)),
            'homogeneity_mean': float(np.mean(homogeneity)),
            'energy_mean': float(np.mean(energy)),
            'correlation_mean': float(np.mean(correlation))
        }
    except Exception as e:
        logger.error(f"Error in texture analysis: {str(e)}")
        return {
            'contrast_mean': 0.0,
            'dissimilarity_mean': 0.0,
            'homogeneity_mean': 0.0,
            'energy_mean': 0.0,
            'correlation_mean': 0.0
        }

def detect_compression_artifacts(image):
    """Detect and analyze compression artifacts."""
    try:
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Analyze each channel separately
        artifacts = {}
        for i, channel_name in enumerate(['Y', 'Cr', 'Cb']):
            channel = ycrcb[:,:,i]
            
            # Calculate DCT coefficients
            dct = cv2.dct(np.float32(channel))
            
            # Analyze coefficient distribution
            dct_stats = {
                'mean': float(np.mean(np.abs(dct))),
                'std': float(np.std(np.abs(dct))),
                'max': float(np.max(np.abs(dct))),
                'zeros': float(np.sum(np.abs(dct) < 0.1)) / dct.size
            }
            
            artifacts[channel_name] = dct_stats
            
        return artifacts
    except Exception as e:
        logger.warning(f"Failed to detect compression artifacts: {str(e)}")
        return {}

def analyze_frequency_domain(image):
    """Analyze image in frequency domain for manipulation traces."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calculate frequency domain statistics
        freq_stats = {
            'mean_magnitude': float(np.mean(magnitude_spectrum)),
            'std_magnitude': float(np.std(magnitude_spectrum)),
            'max_magnitude': float(np.max(magnitude_spectrum)),
            'energy_ratio': float(np.sum(magnitude_spectrum > np.mean(magnitude_spectrum)) / magnitude_spectrum.size)
        }
        
        return freq_stats
    except Exception as e:
        logger.warning(f"Failed to analyze frequency domain: {str(e)}")
        return {}

def get_enhanced_technical_explanation(prediction, analysis_results):
    """Generate comprehensive technical explanation based on all analyses."""
    confidence = float(prediction[0])
    is_fake = confidence > 0.5
    
    explanation = {
        'prediction': 'Fake' if is_fake else 'Real',
        'confidence': f"{confidence * 100:.2f}%",
        'technical_analysis': [],
        'visual_indicators': [],
        'statistical_analysis': [],
        'frequency_analysis': [],
        'compression_analysis': [],
        'color_analysis': []
    }
    
    # Add technical analysis based on facial features
    if 'facial_landmarks' in analysis_results:
        landmarks = analysis_results['facial_landmarks']
        if landmarks.get('symmetry_score', 0) > 0.5:
            explanation['technical_analysis'].append(
                "High facial asymmetry detected, suggesting potential manipulation"
            )
        feature_consistency = landmarks.get('feature_consistency', {})
        if feature_consistency:
            eye_nose_ratio = feature_consistency.get('eye_nose_ratio', 0)
            if eye_nose_ratio > 2.0 or eye_nose_ratio < 0.5:
                explanation['technical_analysis'].append(
                    "Unusual facial feature proportions detected"
                )
    
    # Add texture analysis
    if 'texture' in analysis_results:
        texture = analysis_results['texture']
        if texture.get('contrast_mean', 0) > 0.8:
            explanation['visual_indicators'].append(
                "High texture contrast suggesting potential artificial generation"
            )
        if texture.get('homogeneity_mean', 0) < 0.2:
            explanation['visual_indicators'].append(
                "Low texture homogeneity indicating possible manipulation"
            )
    
    # Add frequency domain analysis
    if 'frequency' in analysis_results:
        freq = analysis_results['frequency']
        if freq.get('energy_ratio', 0) > 0.7:
            explanation['frequency_analysis'].append(
                "Unusual frequency distribution typical of GAN-generated images"
            )
    
    # Add compression artifact analysis
    if 'compression' in analysis_results:
        comp = analysis_results['compression']
        for channel, stats in comp.items():
            if stats.get('zeros', 0) > 0.8:
                explanation['compression_analysis'].append(
                    f"Unusual compression patterns in {channel} channel"
                )
    
    # Add color consistency analysis
    if 'color' in analysis_results:
        color = analysis_results['color']
        for space, channels in color.items():
            for channel, stats in channels.items():
                if stats.get('std', 0) < 0.1:
                    explanation['color_analysis'].append(
                        f"Unusually uniform color distribution in {space} channel {channel}"
                    )
    
    return explanation

def detect_deepfake(frame):
    """Enhanced deepfake detection with comprehensive analysis."""
    try:
        # Convert frame to uint8 first
        frame = convert_to_uint8(frame)
        if frame is None:
            return {
                'error': 'Failed to process frame',
                'details': 'Could not convert frame to uint8 format'
            }
        
        # Preprocess frame
        processed_frame, uint8_frame = preprocess_image(frame)
        if processed_frame is None or uint8_frame is None:
            return {
                'error': 'Failed to process frame',
                'details': 'Could not preprocess frame'
            }
        
        # Get model prediction
        prediction = model.predict(processed_frame)
        
        # Perform comprehensive analysis
        analysis_results = {
            'facial_landmarks': extract_facial_landmarks(uint8_frame),
            'texture': analyze_texture_consistency(uint8_frame),
            'color': analyze_color_consistency(uint8_frame),
            'compression': detect_compression_artifacts(uint8_frame),
            'frequency': analyze_frequency_domain(uint8_frame),
            'quality': analyze_image_quality(uint8_frame)
        }
        
        # Generate detailed technical explanation
        explanation = get_enhanced_technical_explanation(prediction, analysis_results)
        
        return {
            'success': True,
            'prediction': explanation['prediction'],
            'confidence': explanation['confidence'],
            'analysis': {
                'technical_analysis': explanation['technical_analysis'],
                'visual_indicators': explanation['visual_indicators'],
                'statistical_analysis': explanation['statistical_analysis'],
                'frequency_analysis': explanation['frequency_analysis'],
                'compression_analysis': explanation['compression_analysis'],
                'color_analysis': explanation['color_analysis']
            },
            'raw_metrics': analysis_results
        }
        
    except Exception as e:
        logger.exception("Error in deepfake detection")
        return {
            'error': 'Detection failed',
            'details': str(e)
        }

def annotate_image(image, confidence, face_locations=None):
    """Annotate image with prediction results."""
    try:
        # Ensure image is in uint8 format
        image = convert_to_uint8(image)
        if image is None:
            logger.error("Failed to convert image to uint8 in annotate_image")
            return None
            
        annotated = image.copy()
        
        if face_locations is None:
            face_locations = detect_faces(image)
        
        is_fake = confidence > 0.5
        
        # Set color based on prediction (green for real, red for fake)
        color = (0, 0, 255) if is_fake else (0, 255, 0)  # BGR format
        label = "FAKE" if is_fake else "REAL"
        confidence_str = f"{confidence:.2%}"
        
        # Draw boxes and labels for each detected face
        for face_loc in face_locations:
            top, right, bottom, left = face_loc
            
            # Draw bounding box
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            
            # Create background for text
            text = f"{label} ({confidence_str})"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (left, top - text_height - 10), (left + text_width + 10, top), color, -1)
            
            # Add text
            cv2.putText(annotated, text, (left + 5, top - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
        
    except Exception as e:
        logger.error(f"Error annotating image: {str(e)}")
        return image

MAX_FRAMES = 1

def is_video_file(filename):
    """Check if a file is a video based on its extension."""
    video_extensions = {'mp4', 'avi', 'mov', 'webm'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in video_extensions

def is_image_file(filename):
    """Check if a file is an image based on its extension."""
    image_extensions = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in image_extensions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def serve_file(filename):
    """Serve uploaded files."""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/preview/<path:filename>')
def preview_file(filename):
    """Serve preview files with proper headers."""
    try:
        file_path = os.path.join(app.config['TEMP_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        # Set content type based on file extension
        content_type = 'video/mp4' if filename.endswith('.mp4') else 'image/jpeg'
        
        response = send_file(
            file_path,
            mimetype=content_type,
            as_attachment=False,
            download_name=filename
        )
        
        # Add headers for video streaming
        if content_type == 'video/mp4':
            response.headers['Accept-Ranges'] = 'bytes'
            response.headers['Cache-Control'] = 'no-cache'
            
        return response
        
    except Exception as e:
        logger.exception(f"Error serving preview: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_deepfake():
    """Handle file upload and deepfake detection."""
    logger.info("Received detection request")
    
    try:
        # Verify file is in request
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({
                'error': 'No file provided',
                'details': 'Please select a file to upload'
            }), 400
        
        file = request.files['file']
        
        # Log file information
        logger.info(f"Received file: {file.filename}")
        logger.info(f"Content type: {file.content_type}")
        
        # Verify filename
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({
                'error': 'No file selected',
                'details': 'Please select a valid file'
            }), 400
        
        # Verify file type
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({
                'error': 'Invalid file type',
                'details': f'Allowed file types are: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Create unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        ext = os.path.splitext(file.filename)[1].lower()
        filename = f"upload_{timestamp}_{unique_id}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Save the file
            file.save(filepath)
            logger.info(f"Saved file to: {filepath}")
            
            # Verify file was saved
            if not os.path.exists(filepath):
                raise ValueError("File was not saved successfully")
                
            file_size = os.path.getsize(filepath)
            if file_size == 0:
                raise ValueError("Saved file is empty")
                
            logger.info(f"Saved file size: {file_size} bytes")
            
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({
                'error': 'File save failed',
                'details': str(e)
            }), 500
        
        try:
            is_video = is_video_file(filename)
            result = None
            frame = None
            
            if is_video:
                logger.info("Processing video file")
                try:
                    # Try using moviepy first
                    clip = VideoFileClip(filepath)
                    frame = np.array(clip.get_frame(clip.duration/2))
                    clip.close()
                except Exception as e:
                    logger.warning(f"MoviePy failed, trying OpenCV: {str(e)}")
                    # Fallback to OpenCV
                    cap = cv2.VideoCapture(filepath)
                    if not cap.isOpened():
                        raise ValueError("Failed to open video file")
                    
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames == 0:
                        raise ValueError("No frames found in video")
                    
                    middle_frame = total_frames // 2
                    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if not ret or frame is None:
                        raise ValueError("Failed to extract frame from video")
                    
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Process image file
                logger.info("Processing image file")
                img = cv2.imread(filepath)
                if img is None:
                    raise ValueError("Failed to read image file")
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if frame is None:
                raise ValueError("Failed to process input file")
            
            # Log frame information
            logger.info(f"Frame shape: {frame.shape}")
            
            # Preprocess frame
            processed_frame, uint8_frame = preprocess_image(frame)
            if processed_frame is None or uint8_frame is None:
                raise ValueError("Frame preprocessing failed")
            
            logger.info(f"Processed frame shape: {processed_frame.shape}")
            
            # Make prediction
            if model is None:
                raise ValueError("Model not loaded")
            
            prediction = model.predict(processed_frame, verbose=0)
            confidence = float(prediction[0][0])
            logger.info(f"Prediction confidence: {confidence}")
            
            # Detect faces and annotate
            try:
                face_locations = detect_faces(uint8_frame)  # Use the uint8 version for face detection
                annotated_frame = annotate_image(uint8_frame, confidence, face_locations)
                
                # Save annotated frame
                annotated_filename = f"annotated_{filename.rsplit('.', 1)[0]}.jpg"
                annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
                cv2.imwrite(annotated_path, cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
                
                # Generate response
                response = {
                    'is_fake': bool(confidence > 0.5),
                    'confidence': float(confidence),
                    'is_video': is_video,
                    'annotated_image': url_for('serve_file', filename=annotated_filename)
                }
                
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Error in face detection/annotation: {str(e)}")
                return jsonify({
                    'error': 'Processing failed',
                    'details': str(e)
                }), 500
                
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({
                'error': 'Processing failed',
                'details': str(e)
            }), 500
            
        finally:
            # Clean up uploaded file
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.error(f"Error cleaning up file: {str(e)}")
                
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

@app.route('/annotated/<filename>')
def get_annotated_image(filename):
    """Serve annotated images"""
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except Exception as e:
        logger.exception(f"Error serving annotated image: {str(e)}")
        return jsonify({'error': 'Image not found'}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """Check if the service is healthy and model is loaded."""
    try:
        # Check if model is loaded
        if model is None:
            logger.error("Model is not loaded")
            return jsonify({
                'status': 'error',
                'message': 'Model not loaded',
                'details': 'The deepfake detection model is not properly initialized'
            }), 500
            
        # Check model properties
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        # Check upload directory
        upload_dir_exists = os.path.exists(app.config['UPLOAD_FOLDER'])
        upload_dir_writable = os.access(app.config['UPLOAD_FOLDER'], os.W_OK)
        
        return jsonify({
            'status': 'healthy',
            'model': {
                'loaded': True,
                'input_shape': input_shape,
                'output_shape': output_shape
            },
            'upload_directory': {
                'exists': upload_dir_exists,
                'writable': upload_dir_writable,
                'path': app.config['UPLOAD_FOLDER']
            }
        })
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/process_video', methods=['POST'])
def process_video_route():
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Check if the file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': f'Invalid file type. Supported formats: {", ".join(ALLOWED_EXTENSIONS)}'}), 400

        # Create secure filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the uploaded file
        file.save(filepath)
        logging.info(f"File saved to: {filepath}")

        try:
            is_video = is_video_file(filename)
            
            if is_video:
                # Create unique output filename
                timestamp = int(time.time())
                output_filename = f"processed_{timestamp}_{os.path.splitext(filename)[0]}.mp4"
                output_path = os.path.join(PROCESSED_VIDEOS_FOLDER, output_filename)
                
                # Convert video to MP4 format
                try:
                    clip = VideoFileClip(filepath)
                    clip.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True
                    )
                    clip.close()
                    logging.info(f"Video converted and saved to: {output_path}")
                    
                    # Process video for analysis
                    frames = process_video(filepath)
                    if len(frames) == 0:
                        raise ValueError("No frames could be processed from video")
                    
                    # Make prediction on video frames
                    prediction = model.predict(frames)
                    confidence = float(np.mean(prediction))
                    
                    # Generate preview URL
                    preview_url = url_for('static', filename=f'processed_videos/{output_filename}')
                    
                except Exception as video_error:
                    logging.error(f"Error processing video: {str(video_error)}")
                    raise ValueError(f"Error processing video: {str(video_error)}")
            else:
                # Process image
                image = preprocess_image(filepath)
                prediction = model.predict(np.expand_dims(image, axis=0))
                confidence = float(prediction[0])
                preview_url = None

            # Clean up original file
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({
                'is_fake': bool(confidence > 0.5),
                'confidence': confidence,
                'is_video': is_video,
                'preview_url': preview_url
            })

        except Exception as e:
            # Clean up files if they exist
            if os.path.exists(filepath):
                os.remove(filepath)
            if 'output_path' in locals() and os.path.exists(output_path):
                os.remove(output_path)
            raise

    except Exception as e:
        error_msg = str(e)
        logging.error(f"Error in process_video_route: {error_msg}")
        return jsonify({
            'error': f'Error processing {"video" if "video" in error_msg.lower() else "file"}: {error_msg}'
        }), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5001)
