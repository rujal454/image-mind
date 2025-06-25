from flask import Flask, request, jsonify, send_from_directory, url_for, render_template
import json
import os
import traceback
import base64
import io
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import faiss
from tqdm import tqdm
import warnings
import shutil
import pickle
import threading
import configparser
import platform
from werkzeug.utils import safe_join
import urllib.parse
import torch.nn as nn
import sys
import webbrowser
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Get the port from command-line arguments or default to 5002
PORT = int(sys.argv[1].split('=')[1]) if len(sys.argv) > 1 and '--port=' in sys.argv[1] else 5002

INDEXER_STATE_FILE = 'base_model.pkl'
indexer_lock = threading.Lock()

# Define the config file path
if platform.system() == "Windows":
    config_file = os.path.join(os.getenv('APPDATA'), 'imageindexer', 'config.ini')
else:  # macOS and Linux
    config_file = os.path.expanduser('~/Library/Application Support/imageindexer/config.ini')

# Ensure the directory exists
os.makedirs(os.path.dirname(config_file), exist_ok=True)

# Create a ConfigParser object
config = configparser.ConfigParser()

# Default upload folder
UPLOAD_FOLDER = 'upload'
CURRENT_FOLDER_PATH = None

def load_config():
    global UPLOAD_FOLDER, CURRENT_FOLDER_PATH
    if os.path.exists(config_file):
        config.read(config_file)
        if 'Settings' in config:
            UPLOAD_FOLDER = config['Settings'].get('UploadFolder', UPLOAD_FOLDER)
            CURRENT_FOLDER_PATH = config['Settings'].get('CurrentFolderPath', CURRENT_FOLDER_PATH)
    else:
        # If the file doesn't exist, create it with default values
        CURRENT_FOLDER_PATH = CURRENT_FOLDER_PATH or UPLOAD_FOLDER  # Use UPLOAD_FOLDER as default if CURRENT_FOLDER_PATH is None
        config['Settings'] = {'UploadFolder': UPLOAD_FOLDER, 'CurrentFolderPath': CURRENT_FOLDER_PATH}
        save_config()

    # Ensure CURRENT_FOLDER_PATH is set
    if CURRENT_FOLDER_PATH is None:
        CURRENT_FOLDER_PATH = UPLOAD_FOLDER

    # Ensure the folder exists
    if not os.path.exists(CURRENT_FOLDER_PATH):
        os.makedirs(CURRENT_FOLDER_PATH)

def save_config():
    config['Settings'] = {'UploadFolder': UPLOAD_FOLDER, 'CurrentFolderPath': CURRENT_FOLDER_PATH}
    with open(config_file, 'w') as configfile:
        config.write(configfile)

# Call load_config at the start of your application
load_config()

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

indexer = None
CURRENT_FOLDER_PATH = None

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Suppress semaphore warning
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore objects")

app = Flask(__name__)

# Custom MobileNetV3 model for jewelry classification
class JewelryMobileNetV3(nn.Module):
    """
    Custom MobileNetV3 model with attention mechanism for jewelry classification.
    This model extends the base MobileNetV3 architecture with additional layers
    for better feature extraction and classification.
    """
    def __init__(self, num_classes=1000):
        super(JewelryMobileNetV3, self).__init__()
        # Initialize base MobileNetV3 model with pre-trained weights
        self.mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # Freeze early layers to preserve pre-trained features
        for param in list(self.mobilenet.parameters())[:-4]:
            param.requires_grad = False
        
        # Add attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(960, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        # Extract features using MobileNetV3
        features = self.mobilenet.features(x)
        
        # Apply attention mechanism
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Global average pooling and classification
        x = self.global_pool(features)
        x = x.view(x.size(0), -1)
        
        # Classification layer
        x = self.fc(x)
        return x

# FastImageIndexer class for efficient image indexing and searching
class FastImageIndexer:
    """
    FastImageIndexer class handles image feature extraction, indexing, and similarity search.
    It uses FAISS for efficient similarity search and MobileNetV3 for feature extraction.
    """
    def __init__(self, folder_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model with 960 feature dimensions (from MobileNetV3's last layer)
        self.model = JewelryMobileNetV3(num_classes=960)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize feature dimension and index
        self.feature_dim = 960
        self.index = None
        self.image_paths = []
        
        if folder_path:
            self.create(folder_path)

    def extract_features(self, image_path):
        """
        Extract features from an image using the MobileNetV3 model.
        Returns normalized feature vector.
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image).squeeze().cpu().numpy()
            return features / np.linalg.norm(features)
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            raise


    def create(self, folder_path):
        """
        Create an index of images from a folder path.
        Extracts features from all images and builds a FAISS index.
        """
        self.image_paths = []
        self.index = faiss.IndexFlatIP(self.feature_dim)
    
        features_list = []
        for root, dirs, files in os.walk(folder_path):
            for filename in tqdm(files):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, filename)
                    try:
                        features = self.extract_features(image_path)
                        features_list.append(features)
                        self.image_paths.append(os.path.abspath(image_path))  # Store full path
                    except Exception as e:
                        print(f"Error processing {image_path}: {str(e)}")
        
        if features_list:
            features_array = np.array(features_list).astype('float32')
            self.index.add(features_array)
        
        return len(self.image_paths)
    
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    def save_model_state(model, filename):
        torch.save(model.state_dict(), filename, weights_only=True)

    def load_model_state(model, filename):
        model.load_state_dict(torch.load(filename, weights_only=True))
        model.eval()

    def save_state(self, filename):
        state = {
            'image_paths': self.image_paths,
            'index': faiss.serialize_index(self.index),
            'current_folder_path': CURRENT_FOLDER_PATH,
            'model_state': self.model.state_dict()
        }
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load_state(cls, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        indexer = cls()
        indexer.image_paths = state['image_paths']
        indexer.index = faiss.deserialize_index(state['index'])
        global CURRENT_FOLDER_PATH
        CURRENT_FOLDER_PATH = state['current_folder_path']
        indexer.model.load_state_dict(state['model_state'])
        indexer.model.eval()
        return indexer
    
    def insert(self, image_path):
        features = self.extract_features(image_path)
        self.index.add(np.array([features]).astype('float32'))
        self.image_paths.append(os.path.abspath(image_path))
        return len(self.image_paths)

    def rebuild(self, folder_path):
        return self.create(folder_path)

    def search(self, query_image_path, k=5, threshold=0.8):
        """
        Search for similar images using a query image.
        Returns top k similar images with their similarity scores.
        """
        query_features = self.extract_features(query_image_path)
        distances, indices = self.index.search(np.array([query_features]).astype('float32'), k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            similarity = dist  # FAISS returns the actual similarity for IndexFlatIP
            if similarity > threshold:
                results.append((self.image_paths[idx], float(similarity)))

        
        return results
    
    def train_model(self, train_loader, num_epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Initialize indexer
indexer = None
if os.path.exists(INDEXER_STATE_FILE):
    try:
        indexer = FastImageIndexer.load_state(INDEXER_STATE_FILE)
        print(f"Loaded existing indexer state from {INDEXER_STATE_FILE}")
    except Exception as e:
        print(f"Error loading indexer state: {e}")

def save_indexer_state():
    global indexer
    if indexer:
        with indexer_lock:
            indexer.save_state(INDEXER_STATE_FILE)
        app.logger.debug("Indexer state saved")

@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error
    app.logger.error(f"Unhandled exception: {str(e)}")
    app.logger.error(traceback.format_exc())
    # Return JSON instead of HTML for HTTP errors
    return jsonify(error=str(e)), 500

@app.route('/open_url')  
def open_website():
    try:
        port = current_settings.get('port_number', 5002) # Get the port from settings
        url = f"http://127.0.0.1:{port}"  # Build the URL with the correct port
        # print("x")
        webbrowser.open_new_tab(url) 
        # return jsonify({"message": f"Opened URL: {url}"})
        return
    except Exception as e:
        return jsonify({"error": f"Failed to open URL: {str(e)}"}), 500
    
@app.route('/')
def read_root():
    return render_template('index.html')

@app.route('/setup')
def setup():
    current_settings = {
        'port_number': config['Settings'].get('port_number', ''),
        'previous_port_numbers': config['Settings'].get('previous_port_numbers', '').split(',')
    }
    return render_template('settings.html', current_settings=current_settings)

# Load settings from a file
def load_settings():
    try:
        with open('settings.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'port_number': PORT, 'network_path': ''}  # Provide default values if file not found
    except json.JSONDecodeError as e:
        print(f"Error loading settings: {e}")
        return {'port_number': PORT, 'network_path': ''}  # Default on JSON error

# Save settings to a file
def save_settings(port_number, network_path):
    settings = {'port_number': port_number, 'network_path': network_path}
    with open('settings.json', 'w') as f:
        json.dump(settings, f, indent=4)

# Load current settings
current_settings = load_settings()

@app.route('/save_settings', methods=['POST'])
def save_settings_route():
    port_number = request.form.get('port_number')
    network_path = request.form.get('network_path')

    if port_number and network_path:
        try:
            port_number = int(port_number)
            save_settings(port_number, network_path)
            return jsonify({"message": "Settings saved. Please restart the application."})
        except ValueError:
            return jsonify({"error": "Invalid port number"}), 400

    return jsonify({"error": "Invalid input"}), 400

@app.route('/train', methods=['POST'])
def train():
    global indexer, CURRENT_FOLDER_PATH
    try:
        # Read the setup.json file to get network_path
        setup_file = 'setup.json'
        if os.path.exists(setup_file):
            with open(setup_file, 'r') as f:
                settings = json.load(f)
            network_path = settings.get('network_path', None)
        else:
            network_path = None
        
        # Use network_path if folder_path is not provided
        folder_path = request.json.get('folder_path', network_path)
        
        if folder_path is None:
            return jsonify({"error": "No folder path provided and network_path is not set"}), 400
        
        if not os.path.exists(folder_path):
            return jsonify({"error": "Folder path does not exist"}), 400
        
        CURRENT_FOLDER_PATH = folder_path
        
        indexer = FastImageIndexer(folder_path)
        num_images = len(indexer.image_paths)
        save_indexer_state()
        save_config()
        return jsonify({"message": f"Model trained on {num_images} images from the folder and subfolders"})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/rebuild', methods=['POST'])
def rebuild():
    global indexer
    try:
        new_folder_path = request.json['folder_path']
        if not os.path.exists(new_folder_path):
            return jsonify({"error": "New folder path does not exist"}), 400
        
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400
        
        # Rebuild the index using the new folder path
        num_new_images = indexer.rebuild(new_folder_path)
        save_indexer_state()
        return jsonify({"message": f"{num_new_images} images added to the index from the folder and subfolders"})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/insert', methods=['POST'])
def insert():
    global indexer, CURRENT_FOLDER_PATH
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400
        
        if 'image' in request.files:
            # Single image insertion
            image = request.files['image']
            if image.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            # Save the image to the current folder path
            new_path = os.path.join(CURRENT_FOLDER_PATH, image.filename)
            image.save(new_path)
            
            num_images = indexer.insert(new_path)
            save_indexer_state()
            return jsonify({"message": f"Image inserted. Total images: {num_images}"})
        
        elif 'folder_path' in request.form:
            # Folder insertion
            folder_path = request.form['folder_path']
            if not os.path.exists(folder_path):
                return jsonify({"error": "Folder path does not exist"}), 400
            
            inserted_count = 0
            for root, dirs, files in os.walk(folder_path):
                for filename in files:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, filename)
                        indexer.insert(image_path)
                        inserted_count += 1
            
            total_images = len(indexer.image_paths)
            save_indexer_state()
            return jsonify({"message": f"{inserted_count} images inserted from folder and subfolders. Total images: {total_images}"})
        
        else:
            return jsonify({"error": "No image file or folder path provided"}), 400
    
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500
    
@app.route('/insert_folder', methods=['POST'])
def insert_folder():
    global indexer, CURRENT_FOLDER_PATH
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400
        
        folder_path = request.json.get('folder_path')
        if not folder_path:
            return jsonify({"error": "No folder path provided"}), 400
        
        if not os.path.exists(folder_path):
            return jsonify({"error": "Folder path does not exist"}), 400
        
        inserted_count = 0
        skipped_count = 0
        errors = []

        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, filename)
                    try:
                        indexer.insert(image_path)
                        inserted_count += 1
                    except Exception as e:
                        errors.append(f"Error inserting {filename}: {str(e)}")
                else:
                    skipped_count += 1
        
        total_images = len(indexer.image_paths)
        save_indexer_state()
        
        result = {
            "message": f"{inserted_count} images inserted, {skipped_count} files skipped. Total images in index: {total_images}",
            "inserted_count": inserted_count,
            "skipped_count": skipped_count,
            "total_images": total_images
        }
        
        if errors:
            result["errors"] = errors

        return jsonify(result)
    
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
@app.route('/delete', methods=['POST'])
def delete_images():
    global indexer
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400

        filenames = request.json['filenames']
        errors = []
        successes = []

        for filename in filenames:
            full_path = next((path for path in indexer.image_paths if path.endswith(filename)), None)
            
            if full_path is None:
                errors.append(f"File {filename} not found in the index")
                continue
            
            if not os.path.exists(full_path):
                errors.append(f"File {filename} not found on disk")
                continue
            
            # Remove the file from the folder
            os.remove(full_path)
            
            # Remove the file from the indexer
            index = indexer.image_paths.index(full_path)
            indexer.image_paths.pop(index)
            indexer.index.remove_ids(np.array([index]))
            
            successes.append(f"File {filename} successfully deleted")
        
        save_indexer_state()

        return jsonify({"messages": successes, "errors": errors})
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    global indexer
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400
        
        # Check if the request has the correct content type
        if request.content_type.startswith('application/json'):
            data = request.json
        elif request.content_type.startswith('multipart/form-data'):
            data = request.form
        else:
            return jsonify({"error": f"Unsupported content type: {request.content_type}"}), 400

        # Handle image data
        if 'image' in request.files:
            image = request.files['image']
            if image.filename == '':
                return jsonify({"error": "No selected file"}), 400
            image_data = image.read()
        elif 'image' in data:
            # Handle base64 encoded image
            try:
                image_data = base64.b64decode(data['image'].split(',')[1])
            except:
                return jsonify({"error": "Invalid base64 image data"}), 400
        else:
            return jsonify({"error": "No image file or data provided"}), 400
        
        # Save the image temporarily
        temp_image_path = os.path.join(UPLOAD_FOLDER, 'temp_search_image.jpg')
        with open(temp_image_path, 'wb') as f:
            f.write(image_data)
        
        # Get Number_Of_Images_Req and Similarity_Percentage values
        Number_Of_Images_Req = int(data.get('Number_Of_Images_Req', 5))
        Similarity_Percentage_str = data.get('Similarity_Percentage')
        if Similarity_Percentage_str and str(Similarity_Percentage_str).strip():
            Similarity_Percentage = float(Similarity_Percentage_str) / 100
        else:
            Similarity_Percentage = 0.8  # Default similarity percentage if 'Similarity_Percentage' is missing or empty

        results = indexer.search(temp_image_path, k=Number_Of_Images_Req, threshold=Similarity_Percentage)
        print("Search results:", results)  # Debugging line

        # Remove the temporary image
        os.remove(temp_image_path)

        # Convert file paths to URLs
        results_with_urls = [
            {
                "image_name": os.path.basename(filename),
                "url": url_for('serve_image', filename=os.path.basename(filename), _external=True),
                "similarity": float(sim)
            } for filename, sim in results
        ]

        return jsonify({
            "results": results_with_urls,
            "Status_Code": 200,
            "Response": "Success!"
        })
        
    except Exception as e:
        app.logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<path:filename>')
def serve_image(filename):
    try:
        app.logger.debug(f"Attempting to serve image: {filename}")
        
        # Find the full path in indexer.image_paths
        full_path = next((path for path in indexer.image_paths if os.path.basename(path) == filename), None)
        
        if full_path is None or not os.path.exists(full_path):
            app.logger.error(f"File not found: {filename}")
            return jsonify({"error": f"Image {filename} not found"}), 404


        app.logger.debug(f"Full path to image: {full_path}")
        app.logger.debug(f"File exists: {os.path.exists(full_path)}")

        if not os.path.exists(full_path):
            app.logger.error(f"File not found: {full_path}")
            return jsonify({"error": f"Image {filename} not found"}), 404

        # Get the directory and filename separately
        directory = os.path.dirname(full_path)
        basename = os.path.basename(full_path)

        return send_from_directory(directory, basename)
    except Exception as e:
        app.logger.error(f"Error serving image {filename}: {str(e)}")
        return jsonify({"error": f"Error serving image: {str(e)}"}), 500
    
@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/config', methods=['GET', 'POST'])
def config_route():
    global CURRENT_FOLDER_PATH, indexer
    if request.method == 'GET':
        return jsonify({"current_folder": CURRENT_FOLDER_PATH})
    elif request.method == 'POST':
        new_folder_path = request.json.get('folder_path')
        if new_folder_path:
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            CURRENT_FOLDER_PATH = new_folder_path
            save_config()
            if indexer:
                save_indexer_state()
            return jsonify({"message": "Folder path updated successfully", "new_folder_path": CURRENT_FOLDER_PATH})
        else:
            return jsonify({"error": "No folder_path provided in request"}), 400

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    try:
        if indexer is None:
            return jsonify({"error": "Model not trained yet. Use /train first."}), 400

        # Get predictions and true labels
        predictions = []
        true_labels = []
        features = []
        processed_images = 0

        # First, get all unique labels from the dataset
        unique_labels = set()
        for image_path in indexer.image_paths:
            try:
                label = os.path.basename(image_path).split('_')[0]
                if label:
                    unique_labels.add(label)
            except:
                continue

        if len(unique_labels) == 0:
            return jsonify({"error": "No valid labels found in the dataset"}), 400

        label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
        # Process images and collect predictions
        for image_path in indexer.image_paths:
            try:
                label = os.path.basename(image_path).split('_')[0]
                if label not in label_to_idx:
                    continue
                
                label_idx = label_to_idx[label]
                true_labels.append(label_idx)
                
                image = Image.open(image_path).convert('RGB')
                image_tensor = indexer.transform(image).unsqueeze(0).to(indexer.device)
                
                with torch.no_grad():
                    output = indexer.model(image_tensor)
                    logits = output.cpu().numpy()
                    pred = logits.argmax()
                    predictions.append(pred)
                    features.append(logits.flatten())
                
                processed_images += 1
                
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
                continue

        if processed_images == 0:
            return jsonify({"error": "No images could be processed"}), 400

        # Convert to numpy arrays
        true_labels = np.array(true_labels)
        predictions = np.array(predictions)
        features = np.array(features)

        # Calculate accuracy
        accuracy = float(np.mean(true_labels == predictions))

        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, predictions, labels=range(len(unique_labels)))
        cm_list = cm.tolist() if hasattr(cm, 'tolist') else [[float(x) for x in row] for row in cm]

        # Calculate ROC curve
        if len(unique_labels) > 1:
            y_true_bin = np.zeros((len(true_labels), len(unique_labels)))
            y_score = np.zeros((len(predictions), len(unique_labels)))
            
            for i in range(len(true_labels)):
                y_true_bin[i, true_labels[i]] = 1
                y_score[i, predictions[i]] = 1
            
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
            roc_auc = float(auc(fpr, tpr))
            
            # Convert numpy arrays to lists safely
            fpr_list = fpr.tolist() if hasattr(fpr, 'tolist') else [float(x) for x in fpr]
            tpr_list = tpr.tolist() if hasattr(tpr, 'tolist') else [float(x) for x in tpr]
        else:
            fpr_list = [0, 1]
            tpr_list = [1, 1]
            roc_auc = 1.0

        # Calculate feature importance
        feature_importance = np.mean(np.abs(features), axis=0)
        top_k = min(10, len(feature_importance))
        top_indices = np.argsort(feature_importance)[-top_k:]
        
        # Safely convert feature importance values
        importance_values = feature_importance[top_indices]
        importance_values_list = (importance_values.tolist() 
                                if hasattr(importance_values, 'tolist') 
                                else [float(x) for x in importance_values])

        return jsonify({
            "accuracy": accuracy,
            "auc": roc_auc,
            "roc_data": {
                "fpr": fpr_list,
                "tpr": tpr_list
            },
            "confusion_matrix": cm_list,
            "feature_importance": {
                "labels": [f"Feature {i}" for i in top_indices],
                "values": importance_values_list
            },
            "actual_vs_predicted": [[float(a), float(p)] for a, p in zip(true_labels, predictions)],
            "label_mapping": label_to_idx,
            "processed_images": processed_images,
            "num_classes": len(unique_labels)
        })

    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500

if __name__ == '__main__':
    port_number = current_settings.get('port_number', 5002)
    open_website()
    # app.run(port=int(port_number))
    app.run(debug=False, port=int(port_number), host='0.0.0.0')