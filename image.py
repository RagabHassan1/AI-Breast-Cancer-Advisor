from keras.models import load_model
import cv2
import numpy as np

model_path = 'breast_xception.h5'

def load_models(model_path):
    model = load_model(model_path)
    return model
    
model = load_models(model_path)

def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    image = cv2.resize(img, target_size)
    image = np.expand_dims(image, axis=0)
    image = image.astype('float32') / 255.0
    return image

# Example images (replace these paths with your actual image paths)
normal_image_path = "normal.jpg"
benign_image_path = "benign.jpg"
malignant_image_path = "malignent.jpg"

# Preprocess the images
try:
    normal_image = preprocess_image(normal_image_path)
    benign_image = preprocess_image(benign_image_path)
    malignant_image = preprocess_image(malignant_image_path)
except Exception as e:
    print(f"Error loading images: {e}")
    exit()

def print_results(pred):
    # Assuming model outputs probabilities for [normal, benign, malignant]
    class_idx = np.argmax(pred)
    confidence = np.max(pred)
    
    if class_idx == 0:
        result = "Normal breast - no tumor detected"
    elif class_idx == 1:
        result = "Benign tumor detected"
    else:
        result = "Malignant tumor detected"
    
    return f"{result} (confidence: {confidence:.2%})"

def get_prediction(image):
    pred = model.predict(image)
    return pred[0]  # Returns array of probabilities for each class

# Get and print predictions for each case
print("Normal breast prediction:")
normal_pred = get_prediction(normal_image)
print(print_results(normal_pred))

print("\nBenign tumor prediction:")
benign_pred = get_prediction(benign_image)
print(print_results(benign_pred))

print("\nMalignant tumor prediction:")
malignant_pred = get_prediction(malignant_image)
print(print_results(malignant_pred))