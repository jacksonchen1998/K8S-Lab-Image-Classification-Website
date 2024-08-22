from flask import Flask, request, render_template, send_file, jsonify
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import io
import os

app = Flask(__name__)

# Define the model class
class MaskRCNN(torch.nn.Module):
    def __init__(self):
        super(MaskRCNN, self).__init__()
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
    
    def forward(self, x):
        return self.model(x)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MaskRCNN().to(device).eval()

# Load custom weights
model.load_state_dict(torch.load('coco_predictions.pth', map_location=device), strict=False)

# Transform for input image
transform = T.Compose([T.ToTensor()])

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    # Read the image
    image = Image.open(file.stream).convert("RGB")
    
    # Save the original image
    original_image_path = 'static/original_image.png'
    image.save(original_image_path)
    
    # Apply transformation
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Draw predictions on the image
    draw = ImageDraw.Draw(image)
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    
    results = []
    for box, label, score in zip(boxes, labels, scores):
        if score > 0.5:  # You can adjust this threshold
            draw.rectangle(box.tolist(), outline='red', width=3)
            draw.text((box[0], box[1]), f'Label: {label}, Score: {score:.2f}', fill='red')
            results.append({'label': int(label), 'score': float(score)})
    
    # Save the result image
    predicted_image_path = 'static/predicted_image.png'
    image.save(predicted_image_path)
    
    # Return image URLs and results as JSON
    return jsonify({
        'original_image_url': original_image_path,
        'predicted_image_url': predicted_image_path,
        'results': results
    })

if __name__ == '__main__':
    # Create static directory if not exists
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)
