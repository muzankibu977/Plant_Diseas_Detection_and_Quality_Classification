# app.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

# Model definition (must match training)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PlantDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Classifier")
        self.root.geometry("800x600")
        
        # Load model
        self.model = self.load_model()
        if not self.model:
            messagebox.showerror("Error", "Failed to load model!")
            root.destroy()
            return
            
        # Transformation
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # UI Elements
        self.create_widgets()
        
    def load_model(self):
        try:
            checkpoint = torch.load('complete_model.pth', map_location='cpu')
            model = SimpleCNN(len(checkpoint['class_names']))
            model.load_state_dict(checkpoint['model_state_dict'])
            model.class_names = checkpoint['class_names']
            model.eval()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Plant Disease Classifier", 
                             font=("Arial", 20, "bold"))
        title_label.pack(pady=20)
        
        # Image display
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        
        # Prediction label
        self.prediction_label = tk.Label(self.root, text="", 
                                       font=("Arial", 14))
        self.prediction_label.pack(pady=10)
        
        # Confidence label
        self.confidence_label = tk.Label(self.root, text="", 
                                       font=("Arial", 12))
        self.confidence_label.pack(pady=5)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # Upload button
        upload_btn = tk.Button(button_frame, text="Upload Image", 
                              command=self.upload_image,
                              font=("Arial", 12),
                              width=15)
        upload_btn.pack(side=tk.LEFT, padx=10)
        
        # Class list
        class_frame = tk.LabelFrame(self.root, text="Available Classes", 
                                   font=("Arial", 12))
        class_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Create a canvas and scrollbar
        canvas = tk.Canvas(class_frame)
        scrollbar = tk.Scrollbar(class_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Display class names
        for i, class_name in enumerate(self.model.class_names):
            tk.Label(scrollable_frame, text=f"{i+1}. {class_name}", 
                   font=("Arial", 10)).pack(anchor="w", padx=5, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.predict_image(file_path)
    
    def predict_image(self, image_path):
        try:
            # Open and process image
            image = Image.open(image_path)
            input_image = self.transform(image).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                output = self.model(input_image)
                _, predicted = torch.max(output, 1)
                predicted_class = self.model.class_names[predicted.item()]
                probabilities = F.softmax(output, dim=1)[0] * 100
                confidence = probabilities[predicted.item()].item()
            
            # Display image
            image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            # Display prediction
            self.prediction_label.config(text=f"Prediction: {predicted_class}")
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    root.mainloop()