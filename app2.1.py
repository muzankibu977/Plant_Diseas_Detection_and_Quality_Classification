import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import sqlite3
from datetime import datetime
import cv2
import random

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
        
        # Setup database
        self.conn = sqlite3.connect('plant_history.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS diagnoses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                image_path TEXT,
                prediction TEXT,
                confidence REAL
            )
        ''')
        self.conn.commit()
        
        # Treatment knowledge base
        self.treatment_db = {
            "Tomato Early Blight": {
                "organic": "Apply copper-based fungicide every 7-10 days.",
                "chemical": "Use chlorothalonil (Bravo) at first sign."
            },
            "Healthy": "No treatment needed! 🎉"
        }
        
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
        
        # Treatment label
        self.treatment_label = tk.Label(self.root, text="", 
                                      font=("Arial", 12), justify=tk.LEFT)
        self.treatment_label.pack(pady=5)
        
        # Buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=20)
        
        # Upload button
        upload_btn = tk.Button(button_frame, text="Upload Image", 
                              command=self.upload_image,
                              font=("Arial", 12),
                              width=15)
        upload_btn.pack(side=tk.LEFT, padx=10)
        
        # Camera button
        camera_btn = tk.Button(button_frame, text="Live Camera", 
                             command=self.start_camera,
                             font=("Arial", 12),
                             width=15)
        camera_btn.pack(side=tk.LEFT, padx=10)
        
        # History button
        history_btn = tk.Button(button_frame, text="View History", 
                              command=self.show_history,
                              font=("Arial", 12),
                              width=15)
        history_btn.pack(side=tk.LEFT, padx=10)
        
        # Class list
        class_frame = tk.LabelFrame(self.root, text="Available Classes", 
                                   font=("Arial", 12))
        class_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Create a canvas and scrollbar
        canvas = tk.Canvas(class_frame)
        scrollbar = ttk.Scrollbar(class_frame, orient="vertical", command=canvas.yview)
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
            emoji = "🌱" if "healthy" in class_name.lower() else "⚠️"
            tk.Label(scrollable_frame, text=f"{emoji} {i+1}. {class_name}", 
                   font=("Arial", 10)).pack(anchor="w", padx=5, pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def start_camera(self):
        self.cam_window = tk.Toplevel(self.root)
        self.cam_window.title("Live Camera")
        
        # Center window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 250
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 250
        self.cam_window.geometry(f"500x500+{x}+{y}")
        
        self.video_label = tk.Label(self.cam_window)
        self.video_label.pack()
        
        self.capture_btn = tk.Button(self.cam_window, text="Capture & Analyze", 
                                    command=self.capture_frame,
                                    state=tk.DISABLED)
        self.capture_btn.pack(pady=10)
        
        self.cap = cv2.VideoCapture(0)
        self.update_camera()
        
        self.cam_window.protocol("WM_DELETE_WINDOW", self.close_camera)
    
    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail((500, 500))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            self.capture_btn.config(state=tk.NORMAL)
        self.cam_window.after(10, self.update_camera)
    
    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.predict_cv2_frame(frame)
            self.close_camera()
    
    def close_camera(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        self.cam_window.destroy()
    
    def predict_cv2_frame(self, frame):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.predict_image(img, from_camera=True)
    
    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Diagnosis History")
        
        tree = ttk.Treeview(history_window, columns=("Date", "Prediction", "Confidence"), show="headings")
        tree.heading("Date", text="Date")
        tree.heading("Prediction", text="Prediction")
        tree.heading("Confidence", text="Confidence (%)")
        tree.column("Date", width=150)
        tree.column("Prediction", width=200)
        tree.column("Confidence", width=100)
        
        scrollbar = ttk.Scrollbar(history_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill=tk.BOTH, expand=True)
        
        self.cursor.execute("SELECT timestamp, prediction, confidence FROM diagnoses ORDER BY timestamp DESC")
        for row in self.cursor.fetchall():
            tree.insert("", tk.END, values=(row[0], row[1], f"{row[2]:.1f}%"))
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        if file_path:
            self.predict_image(file_path)
    
    def predict_image(self, image_path, from_camera=False):
        try:
            # Process image
            if not from_camera:
                image = Image.open(image_path)
            else:
                image = image_path
            
            # Display image
            display_image = image.copy()
            display_image.thumbnail((400, 400))
            photo = ImageTk.PhotoImage(display_image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            
            # Predict
            input_image = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_image)
                _, predicted = torch.max(output, 1)
                predicted_class = self.model.class_names[predicted.item()]
                probabilities = F.softmax(output, dim=1)[0] * 100
                confidence = probabilities[predicted.item()].item()
            
            # Update UI
            self.prediction_label.config(text=f"Prediction: {predicted_class}")
            
            confidence_color = "green" if confidence > 75 else "orange" if confidence > 50 else "red"
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%", fg=confidence_color)
            
            # Show treatment
            treatment = self.treatment_db.get(predicted_class, "Consult an agricultural expert.")
            if isinstance(treatment, dict):
                treatment = f"Organic: {treatment['organic']}\nChemical: {treatment['chemical']}"
            self.treatment_label.config(text=f"Treatment:\n{treatment}")
            
            # Save to history
            if not from_camera:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.cursor.execute('''
                    INSERT INTO diagnoses (timestamp, image_path, prediction, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp, image_path, predicted_class, confidence))
                self.conn.commit()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image:\n{e}")
    
    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    root.mainloop()