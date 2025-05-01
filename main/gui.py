import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import ImageGrab, ImageOps
import os
from predict import AlphabetRecognizer

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Alphabet Drawing Prediction")
        self.drawing_counter = 1 # Drawing counter initialization
        self.canvas = tk.Canvas(root, width=400, height=400, bg="white") # GUI Setup
        self.canvas.pack(pady=20)
        self.prediction_frame = ttk.Frame(root) # Prediction Display
        self.prediction_frame.pack()
        self.pred_label = ttk.Label(self.prediction_frame, text="Draw a letter A-Z", font=("Arial", 16))
        self.pred_label.pack(pady=10)
        self.confidence_label = ttk.Label(self.prediction_frame, text="Confidence: ", font=("Arial", 12))
        self.confidence_label.pack()
        self.btn_frame = ttk.Frame(root) # Control Buttons
        self.btn_frame.pack(pady=10)
        ttk.Button(self.btn_frame, text="Recognize", command=self.predict).pack(side="left", padx=5)
        ttk.Button(self.btn_frame, text="Clear", command=self.clear).pack(side="left", padx=5)
        self.recognizer = AlphabetRecognizer() # Initialize components
        self.setup_drawing()
        self.create_save_folder()

    def create_save_folder(self):
        os.makedirs("outputs/drawings", exist_ok=True) #Create save folder if it doesn't exist
    def setup_drawing(self):
        self.last_x = None
        self.last_y = None
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=18, fill="black", capstyle=tk.ROUND, smooth=True
            )
        self.last_x = event.x
        self.last_y = event.y
    def reset(self, event):
        self.last_x = None
        self.last_y = None
    def clear(self):
        self.canvas.delete("all")
        self.pred_label.config(text="Draw a letter A-Z")
        self.confidence_label.config(text="Confidence: ")
    def predict(self):
        # Capture drawing
        x = self.root.winfo_rootx() + self.canvas.winfo_x() + 2
        y = self.root.winfo_rooty() + self.canvas.winfo_y() + 2
        x1 = x + self.canvas.winfo_width() - 4
        y1 = y + self.canvas.winfo_height() - 4
        img = ImageGrab.grab((x, y, x1, y1))
        img = img.resize((28, 28)).convert('L')
        img = ImageOps.invert(img)
        # Save and predict
        self.save_drawing(img)
        img_array = np.array(img).reshape(28, 28, 1)
        try:
            char, confidence = self.recognizer.predict(img_array)
            self.update_prediction_display(char, confidence)
            self.rename_with_prediction(char, confidence)
        except Exception as e:
            self.pred_label.config(text=f"Error: {str(e)}")
    def save_drawing(self, img): #Save drawing with incremental number
        self.current_path = f"outputs/drawings/drawing_{self.drawing_counter}.png"
        img.save(self.current_path)
    def rename_with_prediction(self, char, confidence): #Rename file with prediction results
        new_path = f"outputs/drawings/{char}_{confidence:.2f}_{self.drawing_counter}.png"
        os.rename(self.current_path, new_path)
        self.drawing_counter += 1
    def update_prediction_display(self, char, confidence):
        self.pred_label.config(text=f"Prediction: {char}")
        self.confidence_label.config(text=f"Confidence: {confidence:.1%}")

if __name__ == '__main__':
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
