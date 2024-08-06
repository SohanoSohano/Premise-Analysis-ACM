import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import threading
import time
from PIL import Image, ImageTk
import os
from datetime import datetime, timedelta
import csv
import base64
import requests
from tqdm import tqdm
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from google.api_core import exceptions
import logging
from ultralytics import YOLO

#API Key
API_KEY = 'ADD API KEY HERE'
API_ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent'

#Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#Configure the Gemini API
genai.configure(api_key=API_KEY)

class VideoAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SecureSight - By BBSx2")
        self.root.geometry("1200x900")
        
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        self.create_widgets()
        
        self.video_path = None
        self.output_dir = None
        self.processing = False
        
        #Load models
        self.cctv_model = YOLO(os.path.join("..", "besttrain1.pt"))
        self.door_model = YOLO(os.path.join("..", "new_door_model.pt"))

    def create_styled_paragraph(self, text, style):
        return Paragraph(text, style)
        
    def create_widgets(self):
        #Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        #Apply custom styles
        self.style.configure("TButton", padding=6, relief="flat", background="#333333", foreground="#FFFFFF")
        self.style.configure("TFrame", background="#4D4D4D")
        self.style.configure("TLabel", background="#4D4D4D", foreground="#FFFFFF", font=("Helvetica", 12))
        self.style.configure("Horizontal.TProgressbar", thickness=20, troughcolor="#4D4D4D", background="#333333")
        
        #Video selection
        select_video_button = ttk.Button(main_frame, text="Select Video", command=self.select_video)
        select_video_button.pack(pady=5)
        
        #Output directory selection
        select_output_button = ttk.Button(main_frame, text="Select Output Directory", command=self.select_output_dir)
        select_output_button.pack(pady=5)
        
        #Video display
        self.video_canvas = tk.Canvas(main_frame, bg="black", width=800, height=600)
        self.video_canvas.pack(pady=10)
        
        #Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Analysis", command=self.start_analysis, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Analysis", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.analyze_frames_button = ttk.Button(control_frame, text="Analyze Frames", command=self.analyze_frames, state=tk.DISABLED)
        self.analyze_frames_button.pack(side=tk.LEFT, padx=5)
        
        # New button for CSV analysis
        self.analyze_csv_button = ttk.Button(control_frame, text="Analyze CSV", command=self.analyze_csv, state=tk.DISABLED)
        self.analyze_csv_button.pack(side=tk.LEFT, padx=5)
        
        # Save frames checkbox
        self.save_frames_var = tk.BooleanVar()
        self.save_frames_check = ttk.Checkbutton(control_frame, text="Save Frames", variable=self.save_frames_var)
        self.save_frames_check.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Results display
        self.results_text = tk.Text(main_frame, height=15, wrap=tk.WORD, font=("Courier", 10), bg="#333333", fg="#FFFFFF")
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if self.video_path:
            self.results_text.insert(tk.END, f"Selected video: {self.video_path}\n")
            self.check_start_button()
    
    def select_output_dir(self):
        self.output_dir = filedialog.askdirectory()
        if self.output_dir:
            self.results_text.insert(tk.END, f"Selected output directory: {self.output_dir}\n")
            self.check_start_button()
    
    def check_start_button(self):
        if self.video_path and self.output_dir:
            self.start_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.DISABLED)
    
    def analyze_frames(self):
        if not hasattr(self, 'frames_dir'):
            self.results_text.insert(tk.END, "No frames available for analysis. Please run video analysis with 'Save Frames' option enabled first.\n")
            return

        threading.Thread(target=self.process_images_and_detect_brands, daemon=True).start()

    def process_images_and_detect_brands(self):
        csv_path = os.path.join(self.analysis_dir, 'frame_analysis_results.csv')
        brand_csv_path = os.path.join(self.analysis_dir, 'brand_detection_results.csv')

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile, \
             open(brand_csv_path, 'w', newline='', encoding='utf-8') as brand_csvfile:
            csvwriter = csv.writer(csvfile)
            brand_csvwriter = csv.writer(brand_csvfile)
            
            csvwriter.writerow(['Serial Number', 'Frame', 'Response'])
            brand_csvwriter.writerow(['Serial Number', 'Frame', 'Detected Brands'])

            image_files = [f for f in os.listdir(self.frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for serial_number, image_file in enumerate(tqdm(image_files, desc="Analyzing frames"), 1):
                image_path = os.path.join(self.frames_dir, image_file)
                response = self.get_text_from_image(image_path)
                csvwriter.writerow([serial_number, image_file, response])

                #Detect brands
                brands = self.detect_brands(image_path)
                brand_csvwriter.writerow([serial_number, image_file, ', '.join(brands)])
                
                #Update progress
                progress = (serial_number / len(image_files)) * 100
                self.progress_var.set(progress)
                self.root.update_idletasks()

                #Add delay between API calls to prevent errors related to high call rates
                time.sleep(1)

        self.results_text.insert(tk.END, f"Frame analysis complete. Results saved to {csv_path}\n")
        self.results_text.insert(tk.END, f"Brand detection complete. Results saved to {brand_csv_path}\n")

    def detect_brands(self, image_path: str) -> list:
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': API_KEY
        }

        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        data = {
            "contents": [{
                "parts": [
                    {"text": "Analyze the image and identify any visible shop names, brand logos, or product names. List only the detected names or brands which also appear to be large in comparison to other objects in the shop, separated by commas. If no brands or shop names are visible, respond with 'No brands detected'."},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": encoded_image
                        }
                    }
                ]
            }]
        }

        max_retries = 3
        retry_delay = 5  #seconds

        for attempt in range(max_retries):
            try:
                response = requests.post(API_ENDPOINT, headers=headers, json=data)
                response.raise_for_status()  #Raises an HTTPError for bad responses

                if response.status_code == 200:
                    brands_text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                    if brands_text == 'No brands detected':
                        return []
                    return [brand.strip() for brand in brands_text.split(',')]
                else:
                    logging.error(f"Unexpected response: {response.status_code}, {response.text}")
                    return []

            except requests.exceptions.RequestException as e:
                logging.error(f"Error detecting brands (Attempt {attempt + 1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error("Max retries reached. Unable to detect brands.")
                    return []

    def start_analysis(self):
        if not self.video_path or not self.output_dir:
            return
        
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.analyze_frames_button.config(state=tk.DISABLED)
        self.analyze_csv_button.config(state=tk.DISABLED)
        
        threading.Thread(target=self.process_video, daemon=True).start()
    
    def stop_analysis(self):
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if self.save_frames_var.get():
            self.analyze_frames_button.config(state=tk.NORMAL)
        self.analyze_csv_button.config(state=tk.NORMAL)
    
    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.analysis_dir = os.path.join(self.output_dir, f"analysis_{timestamp}")
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        cctv_csv_path = os.path.join(self.analysis_dir, "cctv_output.csv")
        door_csv_path = os.path.join(self.analysis_dir, "door_output.csv")
        
        if self.save_frames_var.get():
            self.frames_dir = os.path.join(self.analysis_dir, "frames")
            os.makedirs(self.frames_dir, exist_ok=True)
        
        with open(cctv_csv_path, 'w', newline='') as cctv_csv, open(door_csv_path, 'w', newline='') as door_csv:
            cctv_writer = csv.writer(cctv_csv)
            door_writer = csv.writer(door_csv)
            
            cctv_writer.writerow(['Timestamp', 'Frame', 'CCTV Detected', 'Confidence'])
            door_writer.writerow(['Timestamp', 'Frame', 'Door Detected', 'Confidence'])
            
            while cap.isOpened() and self.processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                #Process every 15th frame
                if frame_count % 15 != 0:
                    continue
                
                seconds = (frame_count - 1) / fps
                timestamp = str(timedelta(seconds=seconds)).split('.')[0] + '.' + f"{seconds:.1f}".split('.')[1]
                
                #CCTV detection
                cctv_results = self.cctv_model(frame)
                cctv_detections = cctv_results[0].boxes.data
                is_cctv = False
                cctv_confidence = 0
                
                for det in cctv_detections:
                    confidence = det[4].item()
                    if confidence > cctv_confidence:
                        cctv_confidence = confidence
                    if confidence > 0.2:
                        is_cctv = True
                
                frame_name = f'frame_{frame_count:06d}'
                cctv_writer.writerow([timestamp, frame_name, is_cctv, f'{cctv_confidence:.4f}'])
                
                #Door detection
                door_results = self.door_model(frame, conf=0.2)
                is_door = len(door_results[0].boxes) > 0
                door_confidence = max([box.conf.item() for box in door_results[0].boxes]) if is_door else 0
                
                door_writer.writerow([timestamp, frame_name, is_door, f'{door_confidence:.4f}'])
                
                #Update results display
                self.update_results(timestamp, is_cctv, cctv_confidence, is_door, door_confidence)
                
                # Display frame
                self.display_frame(frame, is_cctv, cctv_confidence, is_door, door_confidence)
                
                #Save frame if option is selected
                if self.save_frames_var.get():
                    frame_path = os.path.join(self.frames_dir, f"{frame_name}.jpg")
                    cv2.imwrite(frame_path, frame)
                
                #Update progress
                progress = (frame_count / total_frames) * 100
                self.progress_var.set(progress)
                self.root.update_idletasks()
                
                print(f"Processed frame {frame_count}/{total_frames}")
        
        cap.release()
        self.results_text.insert(tk.END, f"Analysis completed. Results saved in {self.analysis_dir}\n")
        self.stop_analysis()
    
    def update_results(self, timestamp, is_cctv, cctv_confidence, is_door, door_confidence):
        self.results_text.insert(tk.END, f"Timestamp: {timestamp}\n")
        self.results_text.insert(tk.END, f"CCTV detected: {'Yes' if is_cctv else 'No'} (Confidence: {cctv_confidence:.2f})\n")
        self.results_text.insert(tk.END, f"Door detected: {'Yes' if is_door else 'No'} (Confidence: {door_confidence:.2f})\n")
        self.results_text.insert(tk.END, "-" * 50 + "\n")
        self.results_text.see(tk.END)
    
    def display_frame(self, frame, is_cctv, cctv_confidence, is_door, door_confidence):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        
        #Resize the frame to fit the canvas while maintaining aspect ratio
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        scale = min(canvas_width/width, canvas_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        frame = cv2.resize(frame, (new_width, new_height))
        
        cctv_text = f"CCTV: {'Yes' if is_cctv else 'No'} ({cctv_confidence:.2f})"
        door_text = f"Entry/Exit: {'Yes' if is_door else 'No'} ({door_confidence:.2f})"
        
        cv2.putText(frame, cctv_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_cctv else (0, 0, 255), 2)
        cv2.putText(frame, door_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_door else (0, 0, 255), 2)
        
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.video_canvas.create_image(canvas_width//2, canvas_height//2, image=photo, anchor=tk.CENTER)
        self.video_canvas.image = photo

    def analyze_frames(self):
        if not hasattr(self, 'frames_dir'):
            self.results_text.insert(tk.END, "No frames available for analysis. Please run video analysis with 'Save Frames' option enabled first.\n")
            return

        threading.Thread(target=self.process_images, daemon=True).start()

    def process_images(self):
        csv_path = os.path.join(self.analysis_dir, 'frame_analysis_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Serial Number', 'Frame', 'Response'])

            image_files = [f for f in os.listdir(self.frames_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for serial_number, image_file in enumerate(tqdm(image_files, desc="Analyzing frames"), 1):
                image_path = os.path.join(self.frames_dir, image_file)
                response = self.get_text_from_image(image_path)
                csvwriter.writerow([serial_number, image_file, response])
                
                # Update progress
                progress = (serial_number / len(image_files)) * 100
                self.progress_var.set(progress)
                self.root.update_idletasks()

        self.results_text.insert(tk.END, f"Frame analysis complete. Results saved to {csv_path}\n")

    #Better prompt engineering can be done here as needed
    def get_text_from_image(self, image_path: str) -> str:
        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': API_KEY
        }

        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        data = {
            "contents": [{
                "parts": [
                    {"text": "Analyze the image and focus only on CCTVs, entry and exit points, safety, and security aspects. If none of these elements are present, respond with 'No relevant security features detected.' If relevant elements are found, provide a concise description of their location, condition, and potential security implications. Discuss vulnerabilities, risk assessments, and suggest security enhancements only if directly related to the detected elements. Avoid mentioning any irrelevant details or objects unrelated to security and safety. Avoid repitations. If there appears to be any text above any door like object or if some text is above an entry or exit point, please mention that too and say that it could possibly be a shop. Display the largest text you detected. For example if you detect a text that says Levis above an exit or entry point, display that a shop identified as Levis is there."},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": encoded_image
                        }
                    }
                ]
            }]
        }

        response = requests.post(API_ENDPOINT, headers=headers, json=data)

        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
        else:
            return f"Error: {response.status_code}, {response.text}"

    def analyze_csv(self):
        if not hasattr(self, 'analysis_dir'):
            self.results_text.insert(tk.END, "No analysis directory available. Please run video analysis first.\n")
            return

        threading.Thread(target=self.process_csv_files, daemon=True).start()

    def process_csv_files(self):
        input_dir = self.analysis_dir
        output_dir = self.analysis_dir
        
        analyses = []
        csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        
        for i, filename in enumerate(csv_files):
            file_path = os.path.join(input_dir, filename)
            csv_text = self.read_csv_as_text(file_path)
            self.results_text.insert(tk.END, f"Analyzing {filename}\n")
            analysis = self.analyze_with_gemini(csv_text, filename)
            analyses.append((filename, analysis))
            self.results_text.insert(tk.END, f"Finished analyzing {filename}\n")
            
            #Update progress
            progress = ((i + 1) / len(csv_files)) * 100
            self.progress_var.set(progress)
            self.root.update_idletasks()
            
            time.sleep(1)  #Add a small delay between API calls as done previously for the same reason
        
        output_path = os.path.join(output_dir, "comprehensive_security_analysis.pdf")
        self.generate_report(analyses, output_path)
        self.results_text.insert(tk.END, f"Comprehensive report saved to {output_path}\n")
        self.analyze_csv_button.config(state=tk.NORMAL)

    def read_csv_as_text(self, file_path):
        with open(file_path, 'r') as file:
            return file.read()

    #Better prompt engineering can be done here as needed
    def analyze_with_gemini(self, text, filename):
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Analyze the following CSV data from the file '{filename}' and provide a comprehensive security analysis report. 
        Your report should include:

        1. Executive Summary
        2. Detailed Analysis
           - Description of the location/asset
           - Identification of entry/exit points
           - CCTV and surveillance systems
           - Current security measures
        3. Risk Assessment
           - Vulnerabilities identified
           - Potential threats
           - Impact analysis
        4. Recommendations
           - Immediate actions
           - Short-term improvements
           - Long-term security strategy
        5. Additional Considerations
           - Compliance with regulations
           - Staff training needs
           - Technology upgrades

        Ensure your analysis is thorough, professional, and actionable. Provide specific details and justify your recommendations.

        CSV Data:
        {text}
        """
        max_retries = 5
        retry_delay = 10  #seconds

        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                return response.text
            except exceptions.ResourceExhausted:
                if attempt < max_retries - 1:
                    self.results_text.insert(tk.END, f"API quota exceeded. Retrying in {retry_delay} seconds...\n")
                    time.sleep(retry_delay)
                else:
                    self.results_text.insert(tk.END, "Max retries reached. Unable to analyze file.\n")
                    return "Error: Unable to analyze file due to API quota limits."
            except Exception as e:
                self.results_text.insert(tk.END, f"An error occurred: {str(e)}\n")
                return f"Error: An unexpected error occurred while analyzing the file: {str(e)}"

    #Final report generation  
    def generate_report(self, analyses, output_path):
        doc = SimpleDocTemplate(output_path, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        
        #Update existing styles
        styles['Title'].fontSize = 18
        styles['Title'].alignment = TA_CENTER
        styles['Title'].spaceAfter = 12
        
        #Add new styles
        styles.add(ParagraphStyle(name='Subtitle', 
                                  parent=styles['Heading2'], 
                                  fontSize=14, 
                                  spaceBefore=6, 
                                  spaceAfter=6))
        
        styles['BodyText'].fontSize = 10
        styles['BodyText'].leading = 14

        content = []
        
        #Title Page
        content.append(self.create_styled_paragraph("Comprehensive Security Analysis Report", styles['Title']))
        content.append(Spacer(1, 12))
        content.append(self.create_styled_paragraph("Generated By SecureSight-BBSx2", styles['Subtitle']))
        content.append(PageBreak())

        #Table of Contents
        content.append(self.create_styled_paragraph("Table of Contents", styles['Title']))
        for i, (filename, _) in enumerate(analyses, start=1):
            content.append(self.create_styled_paragraph(f"{i}. Analysis of {filename}", styles['BodyText']))
        content.append(PageBreak())

        #Individual Analyses
        for filename, analysis in analyses:
            content.append(self.create_styled_paragraph(f"Analysis of {filename}", styles['Title']))
            content.append(Spacer(1, 12))
            
            for line in analysis.split('\n'):
                if line.strip():
                    if line.strip().endswith(':'):
                        content.append(self.create_styled_paragraph(line, styles['Subtitle']))
                    else:
                        content.append(self.create_styled_paragraph(line, styles['BodyText']))
            
            content.append(PageBreak())

        doc.build(content)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalysisGUI(root)
    root.mainloop()