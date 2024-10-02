import cv2
import tkinter as tk
from tkinter import Label
from simple_facerec import SimpleFacerec
from PIL import Image, ImageTk


class FaceRecognitionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.video_source = 0

        # Load SimpleFacerec
        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images("Images/")

        # Open video source (by default this is the webcam)
        self.vid = cv2.VideoCapture(self.video_source)

        # Create a canvas that can fit the video source size
        self.canvas = tk.Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Button to start face recognition
        self.btn_snapshot = tk.Button(window, text="Start", width=50, command=self.start_video)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)

        # Button to quit the application
        self.btn_quit = tk.Button(window, text="Quit", width=50, command=window.quit)
        self.btn_quit.pack(anchor=tk.CENTER, expand=True)

        self.delay = 10
        self.update()

        self.window.mainloop()

    def start_video(self):
        self.update()

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()

        if ret:
            # Face recognition logic
            face_locs, face_names = self.sfr.detect_known_faces(frame)
            for face_loc, name in zip(face_locs, face_names):
                top, right, bottom, left = face_loc
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert the frame to a format that tkinter understands
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.window.imgtk = imgtk

        self.window.after(self.delay, self.update)


# Create a window and pass it to the Application class
FaceRecognitionApp(tk.Tk(), "Face Recognition App")
