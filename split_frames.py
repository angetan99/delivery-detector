import cv2
import numpy as np
import os

interval = 15
input_folder = "Clips"
output_folder = "Frames"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".MP4"):
        video_path = os.path.join(input_folder, filename)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                # frame = cv2.resize(frame, (224, 224))
                output_filename = f"{os.path.splitext(filename)[0]}_frame{frame_count}.jpeg"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, frame)
                print(f"Saved frame {frame_count} from {filename} as {output_filename}")

            frame_count += 1

        cap.release()

print("Frame extraction completed.")

# Manually sort frames into delivery and non-delivery folders and train/validation split