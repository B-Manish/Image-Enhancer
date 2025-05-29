import cv2
import os

video_path = 'munnar.mp4'
output_dir = 'frames'

os.makedirs(output_dir, exist_ok=True)


cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # No more frames

    # Save the frame as an image
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"Done! Extracted {frame_count} frames to '{output_dir}/'")
