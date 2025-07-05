import cv2
import os

frame_length=512
frame_height=512
video_path = 'munnar.mp4'
output_dir = 'data/high_res'

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # No more frames

    # Resize the frame to desired size
    resized_frame = cv2.resize(frame, (frame_length, frame_height), interpolation=cv2.INTER_AREA)

    # Save the resized frame as an image
    frame_filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
    cv2.imwrite(frame_filename, resized_frame)
    frame_count += 1

cap.release()
print(f"✅ Done! Extracted and resized {frame_count} frames to '{output_dir}/'")
