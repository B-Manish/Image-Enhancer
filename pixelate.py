# from PIL import Image

# def pixelate_image(input_path, output_path, pixel_size=10):
#     # Open the original image
#     image = Image.open(input_path)

#     # Calculate the reduced size
#     small_image = image.resize(
#         (image.width // pixel_size, image.height // pixel_size),
#         resample=Image.BILINEAR
#     )

#     # Scale it back up to original size
#     result = small_image.resize(image.size, Image.NEAREST)

#     # Save or show the result
#     result.save(output_path)
#     result.show()

# # Example usage
# pixelate_image("parrot.jpg", "pixelated_output.jpg", pixel_size=50)


import os
from PIL import Image

def pixelate_image(input_path, output_path, pixel_size=10):
    image = Image.open(input_path)

    # Downscale and then upscale
    small_image = image.resize(
        (image.width // pixel_size, image.height // pixel_size),
        resample=Image.BILINEAR
    )
    result = small_image.resize(image.size, Image.NEAREST)
    result.save(output_path)

# Paths
high_res_dir = "data/high_res"
low_res_dir = "data/low_res"
pixel_size = 5  # Change this to increase or decrease pixelation

# Ensure the output folder exists
os.makedirs(low_res_dir, exist_ok=True)

# Loop over all image files in high_res
for filename in os.listdir(high_res_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        input_path = os.path.join(high_res_dir, filename)
        output_path = os.path.join(low_res_dir, filename)
        pixelate_image(input_path, output_path, pixel_size)
        print(f"Pixelated: {filename}")
