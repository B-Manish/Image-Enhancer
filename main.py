from PIL import Image
import cv2

def pixelate_image(input_path, output_path, pixel_size=10):
    # Open the original image
    image = Image.open(input_path)

    # Calculate the reduced size
    small_image = image.resize(
        (image.width // pixel_size, image.height // pixel_size),
        resample=Image.BILINEAR
    )

    # Scale it back up to original size
    result = small_image.resize(image.size, Image.NEAREST)

    # Save or show the result
    result.save(output_path)
    result.show()



def resize(img_path="data/nature.jpg"):
    img = cv2.imread(img_path)  # Reads image as BGR
    resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    cv2.imwrite("test.jpg", resized_img)    

# pixelate_image("test.jpg", "data/pixelated_test.jpg", pixel_size=5)
resize()