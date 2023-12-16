from PIL import Image
import os

for idx in range(1,10):
    # Load the original image
    original_image_path = f"/home/prathosh/goirik/blended-latent-diffusion/outputs/{idx}.jpg"
    original_image = Image.open(original_image_path)

    # Get the dimensions of the original image
    width, height = original_image.size

    # Calculate the number of 512x512 images that can be extracted
    num_images_x = width // 512
    num_images_y = height // 512

    # Create a directory with the same name as the original image (without extension)
    output_folder = os.path.splitext(original_image_path)[0]

    # Create the directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Loop to split and save the images
    for i in range(num_images_x):
        for j in range(num_images_y):
            # Define the region to crop
            left = i * 512
            upper = j * 512
            right = left + 512
            lower = upper + 512

            # Crop the region
            cropped_image = original_image.crop((left, upper, right, lower))

            # Save the cropped image
            output_path = os.path.join(output_folder, f"image_{i}_{j}.jpg")
            cropped_image.save(output_path)

    print("Images have been split and saved to the folder:", output_folder)
