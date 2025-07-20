import json
import os
from PIL import Image, ImageEnhance
import time

# Define paths based on the directory structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUBSET_FILE = os.path.join(BASE_DIR, "gqa_dataset", "processed", "gqa_val_subset_1000.json")
IMAGES_DIR = os.path.join(BASE_DIR, "gqa_dataset", "images")
AUGMENTED_IMAGES_DIR = os.path.join(BASE_DIR, "gqa_dataset", "images_augmented")
OUTPUT_SUBSET_FILE = os.path.join(BASE_DIR, "gqa_dataset", "processed", "gqa_val_subset_augmented.json")

# Create directories if they don't exist
os.makedirs(AUGMENTED_IMAGES_DIR, exist_ok=True)

def apply_augmentations(image):
    """Apply augmentations to the image and return a list of augmented images with their types."""
    augmented_images = []
    
    # Original image (no augmentation, for reference)
    augmented_images.append(("original", image))
    
    # Rotation (90 degrees)
    rotated_image = image.rotate(90, expand=True)
    augmented_images.append(("rotated_90", rotated_image))
    
    # Horizontal flip
    flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    augmented_images.append(("flipped_horizontal", flipped_image))
    
    # Brightness adjustment (increase by a factor of 1.2)
    enhancer = ImageEnhance.Brightness(image)
    bright_image = enhancer.enhance(1.2)
    augmented_images.append(("brightened", bright_image))
    
    return augmented_images

def augment_gqa_images():
    """Apply data augmentation to GQA images in the subset and create an augmented dataset."""
    # Start timing
    total_start_time = time.time()
    
    # Load the preprocessed subset
    print("Loading preprocessed subset...")
    with open(SUBSET_FILE, 'r') as f:
        subset_data = json.load(f)
    
    print(f"Loaded {len(subset_data)} image-question pairs")
    
    # Process each pair and apply augmentations
    augmented_data = []
    total_pairs = len(subset_data)
    
    for idx, item in enumerate(subset_data):
        image_path = item["image_path"]
        image_id = item["image_id"]
        question_id = item["question_id"]
        question = item["question"]
        answer = item["answer"]
        
        try:
            # Load the image
            image = Image.open(image_path).convert("RGB")
            
            # Apply augmentations
            augmented_images = apply_augmentations(image)
            
            # Save augmented images and create new entries
            for aug_type, aug_image in augmented_images:
                if aug_type == "original":
                    # Keep the original entry as is
                    augmented_data.append(item)
                else:
                    # Save the augmented image
                    aug_image_filename = f"{image_id}_{aug_type}.jpg"
                    aug_image_path = os.path.join(AUGMENTED_IMAGES_DIR, aug_image_filename)
                    aug_image.save(aug_image_path)
                    
                    # Create a new entry for the augmented image
                    new_item = {
                        "question_id": f"{question_id}_{aug_type}",
                        "image_id": f"{image_id}_{aug_type}",
                        "image_path": aug_image_path,
                        "question": question,
                        "answer": answer
                    }
                    augmented_data.append(new_item)
            
            # Print progress
            if (idx + 1) % 100 == 0 or idx + 1 == total_pairs:
                print(f"Processed {idx + 1}/{total_pairs} images... ({(idx + 1)/total_pairs*100:.1f}%)")
        
        except Exception as e:
            print(f"Error processing image {idx + 1}: {str(e)}")
            # Add the original item even if augmentation fails
            augmented_data.append(item)
    
    # Save the augmented dataset
    print("Saving augmented dataset...")
    with open(OUTPUT_SUBSET_FILE, 'w') as f:
        json.dump(augmented_data, f, indent=4)
    
    print(f"Saved {len(augmented_data)} image-question pairs to {OUTPUT_SUBSET_FILE}")
    print(f"Total processing time: {time.time() - total_start_time:.2f} seconds")

if __name__ == "__main__":
    augment_gqa_images()