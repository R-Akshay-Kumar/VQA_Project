import json
import os
import random
import time

# Define paths based on the directory structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QUESTIONS_DIR = os.path.join(BASE_DIR, "gqa_dataset", "questions")
OUTPUT_DIR = os.path.join(BASE_DIR, "gqa_dataset", "processed")
VAL_QUESTIONS_FILE = os.path.join(QUESTIONS_DIR, "val_all_questions.json")  # Assuming this file exists

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_gqa(num_samples=1000, seed=42):
    """
    Preprocess GQA validation questions to extract a subset of image-question pairs.
    
    Args:
        num_samples (int): Number of image-question pairs to extract
        seed (int): Random seed for reproducibility
    """
    # Start timing the entire process
    total_start_time = time.time()
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load validation questions
    print("Loading validation questions...")
    load_start_time = time.time()
    if not os.path.exists(VAL_QUESTIONS_FILE):
        raise FileNotFoundError(f"Validation questions file not found at {VAL_QUESTIONS_FILE}")
    
    with open(VAL_QUESTIONS_FILE, 'r') as f:
        questions_data = json.load(f)
    
    print(f"Loaded {len(questions_data)} questions in {time.time() - load_start_time:.2f} seconds")
    
    # Extract image-question pairs
    print("Extracting image-question pairs...")
    extract_start_time = time.time()
    image_question_pairs = []
    counter = 0
    total_questions = len(questions_data)
    
    for question_id, question_info in questions_data.items():
        image_id = question_info.get('imageId')
        question = question_info.get('question')
        answer = question_info.get('answer')
        
        if image_id and question and answer:
            image_path = os.path.join(BASE_DIR, "gqa_dataset", "images", f"{image_id}.jpg")
            if os.path.exists(image_path):
                image_question_pairs.append({
                    "question_id": question_id,
                    "image_id": image_id,
                    "image_path": image_path,
                    "question": question,
                    "answer": answer
                })
        
        counter += 1
        if counter % 1000 == 0 or counter == total_questions:
            print(f"Processed {counter}/{total_questions} questions... ({(counter/total_questions)*100:.1f}%)")
    
    print(f"Extracted {len(image_question_pairs)} valid pairs in {time.time() - extract_start_time:.2f} seconds")
    
    # Sample a subset
    print("Sampling subset...")
    sample_start_time = time.time()
    if len(image_question_pairs) < num_samples:
        print(f"Warning: Only {len(image_question_pairs)} pairs available, requested {num_samples}")
        sampled_pairs = image_question_pairs
    else:
        sampled_pairs = random.sample(image_question_pairs, num_samples)
    
    print(f"Sampled {len(sampled_pairs)} pairs in {time.time() - sample_start_time:.2f} seconds")
    
    # Save the subset
    print("Saving subset to file...")
    save_start_time = time.time()
    output_file = os.path.join(OUTPUT_DIR, f"gqa_val_subset_{num_samples}.json")
    with open(output_file, 'w') as f:
        json.dump(sampled_pairs, f, indent=4)
    
    print(f"Saved {len(sampled_pairs)} image-question pairs to {output_file} in {time.time() - save_start_time:.2f} seconds")
    print(f"Total processing time: {time.time() - total_start_time:.2f} seconds")

if __name__ == "__main__":
    preprocess_gqa(num_samples=1000)