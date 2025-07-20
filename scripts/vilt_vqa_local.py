import os
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from datetime import datetime
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run VQA using ViLT model")
parser.add_argument("--image_path", required=True, help="Path to the image file")
parser.add_argument("--questions_file", required=True, help="Path to the questions text file")
parser.add_argument("--output_dir", default="./vqa_history", help="Directory to save session logs")
args = parser.parse_args()

# Paths
image_path = args.image_path
questions_file = args.questions_file
history_dir = args.output_dir
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = r"C:\My Projects\VQA_Project2\gqa_dataset\finetuned_model"

# Validate inputs
if not os.path.exists(image_path) or not os.path.exists(questions_file):
    print("Error: Image path or questions file does not exist")
    raise ValueError("Image path or questions file does not exist")
if not os.path.exists(model_dir):
    print(f"Error: Model directory {model_dir} does not exist")
    raise ValueError(f"Model directory {model_dir} does not exist")

# Create history directory if it doesn't exist
os.makedirs(history_dir, exist_ok=True)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model
print("Loading processor and model from fine-tuned directory...")
processor = ViltProcessor.from_pretrained(model_dir)
model = ViltForQuestionAnswering.from_pretrained(model_dir).to(device)

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")
image = image.resize((384, 384), Image.Resampling.LANCZOS)  # Resize to match training

# Load the questions
questions = []
with open(questions_file, "r") as f:
    questions = [line.strip() for line in f if line.strip()]

# Process each question
answers = []
confidences = []
print(f"Processing image: {os.path.basename(image_path)}")
for question in questions:
    # Ensure the question ends with a question mark
    if not question.endswith("?"):
        question = question + "?"

    # Process the question with the model
    encoding = processor(image, question, return_tensors="pt").to(device)

    # Debug: Print the processed inputs
    print(f"Processed input keys: {list(encoding.keys())}")

    # Get the model outputs
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        idx = logits.argmax(-1).item()
        confidence = probs[0, idx].item()  # Confidence score
        answer = model.config.id2label[idx]

    answers.append(answer)
    confidences.append(confidence)

    # Display the answer with confidence
    print(f"Question: {question}, Answer: {answer}, Confidence: {confidence:.4f}")

# Save the questions and answers to a text file in the history folder
session_filename = f"session_{timestamp}.txt"
session_filepath = os.path.join(history_dir, session_filename)
if questions:
    with open(session_filepath, "w") as f:
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write("\n")
        for q, a, c in zip(questions, answers, confidences):
            f.write(f"Question: {q}\n")
            f.write(f"Answer: {a}\n")
            f.write(f"Confidence: {c:.4f}\n")
            f.write("\n")
    print(f"Session log saved to: {session_filepath}")
else:
    print("No questions were provided.")
    with open(session_filepath, "w") as f:
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write("\n")
        f.write("No questions were provided.\n")
    print(f"Session log saved to: {session_filepath}")