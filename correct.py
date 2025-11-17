import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from io import BytesIO
import torch
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
a = pipeline("text-generation", model="gpt2", tokenizer="gpt2", device = 0 if device=="cuda" else -1, return_full_text=True)

def generate(path):
  image = Image.open(path).convert("RGB")
  inputs = processor(images=image, return_tensors="pt").to(device)
  out = model.generate(**inputs, max_new_tokens=50)
  caption = processor.decode(out[0], skip_special_tokens = True)
  return caption

def b(prompt, max_new_tokens):
  result = a(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1, truncation=True)
  if isinstance(result, list):
    if "generated_text" in result[0]:
      return result[0]["generated_text"]
    elif "text" in result[0]:
      return result[0]["text"]
  else:
    return "failed"

def truncate(text, WordLimit):
  words = text.strip().split()
  return "".join(words[:WordLimit])

print("Select your choice.")
print("1. Caption - 5 words")
print("2. Description - 30 words")
print("3. Summary - 50 words")
print("4. Exit")

one = input("Enter image: ")
if not os.path.exists(one):
  print("Invalid image.")
  exit()

try:
  g = generate(one)
  print(g)
except Exception as e:
  print(e)

choice = input("Enter your choice: ")
if choice == "1":
  print(truncate(g,5))
elif choice == "2":
  description = b(g, max_new_tokens=60)
  print(truncate(description,30))
elif choice == "3":
  description = b(g, max_new_tokens=70)
  print(truncate(description,50))
else:
  exit()