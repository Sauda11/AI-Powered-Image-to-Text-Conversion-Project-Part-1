from config import HF_API_KEY
import requests
from PIL import Image
import io 
import os
from colorama import init, Fore, Style
import json
init(autoreset=True)
#
#utility function to send requets
#
def query_hf_api(api_url, payload=None, files=None, method= "post"):
  headers = {"Authourization": f"Bearer {HF_API_KEY}"}
  try:
       if method.lower() == "post":
           response = requests.post(api_url, headers=headers, json=payload, files=files)
       else:
            response = requests.get(api_url, headers=headers, params=payload)
       if response.status_code != 200:
           raise Exception(f"Error: {response.status_code} - {response.text}")
       return response.content
  except Exception as e:  
    print(f"{Fore.RED}Error while calling API: {e}")
    raise
def get_basic_caption(image, model="nlpconnect/vit-gpt2-image-captioning"):
  print(f"{Fore.YELLOW}Generating caption for image...")
  api_url = f"https://api-inference.huggingface.co/models/{model}"
  buffered = io.BytesIO()
  image.save(buffered, format="JPEG")
  buffered.seek(0)
  headers = {"Authorization": f"Bearer {HF_API_KEY}"}
  response = requests.post(api_url, headers=headers,  data=buffered.read())
  result = response.json()
  if isinstance(result,  dict) and "error" in result:
   return(f"Error: {result['error']}")

  caption = result[0].get("generated_text", "no caption generated")
  return caption
# task generate text using a text generatiion
def generate_text(prompt, model="gpt2", max_new_tokens=60):
     print(f"{Fore.YELLOW}Generating text for prompt: {prompt}")
     api_url = f"https://api-inference.huggingface.co/models/{model}"
     payload = {"inputs": prompt, "max_new_tokens": max_new_tokens}
     text_bytes = query_hf_api(api_url, payload)
     try:
          result = json.loads(text_bytes.decode("utf-8"))
     except Exception as e:
          raise  Exception(f"Error decoding response: {e}")
     if isinstance(result, dict) and "error" in result:
          raise Exception(f"Error: {result['error']}")
     generated = result[0].get("generated_text","")
     return generated
def truncate_text(text, word_limit):
     words = text.split()
     return "".join(words[:word_limit])
def print_menu():
     print(f"""{Style.BRIGHT}
{Fore.GREEN}=======================image to text conversion========
select output type:
1. Image caption(5 words)
2. Text generation(30 words)
3. summary(50 words)
4. Exit
""")
def main():
 image_path = input(f"{Fore.BLUE}Enter the path to the image (e.g., test.jpg): {Style.RESET_ALL} ") 
 if not os.path.exists(image_path):
          print(f"{Fore.RED}Image not found at path: {image_path}")
          return
 try:
      image = Image.open(image_path)
 except Exception as e:
      print(f"{Fore.RED}Error loading image: {e}")
      return
 basic_caption = get_basic_caption(image)
 print(f"{Fore.GREEN}Basic caption: {basic_caption}")
 while True:
      print_menu()     
      choice = input(f"{Fore.BLUE}Enter your choice (1-4): {Style.RESET_ALL}")
      if choice == "1":
           caption = truncate_text(basic_caption, 5)
           print(f"{Fore.GREEN}Image caption: {caption}")
      elif choice == "2":
           prompt_text = f"expand the following image caption into a detailed description: {basic_caption}"
           try:
                generated = generate_text(prompt_text, max_new_tokens=40)
                description = truncate_text(generated, 30)
                print(f"{Fore.GREEN}Generated text: {description}\n")
           except Exception as e:
                print(f"{Fore.RED}Error generating text: {e}")
      elif choice == "3":
          prompt_text = f"summarize the following image caption: {basic_caption}"
          try:
               generated = generate_text(prompt_text, max_new_tokens=60)
               summary = truncate_text(generated, 50)
               print(f"{Fore.GREEN}Summary: {summary}\n")
          except Exception as e:
                 print(f"{Fore.RED}Error generating summary: {e}")
      elif choice == "4":
             print(f"{Fore.YELLOW}Exiting...")
             break
      else:
            print(f"{Fore.RED}Invalid choice. Please enter a number between 1 and 4.")     
if __name__ == "__main__":
 main()

                
      

  
  

    
    