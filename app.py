import os, io, requests, time, json
os.environ["SM_FRAMEWORK"] = "tf.keras"
from typing import List
import google.generativeai as genai
import gradio as gr
from PIL import Image
import segmentation_models as sm
import cv2
import numpy as np
import keras

HUGGING_FACE_API_KEY=""
GOOGLE_API_KEY = ""
MAX_PROMPT_TOKENS = 80


chat_engine = None
chat = None

config_dict = {}
latested_llm_model = None
#gemini safety settings
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
]

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,  
  "max_output_tokens": 2048
}

#chat llm models
llm_model_names = [   
]

#diffusion models
diffusion_model_names = [
	"stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-3.5-large",
    "black-forest-labs/FLUX.1-dev",
    "Jovie/Midjourney",
    "XLabs-AI/flux-RealismLora",
    "prithivMLmods/Canopus-LoRA-Flux-UltraRealism-2.0"
]

# Download model
os.system('wget https://huggingface.co/Armandoliv/cars-parts-segmentation-unet-resnet18/resolve/main/best_model.h5')
os.system('pip -qq install pycocotools @ git+https://github.com/philferriere/cocoapi.git@2929bd2ef6b451054755dfd7ceb09278f935f7ad#subdirectory=PythonAPI')

# Color definitions and classes
c = ['_background_', 'back_bumper', 'back_glass', 'back_left_door', 'back_left_light', 'back_right_door', 
    'back_right_light', 'front_bumper', 'front_glass', 'front_left_door', 'front_left_light', 
    'front_right_door', 'front_right_light', 'hood', 'left_mirror', 'right_mirror', 'tailgate', 'trunk', 'wheel']
colors = [(245,255,250), (75,0,130), (0,255,0), (32,178,170), (0,0,255), (0,255,255), (255,0,255), (128,0,128), (255,140,0),
          (85,107,47), (102,205,170), (0,191,255), (255,0,0), (255,228,196), (205,133,63),
          (220,20,60), (255,69,0), (143,188,143), (255,255,0)]

# Initialize model
sm.set_framework('tf.keras')
sm.framework()
BACKBONE = 'resnet18'
n_classes = 19
activation = 'softmax'
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
model.load_weights('best_model.h5')

def get_colored_segmentation_image(seg_arr, n_classes, colors=colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
          # print(sum(sum(seg_arr_c)), colors[c] )
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img/255

def get_legends(class_names, colors, tags):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))
    j = 0
    for (i, (class_name, color)) in class_names_colors:
        if i in tags:
          color = [int(c) for c in color]
          cv2.putText(legend, class_name, (5, (j * 25) + 17),
                      cv2.FONT_HERSHEY_COMPLEX, 0.35, (0, 0, 0), 1)
          cv2.rectangle(legend, (100, (j* 25)), (125, (j * 25) + 25),
                        tuple(color), -1)
          j +=1
    return legend



def preprocess_image(path_img):
  img = Image.open(path_img)
  ww = 512
  hh = 512
  img.thumbnail((hh, ww))
  i = np.array(img)
  ht, wd, cc= i.shape

  # create new image of desired size and color (blue) for padding
  color = (0,0,0)
  result = np.full((hh,ww,cc), color, dtype=np.uint8)

  # copy img image into center of result image
  result[:ht, :wd] = img
  return result, ht, wd

def concat_lengends(seg_img, legend_img):

  new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
  new_w = seg_img.shape[1] + legend_img.shape[1]

  out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

  out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
  out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

  return out_img

def main_convert(filename):

  print(filename)
  #load the image
  img_path = filename
  img = Image.open(img_path).convert("RGB")
  tags = []

  #preprocess the image
  img_scaled_arr = preprocess_image(img_path)
  image = np.expand_dims(img_scaled_arr[0], axis=0)

  #make the predictions
  pr_mask = model.predict(image).squeeze()
  pr_mask_int = np.zeros((pr_mask.shape[0],pr_mask.shape[1]))

  #filter the smallest noisy segments
  kernel = np.ones((5, 5), 'uint8')

  for i in range(1,19):
    array_one = np.round(pr_mask[:,:,i])
    op = cv2.morphologyEx(array_one, cv2.MORPH_OPEN, kernel)
    if sum(sum(op ==1)) > 100:
      tags.append(i)
      pr_mask_int[op ==1] = i 

  img_segmented = np.array(Image.fromarray(pr_mask_int[:img_scaled_arr[1], :img_scaled_arr[2]]).resize(img.size))

  seg = get_colored_segmentation_image(img_segmented,19, colors=colors)

  fused_img = ((np.array(img)/255)/2 + seg/2).astype('float32')

  seg = Image.fromarray((seg*255).astype(np.uint8))
  fused_img  = Image.fromarray((fused_img *255).astype(np.uint8))

  #get the legends
  legend_predicted = get_legends(c, colors, tags)

  final_img = concat_lengends(np.array(fused_img), np.array(legend_predicted))

  return final_img, seg
  

def list_llm_models() -> List[str]:    
    if len(llm_model_names) == 0:
        if GOOGLE_API_KEY:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    llm_model_names.append(m.name.split('models/')[1])
        llm_models = list(llm_model_names)

    return sorted(sorted(llm_models, key=lambda t: t[1]), key=lambda t: t[0], reverse=False)

def list_diffusion_models() -> List[str]:
    models = list(diffusion_model_names)
    return sorted(sorted(models, key=lambda t: t[1]), key=lambda t: t[0], reverse=False)

def load_config():
    global HUGGING_FACE_API_KEY, GOOGLE_API_KEY, MAX_PROMPT_TOKENS, config_dict, generation_config

    if os.path.exists("./config.json"):
        with open("./config.json", "r", encoding="utf8") as file:
            config = file.read()

        config_dict = json.loads(config)
        HUGGING_FACE_API_KEY = config_dict["huggingface_api_key"]
        GOOGLE_API_KEY = config_dict["google_api_key"]
        MAX_PROMPT_TOKENS = config_dict["max_prompt_tokens"]
        generation_config["temperature"] = config_dict["temperature"]
        generation_config["top_p"] = config_dict["top_p"]
        generation_config["top_k"] = config_dict["top_k"]          
        generation_config["max_output_tokens"] = config_dict["max_output_tokens"]
    
    return [*config_dict.values()]

def save_config(*args):
    global config_dict
    
    values_dict = zip(config_dict.keys(), args)
    config_dict_values = dict(values_dict)
    google_api_key = GOOGLE_API_KEY    
    status=""
    try:
        with open('./config.json', 'w') as f:
            json.dump(config_dict_values, f,indent=2)
        load_config()
        if not google_api_key:
            initialize_chat_engine()
    except:
        status = "<center><h3 style='color: #E74C3C;'>There was an error saving the settings!</h3></center>"
        pass
    
    return gr.Tabs(selected=0), status, gr.Dropdown(list_llm_models())

def initialize_chat_engine():
    global chat_engine
    genai.configure(api_key=GOOGLE_API_KEY)

def initialize_bot(llm_model):
    global chat, chat_engine, latested_llm_model   
    chat_engine = genai.GenerativeModel(llm_model, safety_settings=safety_settings, generation_config=generation_config)
    chat = chat_engine.start_chat(history=[])
    latested_llm_model = llm_model

def clean_prompt(prompt:str):
    prompt = prompt.replace("\n\n*","").strip()
    prompt = prompt.replace("English: ","")
    prompt = prompt.replace("* ","")
    return prompt

def request(model, prompt):
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    status=""
    im = None
    try:
        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response
        
        data = query({"inputs": prompt,
                      "negative_prompt":"(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green, duplicated"})
        im = Image.open(io.BytesIO(data.content))
    except:
        status="<center><h3 style='color: #E74C3C;'>Oops! There was an error generating your response, please try again!</h3><center>"
        pass
        
    return im, status

def predict(message, chat_history, llm_model, diffusion_model):                    
   
    status=""
    generated_prompt=""

    if not message:        
        status="<center><h3 style='color: #E74C3C;'>You need to ask me something!</h3><center>"
        return "", None, None, status, generated_prompt
    
    if not HUGGING_FACE_API_KEY or not GOOGLE_API_KEY:        
        status="<center><h3 style='color: #E74C3C;'>You need to set the API keys in the 'Settings' tab before proceeding!</h3></center>"
        return "", None, None, status, generated_prompt
    
    if chat_engine is None or latested_llm_model != llm_model:
        initialize_bot(llm_model)

    image = None
    if "/imagine" in message: #Generate image command
        msg = message.split('/imagine')[1].strip()
        text_question=f"You are creating a prompt for Stable Diffusion to generate an image. Please generate a text prompt for {msg}. Respond only with the prompt itself in the English language, but beautify it as needed, but keep it below {MAX_PROMPT_TOKENS} tokens"
        
        try:
            response = chat.send_message(f"{text_question} {msg}.")
            response.resolve()        
            response_text = response.text.split('\n\n')
        except:
            status="<center><h3 style='color: #E74C3C;'>Oops! There was an error generating your response, please try again!</h3><center>"
            pass
            return "", None, None, status, generated_prompt  
        
        english_prompt_index = 0       
        
        if len(response_text) == 0:
            status="<center><h3 style='color: #E74C3C;'>Oops! There was an error generating your response, please try again!</h3><center>"
            return "", None, None, status, generated_prompt
                
        english_prompt = clean_prompt(response_text[english_prompt_index].strip())        
        bot_message = f"I imagined that: '{english_prompt}'"
        generated_prompt=f"<center><h3 style='color: #2E86C1;'>{english_prompt}</h3><center>"

        image, status = request(diffusion_model, english_prompt)
        if image:
            image = [image]
    else:
        response = chat.send_message(message)
        response.resolve()      
        bot_message = response.text

    chat_history.append((message, bot_message))
    time.sleep(2)
    return "", chat_history, image, status, generated_prompt

# Initialize configuration
load_config()
if GOOGLE_API_KEY:
    initialize_chat_engine()

# Description
title = r"""
<h1 align="center">Gemini-To-Diffusion: Text-to-Image Generation</h1>
"""

description = r"""
<b>Official 🤗 Gradio demo</b> for <a href='https://github.com/DEVAIEXP/gemini-to-diffusion'><b>Text-to-Image Generation</b></a>.<br>
How to use:<br>
1. Configure your <b>Google API</b> key and your <b>Huggingface API</b> key in the 'Settings' tab.
2. Ask what you want to visualize in the Question field, with the command <b>/imagine</b> before the question.
3. Click <b>Submit</b>.
"""

css = """
    footer {visibility: hidden},
    .gradio-container {width: 85% !important}
    """
block = gr.Blocks(theme="soft", css=css)
with block:
    
    # description
    gr.Markdown(title)
    gr.Markdown(description)
    
    with gr.Tabs() as tabs:
        with gr.TabItem("Chat", id=0):  
            with gr.Row(equal_height=True):
                with gr.Column():
                    chatbot = gr.Chatbot()
                    msg = gr.Textbox(label="Question", value="/imagine ", placeholder="You can ask me what I need to generate, example: /imagine a cat")            
                    llm_model = gr.Dropdown(list_llm_models(), value="gemini-1.5-pro-latest", label='Chat model')  
                    diffusion_model = gr.Dropdown(list_diffusion_models(), value="stabilityai/stable-diffusion-xl-base-1.0", label='Image model')  
                with gr.Column(scale=1):
                    gallery = gr.Gallery(label="Generated Image", columns=1, rows=1, height=640, interactive=False,format="png")
                    generated_prompt = gr.HTML(elem_id="generated_tatus", value="")
                    status = gr.HTML(elem_id="status", value="")
            with gr.Row():            
                submit_btn = gr.Button(value="🔎Submit")            
                clear = gr.ClearButton([msg, chatbot, gallery, generated_prompt, status], value="🔁Clear")

            msg.submit(predict, [msg, chatbot, llm_model, diffusion_model], [msg, chatbot, gallery, status, generated_prompt])
            submit_btn.click(predict, [msg, chatbot, llm_model, diffusion_model], [msg, chatbot, gallery, status, generated_prompt])
        with gr.TabItem("Settings", id=1) as TabConfig:
            with gr.Row():
                with gr.Column():
                    google_api_key = gr.Textbox(label="Google API Key", placeholder="Enter your API-Key here")
                    huggingface_api_key = gr.Textbox(label="Huggingface API Key", placeholder="Enter your API-Key here")
                with gr.Column():
                    temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, interactive=True, label="Temperature")
                    top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.1, interactive=True, label="Top P")
                    top_k = gr.Slider(minimum=0.0, maximum=1.0, value=1, step=0.1, interactive=True, label="Top K")
                    max_prompt_tokens = gr.Slider(minimum=0, maximum=2048, value=80, step=20, interactive=True, label="Max output prompt tokens")    
                    max_output_tokens = gr.Slider(minimum=0, maximum=8192, value=2048, step=64, interactive=True, label="Max output tokens")    
            
            save_btn = gr.Button(value="💾Save")        
            save_input_elements = [google_api_key, huggingface_api_key, temperature, top_p, top_k, max_prompt_tokens,  max_output_tokens]
            save_btn.click(save_config,inputs=[*save_input_elements], outputs=[tabs, status, llm_model])
        with gr.TabItem("Car Segmentation"):
            with gr.Column():
                gr.Markdown("""
                    # Car Parts Segmentation
                    This demo uses AI Models to detect 18 different parts of cars.
                """)
                
                with gr.Row():
                    # Left column for input
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            type="filepath",
                            label="Car Image",
                            elem_id="input_image"
                        )
                        
                        with gr.Row():
                            clear_btn = gr.Button("Clear", size="sm")
                            submit_btn = gr.Button("Submit", size="sm", variant="primary")

                    # Right column for outputs
                    with gr.Column(scale=1):
                        detected_segments = gr.Image(
                            type="pil",
                            label="Detected Segments Image",
                            elem_id="detected_segments"
                        )
                        
                        segment_image = gr.Image(
                            type="pil",
                            label="Segment Image",
                            elem_id="segment_image"
                        )

                # Examples section at the bottom
                gr.Examples(
                    examples=[["test_image.png"]],
                    inputs=input_image,
                    outputs=[detected_segments, segment_image],
                    fn=main_convert,
                    cache_examples=True
                )

                # Button functionality
                clear_btn.click(
                    fn=lambda: (None, None),
                    inputs=None,
                    outputs=[detected_segments, segment_image]
                )

                submit_btn.click(
                    fn=main_convert,
                    inputs=input_image,
                    outputs=[detected_segments, segment_image]
                )
        
    # Set configuration inputs
    TabConfig.select(load_config, outputs=[*save_input_elements])

block.launch(inbrowser=True, share=True)