import os, io, requests, time, json
from typing import List
import google.generativeai as genai
import gradio as gr
from PIL import Image

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
    "stablediffusionapi/toonyou",
    "stablediffusionapi/real-cartoon-3d",
    "stablediffusionapi/realcartoon3d",
    "stablediffusionapi/disney-pixar-cartoon",
    "stablediffusionapi/pastel-mix-stylized-anime",
    "stablediffusionapi/anything-v5",    
    "nitrosocke/Ghibli-Diffusion",
    "jinaai/flat-2d-animerge",
    "Lykon/DreamShaper", 
    "SG161222/Realistic_Vision_V6.0_B1_noVAE",
    "black-forest-labs/FLUX.1-dev"
]

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
        
    # Set configuration inputs
    TabConfig.select(load_config, outputs=[*save_input_elements])

block.launch(inbrowser=True, share=True)