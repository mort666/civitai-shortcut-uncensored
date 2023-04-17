import requests
import json
import gradio as gr

import time
import threading
import urllib.request
import urllib.error
import os
from tqdm import tqdm
import re
from requests.exceptions import ConnectionError
import urllib.request
import shutil
from PIL import Image

import modules.extras
import modules.scripts as scripts

from modules import script_callbacks
from modules.ui_common import plaintext_to_html
from modules import images

import sys
import traceback
from modules import sd_samplers
import piexif
import piexif.helper

# this is the default root path
root_path = os.getcwd()

PLACEHOLDER = "<no select>"

current_model_type = None

# Set the URL for the API endpoint
json_data = None

page_dict = {
    "limit" : 50,
}

url_dict = {
    "modelPage":"https://civitai.com/models/",
    "modelId": "https://civitai.com/api/v1/models/",
    "modelVersionId": "https://civitai.com/api/v1/model-versions/",
    "hash": "https://civitai.com/api/v1/model-versions/by-hash/"
}

folders_dict = {
    "Checkpoint": os.path.join("models","Stable-diffusion"),
    "LORA": os.path.join("models","Lora"),
    "Hypernetwork": os.path.join("models","hypernetworks"),
    "TextualInversion": os.path.join("embeddings"),            
    "AestheticGradient": os.path.join("extensions","stable-diffusion-webui-aesthetic-gradients","aesthetic_embeddings"),
    "VAE": os.path.join("models","VAE"),        
    "Controlnet" : os.path.join("extensions","sd-webui-controlnet","models"),
    "Poses" : os.path.join("models","Poses"),
    "ANLORA": os.path.join("extensions","sd-webui-additional-networks","models","lora"),
    "Unknown": os.path.join("models","Unknown"),
}

content_types_dict = {
    "All" : "",    
    "Checkpoint": "Checkpoint",
    "LORA": "LORA",
    "Hypernetwork": "Hypernetwork",
    "TextualInversion": "TextualInversion",            
    "AestheticGradient":"AestheticGradient",    
    "Controlnet" : "Controlnet", 
    "Poses":"Poses"
}

models_exts = (".bin", ".pt", ".safetensors", ".ckpt")
    
def printD(msg):
    print(f"Civitai Browser: {msg}")   
    
def download_file(url, file_name):
    # Maximum number of retries
    max_retries = 5

    # Delay between retries (in seconds)
    retry_delay = 10

    while True:
        # Check if the file has already been partially downloaded
        if os.path.exists(file_name):
            # Get the size of the downloaded file
            downloaded_size = os.path.getsize(file_name)

            # Set the range of the request to start from the current size of the downloaded file
            headers = {"Range": f"bytes={downloaded_size}-"}
        else:
            downloaded_size = 0
            headers = {}

        # Split filename from included path
        tokens = re.split(re.escape('\\'), file_name)
        file_name_display = tokens[-1]

        # Initialize the progress bar
        progress = tqdm(total=1000000000, unit="B", unit_scale=True,
                        desc=f"Downloading {file_name_display}", initial=downloaded_size, leave=False)

        # Open a local file to save the download
        with open(file_name, "ab") as f:
            while True:
                try:
                    # Send a GET request to the URL and save the response to the local file
                    response = requests.get(url, headers=headers, stream=True)

                    # Get the total size of the file
                    total_size = int(response.headers.get("Content-Length", 0))

                    # Update the total size of the progress bar if the `Content-Length` header is present
                    if total_size == 0:
                        total_size = downloaded_size
                    progress.total = total_size

                    # Write the response to the local file and update the progress bar
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                            progress.update(len(chunk))

                    downloaded_size = os.path.getsize(file_name)
                    # Break out of the loop if the download is successful
                    break
                except ConnectionError as e:
                    # Decrement the number of retries
                    max_retries -= 1

                    # If there are no more retries, raise the exception
                    if max_retries == 0:
                        raise e

                    # Wait for the specified delay before retrying
                    time.sleep(retry_delay)

        # Close the progress bar
        progress.close()
        downloaded_size = os.path.getsize(file_name)
        # Check if the download was successful
        if downloaded_size >= total_size:
            print(f"{file_name_display} successfully downloaded.")
            break
        else:
            print(f"Error: File download failed. Retrying... {file_name_display}")   
            
    
def download_file_thread(url, file_name, use_new_folder, model_name, lora_an):
    if file_name and file_name != PLACEHOLDER and model_name and model_name != PLACEHOLDER and url and len(url.strip()) > 0:
        model_folder = make_new_folder(current_model_type, use_new_folder, model_name, lora_an)
        path_to_new_file = os.path.join(model_folder, file_name)

        thread = threading.Thread(target=download_file,args=(url, path_to_new_file))
        # Start the thread
        thread.start()
    return f"Download started"


def make_new_folder(content_type, use_new_folder, model_name, lora_an):
    
    try:
        folder = folders_dict[content_type]
    except: 
        # 알수 없는 타입일경우 언노운 폴더에 저장한다.
        if content_type:
            tmp_type = content_type.replace(" ", "_").replace("(", "_").replace(")", "_").replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_")
            folder = os.path.join(folders_dict['Unknown'], tmp_type ,model_name.replace(" ", "_").replace("(", "_").replace(")", "_").replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_"))
        else:                    
            folder = os.path.join(folders_dict['Unknown'], content_type,model_name.replace(" ", "_").replace("(", "_").replace(")", "_").replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_"))
    
    if lora_an and content_type == "LORA":
        folder = folders_dict['ANLORA']        
            
    model_folder = folder  
    if content_type == "Checkpoint" or content_type == "Hypernetwork" or content_type =="LORA" or content_type == "Poses" or content_type == "TextualInversion" or content_type == "AestheticGradient":
        model_folder = os.path.join(model_folder, model_name.replace(" ", "_").replace("(", "_").replace(")", "_").replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_"))
        
    if use_new_folder:
        model_folder = os.path.join(model_folder, "new") 
                
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
                
    return model_folder


def save_trained_tags(model_version_name, use_new_folder, trained_words, model_name, lora_an):    
    message = ""
    if trained_words and len(trained_words.strip()) > 0 and model_version_name and model_version_name != PLACEHOLDER and model_name and model_name != PLACEHOLDER:
        model_folder = make_new_folder(current_model_type, use_new_folder, model_name, lora_an)                
        path_to_new_file = os.path.join(model_folder, (model_name + "_" + model_version_name).replace("*", "_").replace("?", "_").replace("\"", "_").replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_") + ".txt")            
        if not os.path.exists(path_to_new_file):
            with open(path_to_new_file, 'w') as f:
                f.write(trained_words)
        if os.path.getsize(path_to_new_file) == 0:
            message = "Current model doesn't have any trained tags"
            print(message)
        else:
            message = "Trained tags saved as text file"
            print(message)
            
    return message


def request_civitai_api(api_url=None):
    # Make a GET request to the API
    response = requests.get(api_url)

    # Check the status code of the response
    if response.status_code != 200:
        print("Request failed with status code: {}".format(response.status_code))
        exit()

    data = json.loads(response.text)
    return data


def api_to_data(content_type, sort_type, search_term=None):    
    if search_term:
        search_term = search_term.strip().replace(" ", "%20")
    
    c_types = content_types_dict[content_type]   
    urls = f"{url_dict['modelId']}?limit={page_dict['limit']}"             
    if c_types and len(c_types) > 0:        
        urls = f"{urls}&types={c_types}"
    urls = f"{urls}&sort={sort_type}&query={search_term}"    
                
    return request_civitai_api(urls)


def api_next_page(next_page_url=None):
    global json_data
    try:
        json_data['metadata']['nextPage']
    except:
        return
    if json_data['metadata']['nextPage'] is not None:
        next_page_url = json_data['metadata']['nextPage']
    if next_page_url is not None:
        return request_civitai_api(next_page_url)


def api_prev_page(prev_page_url=None):
    global json_data
    try:
        json_data['metadata']['prevPage']
    except:
        return
    if json_data['metadata']['prevPage'] is not None:
        prev_page_url = json_data['metadata']['prevPage']
    if prev_page_url is not None:
        return request_civitai_api(prev_page_url)    


def update_prev_page(show_nsfw):
    global json_data   
    tmp_json_data = api_prev_page()
    
    if tmp_json_data:
        json_data = tmp_json_data
        
    models_name=[]
    try:
        json_data['items']
    except TypeError:
        return gr.Dropdown.update(choices=[], value=None), gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None),gr.Textbox.update(value=None)
    if show_nsfw:
        for model in json_data['items']:
            models_name.append(model['name'])
    else:
        for model in json_data['items']:
            temp_nsfw = model['nsfw']
            if not temp_nsfw:
                models_name.append(model['name'])
    return gr.Dropdown.update(choices=[PLACEHOLDER] + [v for v in models_name], value=PLACEHOLDER), gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None),gr.Textbox.update(value=None)


def update_next_page(show_nsfw):
    global json_data
    tmp_json_data = api_next_page()
    
    if tmp_json_data:
        json_data = tmp_json_data
            
    models_name = []
    try:
        json_data['items']
    except TypeError:
        return gr.Dropdown.update(choices=[], value=None), gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None),gr.Textbox.update(value=None)
    if show_nsfw:
        for model in json_data['items']:
            models_name.append(model['name'])
    else:
        for model in json_data['items']:
            temp_nsfw = model['nsfw']
            if not temp_nsfw:
                models_name.append(model['name'])
    return gr.Dropdown.update(choices=[PLACEHOLDER] + [v for v in models_name], value=PLACEHOLDER), gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None),gr.Textbox.update(value=None)


def update_models_list_nsfw(show_nsfw):
    global json_data
    models_name = []
    try:
        json_data['items']
    except TypeError:
        return gr.Dropdown.update(choices=[], value=None), gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None),gr.Textbox.update(value=None)
    if show_nsfw:
        for model in json_data['items']:
            models_name.append(model['name'])
    else:
        for model in json_data['items']:
            temp_nsfw = model['nsfw']
            if not temp_nsfw:
                models_name.append(model['name'])
    return gr.Dropdown.update(choices=[PLACEHOLDER] + [v for v in models_name], value=PLACEHOLDER), gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None),gr.Textbox.update(value=None)

    
def update_models_list(content_type, sort_type, search_term, show_nsfw):
    global json_data
    json_data = api_to_data(content_type, sort_type, search_term)
    models_name=[]
    if show_nsfw:
        for model in json_data['items']:
            models_name.append(model['name'])
    else:
        for model in json_data['items']:
            temp_nsfw = model['nsfw']
            if not temp_nsfw:
                models_name.append(model['name'])
    return gr.Dropdown.update(choices=[PLACEHOLDER] + [v for v in models_name], value=PLACEHOLDER), gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None),gr.Textbox.update(value=None)


def update_model_versions(model_name=None):
    if model_name is not None and model_name != PLACEHOLDER:
        global json_data
        versions_name = []
        for model in json_data['items']:
            if model['name'] == model_name:
                for model_version in model['modelVersions']:
                    versions_name.append(model_version['name'])
                    
        return gr.Dropdown.update(choices=[PLACEHOLDER] + [v for v in versions_name], value=PLACEHOLDER),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None),gr.Textbox.update(value=None)
    else:
        return gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None),gr.Textbox.update(value=None)


def update_dl_url(model_name=None, model_version_name=None, model_filename=None):
    if model_filename:
        global json_data
        dl_url = None
        for model in json_data['items']:
            if model['name'] == model_name:
                for model_version in model['modelVersions']:
                    if model_version['name'] == model_version_name:
                        for file in model_version['files']:
                            if file['name'] == model_filename:
                                dl_url = file['downloadUrl']
        return gr.Textbox.update(value=dl_url)
    else:
        return gr.Textbox.update(value=None)


def update_model_version_info(model_name=None, model_version_name=None):
    global current_model_type 
    global json_data
    output_html = ""
    output_training = ""

    #dl_dict = {}
    files_name = []
    
    html_typepart = ""
    html_creatorpart = ""
    html_trainingpart = ""
    html_modelpart = ""
    html_versionpart = ""
    html_descpart = ""
    html_dnurlpart = ""
    html_imgpart = ""
    html_modelurlpart = ""
    model_version_id = ""
    
    if model_name and model_version_name and model_name != PLACEHOLDER and model_version_name != PLACEHOLDER:       
        current_model_type = None
        
        for model in json_data['items']:
            if model['name'] == model_name:
                
                # 선택한 모델의 타입을 저장한다
                current_model_type = model['type']
                html_typepart = f'<br><b>Type: {current_model_type}</b>'
                
                model_id = model['id']    
                model_uploader = model['creator']['username']
                html_creatorpart = f"<br><b>Uploaded by:</b> {model_uploader}"
                
                model_url = url_dict["modelPage"]+str(model_id)
                html_modelpart = f'<br><b>Model: <a href="{model_url}" target="_blank">{model_name}</a></b>'
                html_modelurlpart = f'<br><b><a href="{model_url}" target="_blank">Civitai Hompage Here</a></b><br>'

                html_versionpart = f"<br><b>Version:</b> {model_version_name}"
                                        
                if model['description']:
                    html_descpart = f"<br><b>Description</b><br>{model['description']}<br>"
                    
                for model_version in model['modelVersions']:
                    if model_version['name'] == model_version_name:
                        model_version_id = model_version['id']
                        if model_version['trainedWords']:
                            output_training = ", ".join(model_version['trainedWords'])
                            html_trainingpart = f'<br><b>Training Tags:</b> {output_training}'

                        for file in model_version['files']:
                            #dl_dict[file['name']] = file['downloadUrl']
                            files_name.append(file['name'])
                            html_dnurlpart = html_dnurlpart + f"<br><a href={file['downloadUrl']}><b>Download Here</b></a>"
                        
                        # html_imgpart = "<HEAD><style>img { display: inline-block; }</style></HEAD><div class='column'>"
                        # for pic in model_version['images']:
                        #     image_url = get_full_size_image_url(pic["url"], pic["width"])
                        #     html_imgpart = html_imgpart + f'<img src="{image_url}" width=400px></img>'                            
                        # html_imgpart = html_imgpart + '</div><br>'
                                               
                        output_html = html_typepart + html_modelpart + html_versionpart + html_creatorpart + html_trainingpart + "<br>" +  html_modelurlpart + html_dnurlpart + "<br>" + html_descpart + "<br><div align=center>" + html_imgpart + "</div>"
                        break
       
        return gr.HTML.update(value=output_html), gr.Textbox.update(value=output_training), gr.Dropdown.update(choices=[PLACEHOLDER] + [v for v in files_name], value=PLACEHOLDER), gr.Textbox.update(value=None)
    else:
        return gr.HTML.update(value=None), gr.Textbox.update(value=None), gr.Dropdown.update(choices=[], value=None), gr.Textbox.update(value=None)

from io import BytesIO

def get_model_versionId(model_name=None, model_version_name=None):
    global json_data
    model_versionid = None
    for model in json_data['items']:
        if model['name'] == model_name:
            for model_version in model['modelVersions']:
                if model_version['name'] == model_version_name:
                    model_versionid = model_version['id']
    return model_versionid

def get_model_version_images(model_name=None, model_version_name=None):
    global json_data
    for model in json_data['items']:
        if model['name'] == model_name:
            for model_version in model['modelVersions']:
                if model_version['name'] == model_version_name:
                    return model_version['images']
    return None


# get image with full size
# width is in number, not string
# 파일 인포가 있는 원본 이미지 주소이다.
def get_full_size_image_url(image_url, width):
    return re.sub('/width=\d+/', '/width=' + str(width) + '/', image_url)

  
def update_model_versionid_gallery(model_name=None, model_version_name=None):
    
    version_images = []
    version_images_url = []
    
    if model_name and model_version_name and model_name != PLACEHOLDER and model_version_name != PLACEHOLDER:
        
        ver_id = get_model_versionId(model_name, model_version_name)
        request_url = url_dict['modelVersionId'] + str(ver_id)
        
        # use this versionId to get model info from civitai
        r = requests.get(request_url)        
        if not r.ok:
            printD("Get errorcode: " + str(r.status_code))
            return None,None,None
        
        # try to get content
        content = None
        try:
            content = r.json()
        except Exception as e:
            printD(e.text)
            return
        
        if content["images"]:
            for pic in content["images"]:                
                if "url" in pic:
                    img_url = pic["url"]
                    # use max width
                    # 파일 인포가 있는 원본 이미지 주소이다.
                    if "width" in pic:
                        if pic["width"]:
                            img_url = get_full_size_image_url(img_url, pic["width"])                                            
                                            
                    version_images_url.append(img_url)     
                    # try:
                    #     c = requests.get(img_url,stream=True)    
                    #     c.raw.decode_content=True
                    #     image = Image.open(c.raw)
                    #     version_images.append(image)                                                            
                    # except Exception as e:
                    #     print(str(e))                                                                                                

    #return version_images,version_images_url    
    return version_images_url,version_images_url    
    
def save_image_files(model_name, model_version_name, use_new_folder, lora_an):

    if model_name is None or model_name == PLACEHOLDER or model_version_name is None or model_version_name == PLACEHOLDER:
        return 
            
    model_folder = make_new_folder(current_model_type, use_new_folder, model_name, lora_an)
    base = os.path.join(root_path, model_folder, (model_name + "_" + model_version_name).replace("*", "_").replace("?", "_").replace("\"", "_").replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_"))
        
    if base and len(base.strip()) > 0:
        ver_id = get_model_versionId(model_name, model_version_name)
        request_url = url_dict['modelVersionId'] + str(ver_id)
        
        # use this versionId to get model info from civitai
        r = requests.get(request_url)        
        if not r.ok:
            printD("Get errorcode: " + str(r.status_code))
            return

        # try to get content
        content = None
        try:
            content = r.json()
        except Exception as e:
            printD(e.text)
            return
        
        if not content:
            printD("error, content from civitai is None")
            return
                                             
        image_count = 1
        if content["images"]:
            for img_dict in content["images"]:
                # if "nsfw" in img_dict:
                #     if img_dict["nsfw"]:
                #         printD("This image is NSFW")

                if "url" in img_dict:
                    img_url = img_dict["url"]
                    # use max width
                    if "width" in img_dict:
                        if img_dict["width"]:
                            img_url = get_full_size_image_url(img_url, img_dict["width"])

                    # get image
                    img_r = requests.get(img_url, stream=True)
                    if not img_r.ok:
                        printD("Get error code: " + str(r.status_code))
                        printD(r.text)
                        return

                    # write to file
                    description_img = f'{base}_{image_count}.png'
                    with open(description_img, 'wb') as f:
                        img_r.raw.decode_content = True
                        shutil.copyfileobj(img_r.raw, f)

                    # set image_counter
                    image_count = image_count + 1
    return

def save_image_file_by_index(model_name, model_version_name, use_new_folder, lora_an, img_index=0):

    if model_name is None or model_name == PLACEHOLDER or model_version_name is None or model_version_name == PLACEHOLDER:
        return 
            
    model_folder = make_new_folder(current_model_type, use_new_folder, model_name, lora_an)
    base = os.path.join(root_path, model_folder, (model_name + "_" + model_version_name).replace("*", "_").replace("?", "_").replace("\"", "_").replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_"))
        
    if base and len(base.strip()) > 0:
        ver_id = get_model_versionId(model_name, model_version_name)
        request_url = url_dict['modelVersionId'] + str(ver_id)
        
        # use this versionId to get model info from civitai
        r = requests.get(request_url)        
        if not r.ok:
            printD("Get errorcode: " + str(r.status_code))
            return

        # try to get content
        content = None
        try:
            content = r.json()
        except Exception as e:
            printD(e.text)
            return
        
        if not content:
            printD("error, content from civitai is None")
            return
                                             
        image_count = 1
        if content["images"]:
            for img_dict in content["images"]:

                if "url" in img_dict:
                    img_url = img_dict["url"]
                    # use max width
                    if "width" in img_dict:
                        if img_dict["width"]:
                            img_url = get_full_size_image_url(img_url, img_dict["width"])

                    if (img_index + 1) == image_count:
                        # get image
                        img_r = requests.get(img_url, stream=True)
                        if not img_r.ok:
                            printD("Get error code: " + str(r.status_code))
                            printD(r.text)
                            return

                        # write to file
                        description_img = f'{base}_{image_count}.png'
                        with open(description_img, 'wb') as f:
                            img_r.raw.decode_content = True
                            shutil.copyfileobj(img_r.raw, f)
                            return description_img                         

                    # set image_counter
                    image_count = image_count + 1
    return None


def save_image_file(image_url, model_name, model_version_name, use_new_folder, lora_an, img_index=0):
    if model_name is None or model_name == PLACEHOLDER or model_version_name is None or model_version_name == PLACEHOLDER:
        return None
    
    model_folder = make_new_folder(current_model_type, use_new_folder, model_name, lora_an)
    base = os.path.join(root_path, model_folder, (model_name + "_" + model_version_name).replace("*", "_").replace("?", "_").replace("\"", "_").replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_"))
    
    if base and len(base.strip()) > 0:
        # get image
        try:
            img_r = requests.get(image_url, stream=True)
        except Exception as e:
            printD("Get error code: " + str(e.status_code))
            return None
        
        # write to file
        description_img = f'{base}_{img_index}.png'
        with open(description_img, 'wb') as f:
            img_r.raw.decode_content = True
            shutil.copyfileobj(img_r.raw, f)
      
    return description_img

def show_image_info(img_index,version_images_url):  
    # print(int(img_index))
    # image = Image.new('RGB', (10,int(img_index)+10))
    # return img_index, image
    return img_index, version_images_url[int(img_index)]

def civitai_browser_ui():    
    with gr.Box():
        gr.Markdown("###  Search ")  
        with gr.Box(): 
            with gr.Column():
                with gr.Row():
                    content_type = gr.Radio(label='Content type:', choices=[k for k, v in content_types_dict.items()], value="All", type="value")                        
                    sort_type = gr.Radio(label='Sort List by:', choices=["Newest", "Most Downloaded", "Highest Rated", "Most Liked"], value="Newest", type="value")
                with gr.Row():
                    search_term = gr.Textbox(label="Search Term", placeholder="Enter your prompt", max_lines=1).style(container=False)
                    get_search_from_api = gr.Button(value="Search",variant="primary").style(full_width=False)
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Box():
                gr.Markdown("###  Models ")    
                with gr.Box():
                    with gr.Column():   
                        # test_js = gr.Button(value="java script test")
                        show_nsfw = gr.Checkbox(label="Show NSFW", value=True)
                        list_models = gr.Dropdown(label="Model", choices=[], interactive=True, value=None) 
                        with gr.Row():
                            get_prev_page = gr.Button(value="Previous Page")
                            get_next_page = gr.Button(value="Next Page")                            
                        
                        list_versions = gr.Dropdown(label="Version", choices=[], interactive=True, value=None)
                        trained_tag = gr.Textbox(label="Trained Tags",value="", interactive=False, lines=1)
                        list_filename = gr.Dropdown(label="Model Filename", choices=[], interactive=True, value=None)
                        dl_url = gr.Textbox(label="Download Url",interactive=False, value=None)

                        with gr.Row():
                            save_model_in_new = gr.Checkbox(label="Save Model to new folder", value=False)
                            an_lora = gr.Checkbox(label="Save LoRA to additional-networks", value=False)

                        save_text = gr.Button(value="Save Trained Tags as Text")
                        save_images = gr.Button(value="Save Images")
                        download_model = gr.Button(value="Download Model")
                        message_log = gr.Markdown("")
            with gr.Box():
                gr.Markdown("### Generate Info")    
                with gr.Column():
                    img_file_info = gr.Textbox(label="Generate Info", interactive=False, lines=6, show_label=False)
                    #img_file_name = gr.Textbox(value="", label="File Name", interactive=False)
                    with gr.Row():
                        try:
                            send_to_buttons = modules.generation_parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
                        except:
                            pass                    
                      

        # info_civitai_browser, civitai_browser_gallery 로 그냥 정한것 해당 익스텐션에선 이리쓸것임 
        # 그룹이름을(info) 다음에 익스텐션 표시를 (civitai_browser) 그담에 해당 컨트롤의 아이디를 그담에 표시
        with gr.Column(scale=3):
            with gr.Row(elem_id="info_civitai_browser"):
                with gr.Box():
                    gr.Markdown("###  Description ")                                   
                    with gr.Box():
                        preview_image_html = gr.HTML()        
                        preview_gallery = gr.Gallery(show_label=False, elem_id="info_civitai_browser_gallery").style(grid=[5], height="auto")                        
                    
                with gr.Row(visible=False):     
                    set_index = gr.Button('set_index', elem_id="info_civitai_browser_set_index")
                    img_index = gr.Textbox(value=-1)
                    version_images_url = gr.State([])
                    hidden = gr.Image(type="pil")
                    info1 = gr.Textbox()
                    info2 = gr.Textbox()                  
                    
            
    set_index.click(show_image_info, _js="civitai_browser_get_current_img", inputs=[img_index,version_images_url], outputs=[img_index,hidden])
    hidden.change(fn=modules.extras.run_pnginfo, inputs=[hidden], outputs=[info1, img_file_info, info2])      

    try:
        modules.generation_parameters_copypaste.bind_buttons(send_to_buttons, hidden, img_file_info)
    except:
        pass
                
    save_text.click(
        fn=save_trained_tags,
        inputs=[
            list_versions,
            save_model_in_new,
            trained_tag,
            list_models,
            an_lora,
        ],
        outputs=[message_log]
    )
    save_images.click(
        fn=save_image_files,
        inputs=[
            list_models,
            list_versions,
            save_model_in_new,
            an_lora,
        ],
        outputs=[message_log]
    )
    download_model.click(
        fn=download_file_thread,
        inputs=[
            dl_url,
            list_filename,
            save_model_in_new,
            list_models,
            an_lora,            
        ],
        outputs=[message_log]
    )
    get_search_from_api.click(
        fn=update_models_list,
        inputs=[
            content_type,
            sort_type,
            search_term,
            show_nsfw,
        ],
        outputs=[
            list_models,
            list_versions,
            preview_image_html,
            trained_tag,
            list_filename,
            dl_url,
        ]
    )
    get_next_page.click(
        fn=update_next_page,
        inputs=[
            show_nsfw,
        ],
        outputs=[
            list_models,
            list_versions,
            preview_image_html,
            trained_tag,
            list_filename,
            dl_url,
        ]
    )
    get_prev_page.click(
        fn=update_prev_page,
        inputs=[
            show_nsfw,
        ],
        outputs=[
            list_models,
            list_versions,
            preview_image_html,
            trained_tag,
            list_filename,
            dl_url,
        ]
    )    
    list_models.change(
        fn=update_model_versions,
        inputs=[
            list_models,
        ],
        outputs=[
            list_versions,
            preview_image_html,
            trained_tag,
            list_filename,
            dl_url,
        ]
    )
    list_versions.change(
        fn=update_model_version_info,
        inputs=[
            list_models,
            list_versions,
        ],
        outputs=[
            preview_image_html,
            trained_tag,
            list_filename,
            dl_url,
        ]
    )
    list_filename.change(
        fn=update_dl_url,
        inputs=[list_models, list_versions, list_filename,],
        outputs=[dl_url,],
    )
    show_nsfw.change(
        fn=update_models_list_nsfw,
        inputs=[
            show_nsfw,
        ],
        outputs=[
            list_models,
            list_versions,
            preview_image_html,
            trained_tag,
            list_filename,
            dl_url,
        ]
    )
    preview_image_html.change(
        fn=update_model_versionid_gallery,
        inputs=[
            list_models,
            list_versions,
        ],
        outputs=[
            preview_gallery,
            version_images_url
        ]
    )
        
    
def on_ui_tabs():
    with gr.Blocks() as civitai_browser:
    	civitai_browser_ui()
    
    return (civitai_browser, "CivitAi", "civitai_browser"),


script_callbacks.on_ui_tabs(on_ui_tabs)
