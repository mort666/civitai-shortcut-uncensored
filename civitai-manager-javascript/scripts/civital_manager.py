import requests
import json
import gradio as gr

import time
import threading
import os
from tqdm import tqdm
import re
from requests.exceptions import ConnectionError
import shutil
from PIL import Image

import modules.extras

from modules import shared
from modules import script_callbacks

# this is the default root path
root_path = os.getcwd()

PLACEHOLDER = "<no select>"


content_dic = {}

current_model_type = None
current_model_id = None


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
    "LoCon": os.path.join("models","Lora"),
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
    "LyCORIS": "LoCon",
    "Hypernetwork": "Hypernetwork",
    "TextualInversion": "TextualInversion",            
    "AestheticGradient":"AestheticGradient",    
    "Controlnet" : "Controlnet", 
    "Poses":"Poses"
}

models_exts = (".bin", ".pt", ".safetensors", ".ckpt")

            
def printD(msg):
    print(f"Civitai Manager: {msg}")   
    
        
# get cusomter model path
def init_civitai_manager():

    global folders_dict

    if shared.cmd_opts.embeddings_dir:
        folders_dict["TextualInversion"] = shared.cmd_opts.embeddings_dir

    if shared.cmd_opts.hypernetwork_dir :
        folders_dict["Hypernetwork"] = shared.cmd_opts.hypernetwork_dir

    if shared.cmd_opts.ckpt_dir:
        folders_dict["Checkpoint"] = shared.cmd_opts.ckpt_dir

    if shared.cmd_opts.lora_dir:
        folders_dict["LORA"] = shared.cmd_opts.lora_dir
        folders_dict["LoCon"] = shared.cmd_opts.lora_dir


def get_model_info_by_id(id)-> dict:

    if not id:
        return

    r = requests.get(url_dict["modelId"]+str(id))
    if not r.ok:
        if r.status_code == 404:
            # this is not a civitai model
            return {}
        else:
            return

    # try to get content
    content = None
    try:
        content = r.json()
    except Exception as e:
        return
    
    if not content:
        return
    
    return content


# get id from url
def get_model_id_from_url(url):
    id = ""

    if not url:
        return ""

    if url.isnumeric():
        # is already an id
        id = str(url)
        return id
    
    s = url.split("/")
    if len(s) < 2:
        return ""
    
    if s[-2].isnumeric():
        id  = s[-2]
    elif s[-1].isnumeric():
        id  = s[-1]
    else:
        return ""
    
    return id

# get id from model name
# json 데이터 에서 가져온다.
def get_model_id_from_name(name):
    global json_data

    if not name:
        return ""

    if name is not None:        
        for model in json_data['items']:
            if model['name'] == name:
                id = model['id']    
                return id
    return None


def get_model_version_id(model_name=None, model_version_name=None):
    global json_data
    model_versionid = None
    for model in json_data['items']:
        if model['name'] == model_name:
            for model_version in model['modelVersions']:
                if model_version['name'] == model_version_name:
                    model_versionid = model_version['id']
    return model_versionid


# get image with full size
# width is in number, not string
# 파일 인포가 있는 원본 이미지 주소이다.
def get_full_size_image_url(image_url, width):
    return re.sub('/width=\d+/', '/width=' + str(width) + '/', image_url)

    
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
            
                
def download_file_thread(url, file_name, version_name, model_name, lora_an, save_tags, trained_words):
    if file_name and model_name and url:
        model_folder = make_new_folder(current_model_type, model_name, lora_an)

        try:
            if save_tags and trained_words and len(trained_words.strip()) > 0:
                path_tags_file = os.path.join(model_folder, (model_name + "." + version_name).replace("*", "_").replace("?", "_").replace("\"", "_").replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_") + ".tags.txt")
                if not os.path.exists(path_tags_file):
                    with open(path_tags_file, 'w') as f:
                        f.write(trained_words)            
        except:
            pass
        
        path_dl_file = os.path.join(model_folder, file_name)                    
        thread = threading.Thread(target=download_file,args=(url, path_dl_file))
        # Start the thread
        thread.start()
        return f"Download started"
    return ""


def make_new_folder(content_type, model_name, lora_an):
    
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
                
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
                
    return model_folder


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
        return None
    if show_nsfw:
        for model in json_data['items']:
            models_name.append(model['name'])
    else:
        for model in json_data['items']:
            temp_nsfw = model['nsfw']
            if not temp_nsfw:
                models_name.append(model['name'])
    return [v for v in models_name]


def update_next_page(show_nsfw):
    global json_data
    tmp_json_data = api_next_page()
    
    if tmp_json_data:
        json_data = tmp_json_data
            
    models_name = []
    try:
        json_data['items']
    except TypeError:
        return None
    if show_nsfw:
        for model in json_data['items']:
            models_name.append(model['name'])
    else:
        for model in json_data['items']:
            temp_nsfw = model['nsfw']
            if not temp_nsfw:
                models_name.append(model['name'])
    return [v for v in models_name]


def update_models_list_nsfw(show_nsfw):
    global json_data
    models_name = []
    try:
        json_data['items']
    except TypeError:
        return None
    if show_nsfw:
        for model in json_data['items']:
            models_name.append(model['name'])
    else:
        for model in json_data['items']:
            temp_nsfw = model['nsfw']
            if not temp_nsfw:
                models_name.append(model['name'])
    return [v for v in models_name]

    
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
    return [v for v in models_name]

    
def update_model_versions(model_name=None):
    if model_name is not None and model_name != PLACEHOLDER:
        global json_data
        versions_name = []
        for model in json_data['items']:
            if model['name'] == model_name:
                for model_version in model['modelVersions']:
                    versions_name.append(model_version['name'])
        
        return [v for v in versions_name]                    
    else:
        return None


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
                                return dl_url
    return None


def update_model_version_info_by_id(id=None, version_id=None):
    global current_model_type 
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
        
    model = ""
            
    if id and model_version_id: 

        r = requests.get(url_dict["modelId"]+str(id))
        if not r.ok:
            return None,None,None
        else:
            # try to get content
            model = None
            try:
                model = r.json()
            except Exception as e:
                return None,None,None
            
            current_model_type = model['type']
            html_typepart = f'<br><b>Type: {current_model_type}</b>'
            
            model_id = model['id']
            model_uploader = model['creator']['username']
            html_creatorpart = f"<br><b>Uploaded by:</b> {model_uploader}"
            
            model_url = url_dict["modelPage"]+str(model_id)
            html_modelpart = f'<br><b>Model: <a href="{model_url}" target="_blank">{model["name"]}</a></b>'
            html_modelurlpart = f'<br><b><a href="{model_url}" target="_blank">Civitai Hompage << Here</a></b><br>'
                                                
            if model['description']:
                html_descpart = f"<br><b>Description</b><br>{model['description']}<br>"
                
            for model_version in model['modelVersions']:
                if model_version['id'] == version_id:
                    model_version_name = model_version['name']
                    
                    if model_version['trainedWords']:
                        output_training = ", ".join(model_version['trainedWords'])
                        html_trainingpart = f'<br><b>Training Tags:</b> {output_training}'

                    for file in model_version['files']:
                        #dl_dict[file['name']] = file['downloadUrl']
                        files_name.append(file['name'])
                        html_dnurlpart = html_dnurlpart + f"<br><a href={file['downloadUrl']}><b>Download << Here</b></a>"
                    
                    # html_imgpart = "<HEAD><style>img { display: inline-block; }</style></HEAD><div class='column'>"
                    # for pic in model_version['images']:
                    #     image_url = get_full_size_image_url(pic["url"], pic["width"])
                    #     html_imgpart = html_imgpart + f'<img src="{image_url}" width=400px></img>'                            
                    # html_imgpart = html_imgpart + '</div><br>'
                    
                    html_versionpart = f"<br><b>Version:</b> {model_version_name}"
                                            
                    output_html = html_typepart + html_modelpart + html_versionpart + html_creatorpart + html_trainingpart + "<br>" +  html_modelurlpart + html_dnurlpart + "<br>" + html_descpart + "<br><div align=center>" + html_imgpart + "</div>"
                    break
                
        return output_html,output_training,[v for v in files_name]
    else:
        return None,None,None
       
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
    
    if model_name and model_version_name:       
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
                html_modelurlpart = f'<br><b><a href="{model_url}" target="_blank">Civitai Hompage << Here</a></b><br>'

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
                            html_dnurlpart = html_dnurlpart + f"<br><a href={file['downloadUrl']}><b>Download << Here</b></a>"
                        
                        # html_imgpart = "<HEAD><style>img { display: inline-block; }</style></HEAD><div class='column'>"
                        # for pic in model_version['images']:
                        #     image_url = get_full_size_image_url(pic["url"], pic["width"])
                        #     html_imgpart = html_imgpart + f'<img src="{image_url}" width=400px></img>'                            
                        # html_imgpart = html_imgpart + '</div><br>'
                                               
                        output_html = html_typepart + html_modelpart + html_versionpart + html_creatorpart + html_trainingpart + "<br>" +  html_modelurlpart + html_dnurlpart + "<br>" + html_descpart + "<br><div align=center>" + html_imgpart + "</div>"
                        break    

        return output_html,output_training,[v for v in files_name]
    else:
        return None,None,None
    
def update_model_version_gallery(model_name=None, model_version_name=None):
    
    version_images = []
    version_state_image_urls = []
    version_image_urls=[]
    
    if model_name and model_version_name:
        
        ver_id = get_model_version_id(model_name, model_version_name)
        request_url = url_dict['modelVersionId'] + str(ver_id)
                
        # try to get content
        try:
            r = requests.get(request_url)
            content = r.json()
        except Exception as e:
            return None,None
        
        if content["images"]:
            for pic in content["images"]:                
                if "url" in pic:
                    img_url = pic["url"]
                    # use max width
                    # 파일 인포가 있는 원본 이미지.
                    if "width" in pic:
                        if pic["width"]:
                            img_url = get_full_size_image_url(img_url, pic["width"])                                            
                    
                    try:
                        c = requests.get(pic["url"],stream=True)    
                        c.raw.decode_content=True
                        image = Image.open(c.raw)
                        version_images.append(image)

                        #제네레이션 정보는 원본에만 있다                        
                        version_state_image_urls.append(img_url)     
                        #로드는 작은 이미지로 한다
                        version_image_urls.append(pic["url"])

                    except Exception as e:
                        pass
        
    #return version_images,version_state_image_urls,title_name,None  
    #return version_image_urls,version_state_image_urls,title_name,None  
    return version_images,version_state_image_urls
    
    
def save_image_files(model_name, model_version_name, lora_an):

    message =""
    if model_name and model_version_name:
            
        model_folder = make_new_folder(current_model_type, model_name, lora_an)
        base = os.path.join(root_path, model_folder, (model_name + "." + model_version_name).replace("*", "_").replace("?", "_").replace("\"", "_").replace("|", "_").replace(":", "_").replace("/", "_").replace("\\", "_"))
        
        if base and len(base.strip()) > 0:
            ver_id = get_model_version_id(model_name, model_version_name)
            request_url = url_dict['modelVersionId'] + str(ver_id)
        
            # use this versionId to get model info from civitai
            try:
                r = requests.get(request_url)        
                content = r.json()
            except Exception as e:
                return "image saving failed"
        
            if not content:
                printD("error, content from civitai is None")
                return "image saving failed"
                                             
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

                        try:
                            # get image
                            img_r = requests.get(img_url, stream=True)
                            if not img_r.ok:
                                # 작은 이미지로 해보고 
                                img_r = requests.get(img_dict["url"], stream=True) 
                                # 그래도 안되면
                                if not img_r.ok:
                                    printD("Get error code: " + str(r.status_code) + ": proceed to the next file")
                                    continue

                            # write to file
                            description_img = f'{base}_{image_count}.preview.png'
                            with open(description_img, 'wb') as f:
                                img_r.raw.decode_content = True
                                shutil.copyfileobj(img_r.raw, f)
                        except Exception as e:
                            pass
                        
                        # set image_counter
                        image_count = image_count + 1
            
                if image_count > 2:        
                    message = f"Saved {image_count - 1} images"
                else:
                    message = f"Saved image"                    
    return message                    

    
###################################################################################
def show_nsfw_change(show_nsfw):
    modellist = update_models_list_nsfw(show_nsfw)
    if modellist:
        return gr.Dropdown.update(choices=[PLACEHOLDER] + modellist, value=PLACEHOLDER), gr.Dropdown.update(choices=[], value=None), gr.HTML.update(value=None), gr.Textbox.update(value=None), gr.Dropdown.update(choices=[], value=None)
    else:   
        return gr.Dropdown.update(choices=[], value=None), gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None), gr.Dropdown.update(choices=[], value=None)

def search_btn_click(content_type, sort_type, search_term, show_nsfw):
    modellist = update_models_list(content_type, sort_type, search_term, show_nsfw)
    if modellist:
        return gr.Dropdown.update(choices=[PLACEHOLDER] + modellist, value=PLACEHOLDER), gr.Dropdown.update(choices=[], value=None), gr.HTML.update(value=None), gr.Textbox.update(value=None), gr.Dropdown.update(choices=[], value=None)
    else:   
        return gr.Dropdown.update(choices=[], value=None), gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None), gr.Dropdown.update(choices=[], value=None)

def next_page_btn_click(show_nsfw):
    modellist = update_next_page(show_nsfw)
    if modellist:
        return gr.Dropdown.update(choices=[PLACEHOLDER] + modellist, value=PLACEHOLDER), gr.Dropdown.update(choices=[], value=None), gr.HTML.update(value=None), gr.Textbox.update(value=None), gr.Dropdown.update(choices=[], value=None)
    else:   
        return gr.Dropdown.update(choices=[], value=None), gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None), gr.Dropdown.update(choices=[], value=None)

def prev_page_btn_click(show_nsfw):
    modellist = update_prev_page(show_nsfw)
    if modellist:
        return gr.Dropdown.update(choices=[PLACEHOLDER] + modellist, value=PLACEHOLDER), gr.Dropdown.update(choices=[], value=None), gr.HTML.update(value=None), gr.Textbox.update(value=None), gr.Dropdown.update(choices=[], value=None)
    else:   
        return gr.Dropdown.update(choices=[], value=None), gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None), gr.Dropdown.update(choices=[], value=None)

def list_models_change(model_name):    
    versionlist = update_model_versions(model_name)
    if versionlist:
        return gr.Dropdown.update(choices=[PLACEHOLDER] + versionlist, value=PLACEHOLDER),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None)
    else:
        return gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None)

####################################################################################
def list_versions_change(model_name=None, model_version_name=None):       
    output_html=None
    output_training=None
    filelist=None
    if model_name and model_version_name and model_name != PLACEHOLDER and model_version_name != PLACEHOLDER:     
         output_html,output_training,filelist = update_model_version_info(model_name, model_version_name)    
         
    if filelist:
        return gr.HTML.update(value=output_html),gr.Textbox.update(value=output_training),gr.Dropdown.update(choices=[PLACEHOLDER] + filelist, value=PLACEHOLDER)
    else:
        return gr.HTML.update(value=None), gr.Textbox.update(value=None), gr.Dropdown.update(choices=[], value=None)

def save_images_click(model_name, model_version_name, lora_an):
    msg = None
    if model_name and model_name != PLACEHOLDER and model_version_name and model_version_name != PLACEHOLDER:
        msg = save_image_files(model_name, model_version_name, lora_an)        
    return msg

def download_model_click(url, file_name, version_name, model_name, lora_an, save_tags, trained_words):
    msg = None
    if file_name and file_name != PLACEHOLDER and model_name and model_name != PLACEHOLDER and url and len(url.strip()) > 0:
        msg = download_file_thread(url, file_name, version_name, model_name, lora_an, save_tags, trained_words)
    return msg

def preview_image_html_change(model_name,model_version_name):
    title_name = f"### {model_name} : {model_version_name}"
    if model_name and model_version_name and model_name != PLACEHOLDER and model_version_name != PLACEHOLDER:
        version_dip,version_images_url = update_model_version_gallery(model_name, model_version_name)
        return version_dip,version_images_url,title_name,None
    else:
        return None,None,title_name,None

def list_filename_change(model_name, model_version_name, model_filename):
    dl_url = update_dl_url(model_name, model_version_name, model_filename)
    return gr.Textbox.update(value=dl_url)

def show_image_info(img_index,version_images_url):  
    # print(int(img_index))
    # image = Image.new('RGB', (10,int(img_index)+10))
    # return img_index, image
    return img_index, version_images_url[int(img_index)]

def get_civitai_model_info_click(url:str):    
    return gr.Dropdown.update(choices=[], value=None),gr.Dropdown.update(choices=[], value=None),gr.HTML.update(value=None), gr.Textbox.update(value=None),gr.Dropdown.update(choices=[], value=None)
#######################################################################################    
    
def civitai_manager_ui():
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Box():            
                gr.Markdown("###  Search")    
                with gr.Box(): 
                    with gr.Column():
                        show_nsfw = gr.Checkbox(label="Show NSFW", value=True)
                        with gr.Row():
                            content_type = gr.Dropdown(label='Content type:', choices=[k for k in content_types_dict], value="All", type="value")                                                        
                            sort_type = gr.Dropdown(label='Sort List by:', choices=["Newest", "Most Downloaded", "Highest Rated", "Most Liked"], value="Newest", type="value")                        
                        search_term = gr.Textbox(label="Search Term", placeholder="Enter your prompt", max_lines=1).style(container=False)
                        search_btn = gr.Button(value="Search",variant="primary").style(full_width=False)
                        with gr.Row():       
                            prev_page_btn = gr.Button(value="Prev Page")
                            next_page_btn = gr.Button(value="Next Page") 
                        list_models = gr.Dropdown(label="Model Name", choices=[], interactive=True, value=None)                                                                            
                                                                                             
        with gr.Column(scale=4):
            with gr.Box():
                with gr.Row():    
                    with gr.Column(scale=4):                    
                        civitai_model_url_txt = gr.Textbox(label="Model Url", placeholder="Enter your civitai url", max_lines=1).style(container=False)
                    with gr.Column(scale=1):                    
                        civitai_model_info_btn = gr.Button(value="Get Model Info",variant="primary").style(full_width=False)                                                
                
            with gr.Box():                                                                                                                           
                # info_civitai_manager, civitai_manager_gallery 로 그냥 정한것 해당 익스텐션에선 이리쓸것임 
                # 그룹이름을(info) 다음에 익스텐션 표시를 (civitai_manager) 그담에 해당 컨트롤의 아이디를 그담에 표시                    
                with gr.Row():                        
                    with gr.Column(scale=1):
                        model_title_name = gr.Markdown(visible=True)                                                                                                                                         
                        list_versions = gr.Dropdown(label="Model Version", choices=[], interactive=True, value=None)
                        trained_tag = gr.Textbox(label="Trained Tags",value="", interactive=False, lines=1)                               
                        list_filename = gr.Dropdown(label="Model Version File", choices=[], interactive=True, value=None)                            
                        
                        
                        #save_text = gr.Button(value="Save Trained Tags as Text")                            
                        save_tags = gr.Checkbox(label="Save Trained Tags", value=True)
                        download_model = gr.Button(value="Download",variant="primary")                             
                        an_lora = gr.Checkbox(label="Save LoRA to additional-networks", value=False)                            
                                                                                            
                    with gr.Column(elem_id="info_civitai_manager", scale=4):                                                     
                        with gr.Box():                  
                            preview_gallery = gr.Gallery(show_label=False, elem_id="info_civitai_manager_gallery").style(grid=[4], height="auto")   
                            message_log = gr.Markdown("###")
                            preview_image_html = gr.HTML()   
                                                                    
                        with gr.Row(visible=False):                                         
                            set_index = gr.Button('set_index', elem_id="info_civitai_manager_set_index")
                            img_index = gr.Textbox(value=-1)
                            version_images_url = gr.State([])
                            hidden = gr.Image(type="pil")
                            info1 = gr.Textbox()
                            info2 = gr.Textbox()        
                            dl_url = gr.Textbox(value=None) 

                    with gr.Column(scale=1):
                            save_images = gr.Button(value="Save Images",variant="primary")
                            img_file_info = gr.Textbox(label="Generate Info", interactive=False, lines=6)
                            #img_file_name = gr.Textbox(value="", label="File Name", interactive=False)   
                            with gr.Row():
                                try:
                                    send_to_buttons = modules.generation_parameters_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
                                except:
                                    pass     
                                                                            

     
                                 
    set_index.click(show_image_info, _js="civitai_manager_get_current_img", inputs=[img_index,version_images_url], outputs=[img_index,hidden])
    hidden.change(fn=modules.extras.run_pnginfo, inputs=[hidden], outputs=[info1, img_file_info, info2])      

    try:
        modules.generation_parameters_copypaste.bind_buttons(send_to_buttons, hidden, img_file_info)
    except:
        pass

    show_nsfw.change(
        fn=show_nsfw_change,
        inputs=[
            show_nsfw,
        ],
        outputs=[
            list_models,
            list_versions,
            preview_image_html,
            trained_tag,
            list_filename,
        ]
    )                    
    search_btn.click(
        fn=search_btn_click,
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
        ]
    )
    next_page_btn.click(
        fn=next_page_btn_click,
        inputs=[
            show_nsfw,
        ],
        outputs=[
            list_models,
            list_versions,
            preview_image_html,
            trained_tag,
            list_filename,
        ]
    )
    prev_page_btn.click(
        fn=prev_page_btn_click,
        inputs=[
            show_nsfw,
        ],
        outputs=[
            list_models,
            list_versions,
            preview_image_html,
            trained_tag,
            list_filename,
        ]
    )        
    list_models.change(
        fn=list_models_change,
        inputs=[
            list_models,
        ],
        outputs=[
            list_versions,
            preview_image_html,
            trained_tag,
            list_filename,
        ]
    )
    
    # model의 정보 표시
    list_versions.change(
        fn=list_versions_change,
        inputs=[
            list_models,
            list_versions,
        ],
        outputs=[
            preview_image_html,
            trained_tag,
            list_filename,
        ]
    )               
    list_filename.change(
        fn=list_filename_change,
        inputs=[list_models, list_versions, list_filename,],
        outputs=[
            dl_url
        ]
    )
    preview_image_html.change(
        fn=preview_image_html_change,
        inputs=[
            list_models,
            list_versions,
        ],
        outputs=[
            preview_gallery,
            version_images_url,
            model_title_name,
            message_log
        ]
    )
    
    # 다운로드
    save_images.click(
        fn=save_images_click,
        inputs=[
            list_models,
            list_versions,
            an_lora,
        ],
        outputs=[message_log]
    )
    download_model.click(
        fn=download_model_click,
        inputs=[
            dl_url,
            list_filename,            
            list_versions,
            list_models,
            an_lora,            
            save_tags,
            trained_tag,
        ],
        outputs=[message_log]
    )     
    civitai_model_info_btn.click(
        fn=get_civitai_model_info_click,
        inputs=[
            civitai_model_url_txt,
        ],
        outputs=[
            list_models,
            list_versions,
            preview_image_html,
            trained_tag,
            list_filename,
        ]                
    )
   
