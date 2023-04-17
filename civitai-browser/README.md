# Civitai browser
This is an extension for [web ui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
To install it, clone the repo into the `extensions` directory and restart the web ui:
설치방법 
1. 단독 설치
    git clone https://github.com/joonsun01/civitai-browser
2. civitai helper 에 추가 설치
    먼저 lib 폴더의 civitai_browser.py 파일에서 
    
    def on_ui_tabs():
        with gr.Blocks() as civitai_browser:
            with gr.Tab("Browser"):
                civitai_browser_ui()
        return (civitai_browser, "CivitAi", "civitai_browser"),
    
    script_callbacks.on_ui_tabs(on_ui_tabs)
    
    이 부분을 주석 처리한다.civitai helper 익스텐션의 civitai_helper.py 에서 
    선언부에 from scripts import civital_browser 를 추가
    ui 부분에 적당한 곳에 civital_browser.civitai_browser_ui() 를 추가 해준다.

    예)
    with gr.Tab("Browser"):
        civital_browser.civitai_browser_ui()
                        
