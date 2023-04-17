
# Civitai Shortcut

Stable Diffusion Webui Extension for Civitai, to download civitai shortcut and models.

# Install

Stable Diffusion Webui's Extension tab, go to Install from url sub-tab. Copy this project's url into it, click install.

    git clone https://github.com/sunnyark/civitai-shortcut

Stable Diffusion Webui의 Extension 탭에서 'URL에서 설치' 서브 탭으로 이동하세요. 해당 프로젝트의 URL을 복사하여 붙여넣고 '설치'를 클릭하세요.

    git clone https://github.com/sunnyark/civitai-shortcut


# Usage instructions

![screenshot 2023-04-16 182259](https://user-images.githubusercontent.com/40237431/232289602-0b0cab3d-e90c-49c1-95cc-decf3cbe40bf.png)
1. The information in the Civitai model information tab is obtained in real-time from the Civitai website.
   Download : downloads the model for the selected version. You can choose to create specific folders for each version. The downloaded model will be automatically saved in the appropriate location, and a preview image and info will be generated together with it.


![screenshot 2023-04-16 182316](https://user-images.githubusercontent.com/40237431/232289625-48abfaab-8eb4-4af0-a58e-1a9716751302.png)
2. The information in the Saved model information tab are composed of the information saved on the Civitai website when creating the shortcut.
Update Model Information : updates the information of an individual shortcut to the latest information. This function only works when the site is functioning normally.
Delete Shortcut : deletes the information of a registered shortcut.


![screenshot 2023-04-16 182333](https://user-images.githubusercontent.com/40237431/232289652-89ec3225-1849-44f8-8387-6216b576574e.png)
3. The information in the Downloaded tab of the Civitai model information is obtained from the downloaded folder when the model is downloaded.
Goto civitai shortcut tab : switches to #1 screen to view new information.
Open Download Folder : opens the file explorer to the folder where the model was downloaded.

For #1, since the information is obtained in real-time from the website, it can only be accessed when the site is operational. 
For #2 and #3, the information is displayed based on the saved information, so it can be accessed even when the site is not operational.


![screenshot 2023-04-16 182342](https://user-images.githubusercontent.com/40237431/232289694-e9f75198-75b4-492a-8f60-6fc453137f5c.png)
* Upload : This function creates a shortcut that can be used by the extension when you enter the Civitai site's model URL. It only works when the site is functioning properly. You can either click and drag the URL from the address bar or drag and drop saved internet shortcuts. You can also select multiple internet shortcuts and drop them at once.
* Update Shortcut's Model Information : This function updates the information of the shortcut you have received by referencing the site.
* Scan Downloaded Models to Shortcut : This function searches for downloaded models and registers shortcuts for them, in case you have accidentally deleted the shortcut or downloaded the models to a different location. Only models with version info files are registered. You can use Civitai Helper's Scan function to generate version info for models that don't have it. Note that only models available on the Civitai site can be registered with this function. This function only works when the site is functioning properly.

![screenshot 2023-04-16 182349](https://user-images.githubusercontent.com/40237431/232289696-54479aec-4013-42f3-9ae8-9c1f203095f2.png)
* Browsing : This function displays the registered shortcuts in thumbnail format, and when selected, displays their details on the right-hand side of the window. This function works independently of the Civitai site.

![screenshot 2023-04-16 182354](https://user-images.githubusercontent.com/40237431/232289699-c8b87b7a-85f7-436a-91e0-0c10620599c3.png)
* Scan New Version : This is a function that searches for the latest version of downloaded models on the Civitai site. It retrieves information from the site and only functions properly when the site is operational.

# Features

You can save the model URL of the Civitai site for future reference and storage.
This allows you to download the model when needed and check if the model has been updated to the latest version.
The downloaded models are saved to the designated storage location.

Civitai 사이트의 모델 URL을 저장하여 나중에 참조하고 보관할 수 있습니다.
이를 통해 필요할 때 모델을 다운로드하고 모델이 최신 버전으로 업데이트되었는지 확인할 수 있습니다.
다운로드 된 모델은 지정된 저장 위치에 저장됩니다.

# Notice

When using Civitai Shortcut, four items will be created:

* sc_saves: a folder where registered model URLs are backed up and stored.
* sc_thumb_images: a folder where thumbnails of registered URLs are stored.
* sc_infos: a folder where model info and images are saved when registering a shortcut.
* CivitaiShortCut.json: a JSON file that records and manages registered model URLs.

세개의 폴더와 하나의 json 파일이 생성되며 각각의 역할은 다음과 같습니다.

* sc_saves : 등록된 model url이 백업되는 폴더, 등록된 모든 url이 저장되는 폴더
* sc_thumb_images : 등록된 url의 thumbnail이 저장되는 폴더
* sc_infos : 등록시 모델정보와 이미지가 저장되는 폴더
* CivitaiShortCut.json : 등록된 model url 을 기록 관리 하는  json 파일

# Change Log

v 1.1c

* Measures have been taken to alleviate bottleneck issues during information loading.
* The search function now includes a #tag search feature.
  Search terms are separated by commas (,) and are connected with an "or" operation within the search terms and within the tags. There is an "and" operation between the search terms and tags.
* The shortcut storage table has been changed to add the #tag search function.
  Existing shortcuts require an "update shortcut model information" for tag searches.
* information 로딩에 병목현상을 일부 완화
* Search 에 #tags 검색 기능 추가
  "," 구분되며 search keywords 끼리는 or, tags 끼리도 or , search keyword와 tags는 and 이 적용됩니다.
* Tags검색 기능 추가를 위해 shortcut 저장 테이블이 변경되었습니다.
  기존의 shortcut은 tag 검색을 위해 Update Shortcut's Model Information 이 필요합니다.

v 1.1

* When registering a shortcut, model information and images are saved in a separate folder.
* This allows users to access model information from "Saved Model Information" Tab even if there is no connection to the Civitai site.
* "Thumbnail Update" button is removed and replaced with an "Update Shortcut's Model Information" button to keep the model information and images up to date.
* "Download images Only" button is removed from "Civitai Model Information" Tab that can be accessed live, and "Delete shortcut" button is moved to "Saved Model Information" Tab.
* "Delete shortcut" button removes the model information and images stored in sc_infos in one go.
* "Update Model Information" button is added to "Saved Model Information" Tab for individual updating of model information, in addition to "Update Shortcut's Model Information" that updates all model information.
* Shortcut 등록 시 모델정보와 이미지가 별도의 폴더에 저장됩니다.
* 이를 통해 사용자는 Civitai 사이트에 연결되지 않은 경우에도Saved Model Information 탭에서 모델 정보에 액세스할 수 있습니다.
* 모델 정보와 이미지를 최신 상태로 유지하기 위해 "Thumbnail Update" 버튼이 제거되고 "Update Shortcut's Model Information" 버튼으로 대체됩니다.
* 섬네일 업데이트 버튼을 삭제하고 shortcut 의 모델 정보와 이미지를 최신 정보로 유지하기위한 Update Shortcut's Model Information 버튼을 추가
* 라이브로 접속 가능한 "Civitai Model Information" 탭에서 Download images Only 버튼이 제거되고, Delete shortcut 버튼이 Saved Model Information 탭으로 이동됩니다.
* Delete shortcut는 sc_infos에 저장된 모델 정보와 이미지를 한 번에 제거합니다.
  *전체 모델 정보를 업데이트하는 "Update Shortcut's Model Information" 버튼 외에 개별적으로 모델 정보를 업데이트할 수 있는 "Update Model Information" 버튼이 "Saved Model Information" 탭에 추가됩니다.

# Screenshot

![screenshot 2023-04-13 200422](https://user-images.githubusercontent.com/40237431/231810541-c91471e5-e7ae-4d3c-a825-2bfed6746b73.png)

![screenshot 2023-04-13 200432](https://user-images.githubusercontent.com/40237431/231810585-63f6bffd-defa-4582-a7da-750dae29f589.png)

![screenshot 2023-04-12 101437-1](https://user-images.githubusercontent.com/40237431/231810628-c962429a-5b0b-46a9-9cb4-fe52a9e4d998.png)

![screenshot 2023-04-12 152645-1](https://user-images.githubusercontent.com/40237431/231810678-19876694-d023-4f62-960d-9ce774cccf67.png)