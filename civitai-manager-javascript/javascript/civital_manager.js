function test_alert(){
    alert('자바스크립트 테스트');
}

var civitai_manager_click_image = function(){
    console.log("civitai_manager_click_image !")
    console.log(this.classList)

    /*
    if (!this.classList.contains("transform")){        
        var gallery = civitai_manager_get_parent_by_class(this, "civitai_manager_containor");
        var buttons = gallery.querySelectorAll(".gallery-item");
        var i = 0;
        var hidden_list = [];
        buttons.forEach(function(e){
            if (e.style.display == "none"){
                hidden_list.push(i);
            }
            i += 1;
        })
        if (hidden_list.length > 0){
            setTimeout(images_history_hide_buttons, 10, hidden_list, gallery);
        }        
    } 
    */   
    civitai_manager_set_image_info(this);     
}

/*
function civitai_manager_hide_buttons(hidden_list, gallery){
    var buttons = gallery.querySelectorAll(".gallery-item");
    var num = 0;
    buttons.forEach(function(e){
        if (e.style.display == "none"){
            num += 1;
        }
    });
    if (num == hidden_list.length){
        setTimeout(civitai_manager_hide_buttons, 10, hidden_list, gallery);
    } 
    for( i in hidden_list){
        buttons[hidden_list[i]].style.display = "none";
    }    
}
*/

// 파이썬 의 함수와 리턴인자와 입력인자를 맞춰줘야 한다.
function civitai_manager_get_current_img(img_index){
    console.log("img_index",img_index)
    console.log('info_civitai_manager_set_index')

    console.log("img_index",gradioApp().getElementById('info_civitai_manager_set_index').getAttribute("img_index"))

    return [
        gradioApp().getElementById('info_civitai_manager_set_index').getAttribute("img_index")

    ];
}

function civitai_manager_set_image_info(button){
    var buttons = civitai_manager_get_parent_by_tagname(button, "DIV").querySelectorAll(".gallery-item");
    var index = -1;
    var i = 0;
    buttons.forEach(function(e){
        if(e == button){
            index = i;
        }
        if(e.style.display != "none"){
            i += 1;
        }        
    });
    
    console.log("img_index",index)

    var gallery = civitai_manager_get_parent_by_class(button, "civitai_manager_containor");
    var set_btn = gallery.querySelector(".civitai_manager_set_index");
    var curr_idx = set_btn.getAttribute("img_index", index);  
    if (curr_idx != index) {
        set_btn.setAttribute("img_index", index);        
    }    
    set_btn.click();    
}

function civitai_manager_get_parent_by_class(item, class_name){
    var parent = item.parentElement;
    while(!parent.classList.contains(class_name)){
        parent = parent.parentElement;
    }
    return parent;  
}

function civitai_manager_get_parent_by_tagname(item, tagname){
    var parent = item.parentElement;
    tagname = tagname.toUpperCase()
    while(parent.tagName != tagname){
        parent = parent.parentElement;
    }  
    return parent;
}

function civitai_manager_init(){ 
    gradioApp().getElementById('info_civitai_manager').classList.add("civitai_manager_containor");
    gradioApp().getElementById('info_civitai_manager_set_index').classList.add("civitai_manager_set_index");
    //gradioApp().getElementById(tab + '_images_history_del_button').classList.add("images_history_del_button");
    gradioApp().getElementById('info_civitai_manager_gallery').classList.add("civitai_manager_gallery");  
}

let cbtimer
var civitai_manager_tab_list = "";
setTimeout(civitai_manager_init, 500);
document.addEventListener("DOMContentLoaded", function() {
    var mutationObserver = new MutationObserver(function(m){
        var buttons = gradioApp().querySelectorAll('#info_civitai_manager .gallery-item');
            buttons.forEach(function(bnt){    
                bnt.addEventListener('click', civitai_manager_click_image, true);
                document.onkeyup = function(e){
                    clearTimeout(cbtimer)
                    cbtimer = setTimeout(() => {
                        // 아마도 tab_civitai_manager는 익스텐션의 tab을 stable_diffusion 시스템에서 정의한것 일것
                        let tab = gradioApp().getElementById("tab_civitai_manager").getElementsByClassName("bg-white px-4 pb-2 pt-1.5 rounded-t-lg border-gray-200 -mb-[2px] border-2 border-b-0")[0].innerText
                        bnt = gradioApp().getElementById(tab + "_civitai_manager_gallery").getElementsByClassName('gallery-item !flex-none !h-9 !w-9 transition-all duration-75 !ring-2 !ring-orange-500 hover:!ring-orange-500 svelte-1g9btlg')[0]
                        civitai_manager_click_image.call(bnt)
                    },500)

                }
            });
    });
    mutationObserver.observe(gradioApp(), { childList:true, subtree:true });
});