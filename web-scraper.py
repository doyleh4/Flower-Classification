#followed this tutorial https://medium.com/geekculture/scraping-images-using-selenium-f35fab26b122

import platform
import sys
import time
from selenium import webdriver
import os.path
from os import path
import requests 
from PIL import Image
from PIL import ImageChops
from io import BytesIO

NUM_IMAGES = 3000       # allow extra images for when dimensions are not met
SLEEP_BETWEEN_INTERACTIONS = 0.2 #allow sufficent time for div tree to be opened
MIN_WIDTH = 300
MIN_HEIGHT = 300

def fetch_urls(name, driver):
    def scroll_to_end(driver):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SLEEP_BETWEEN_INTERACTIONS)
    """
    Fetch the urls of images given the input arguement of image name
    """
    urls = []

    #build the query search string
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    #open broswer to query
    driver.get(search_url.format(q=name))
     

    image_count = 0
    results_start = 0

    while(image_count < NUM_IMAGES):
        # get all image thumbnail results
        thumbnail_results = driver.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        for e in thumbnail_results:
            # click every thumbnail such that we can get the real image behind it
            try:
                e.click()
                
                time.sleep(SLEEP_BETWEEN_INTERACTIONS)
                element = driver.find_elements_by_class_name('v4dQwb')
                # Google image web site logic
                if image_count == 0:
                    big_img = element[0].find_element_by_class_name('n3VNCb')
                else:
                    big_img = element[1].find_element_by_class_name('n3VNCb')

                #if image url will work with  requests append url to urls
                if big_img.get_attribute('src') and 'http' in big_img.get_attribute('src') :
                    urls.append(big_img.get_attribute("src"))
                    image_count+=1
                    print("Images obtained: "+ str(image_count))
                    if image_count == NUM_IMAGES:
                        return urls
                
            except Exception as e:
                print(e)    
        #scroll to following page    
        scroll_to_end(driver)
    return urls

def download_urls(name, urls):
    print("Downloading images")
    working_dir = os.getcwd()
    print(working_dir)

    #if folder does not exist create
    if path.exists("sample-images/" + name) == False:
        print("Folder not found, creating one in sample-images")
        os.mkdir("sample-images/" + name)

    
    #set working directory (where our images will be saved to)
    os.chdir("sample-images/" + name)
    
    counter = 0
    #Download images to directory
    for url in urls:
        image = requests.get(url).content
        print("Next image")
        try:   #try and open the image with PIL
            #open image before downloading
            temp = Image.open(BytesIO(image))
            width, height = temp.size
            temp.close()
            #if size fits requirements
            if width >= MIN_WIDTH and height >= MIN_HEIGHT:
                print("downloading: " + url)
                try:
                    f = open(str(counter)+".jpg", 'wb')
                    f.write(image)
                except Exception:
                    print("failed to download")
                    f.close()
                    continue
                f.close()
            else:
                print("Image url: " + url + " does not fit constarints")
                continue
            counter+=1
        except Exception:
            print("image url : " + url + "could not be opened, did not download")
            f.close()
def delete_duplicates():
    os.chdir("sample-images")
    for folder in os.listdir(os.getcwd()):
        os.chdir(folder)
        print("Num items in " + folder + " folder: " + str(len(os.listdir(os.getcwd()))))
        for image_name in os.listdir(os.getcwd()):
            #Not efficient atall
            image = Image.open(image_name)
            for temp_name in os.listdir(os.getcwd()):
                if temp_name == image_name:
                    continue
                temp = Image.open(temp_name)
                diff = ImageChops.difference(image.convert("RGB"), temp.convert("RGB"))
                if diff.getbbox() is None:
                    print("deleting image")
                    os.remove(image_name)
                    break
            print("Current folder: " + folder + " checking next image: " + image_name)
        print("checking next folder")  
        print("duplicates deleted ")
        print("Num remainng items: " + str(len(os.listdir(os.getcwd()))))
        os.chdir("..") #go back to parent directory

if __name__ == '__main__':
    if platform.system() == "Windows":
        print("Running webscraper for Windows")
        
        webdriver = webdriver.Chrome()
        urls = fetch_urls(sys.argv[1], webdriver)
        webdriver.close()
        download_urls(sys.argv[1], urls)
        #delete_duplicates()
    else:
        print("Error running webscraper, please use a Windows machine")

