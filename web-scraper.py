#followed this tutorial https://medium.com/geekculture/scraping-images-using-selenium-f35fab26b122

import platform
import sys
import time
from selenium import webdriver
import os.path
from os import path
import requests

NUM_IMAGES = 10
SLEEP_BETWEEN_INTERACTIONS = 0

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

    driver.get(search_url.format(q=name))
     

    image_count = 0
    results_start = 0

    while(image_count < NUM_IMAGES):
        # get all image thumbnail results
        thumbnail_results = driver.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)

        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")

        for img in thumbnail_results:
            # click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                
                #Extract image url (only downloadable images)
                if img.get_attribute('src') and 'http' in img.get_attribute('src'):
                    urls.append(img.get_attribute('src'))
                    image_count+=1
                    if image_count == NUM_IMAGES-1:
                        return urls
                
                time.sleep(SLEEP_BETWEEN_INTERACTIONS)
            except Exception:
                continue        
        scroll_to_end(driver)
        print("Urls found for " + str(NUM_IMAGES) + "images.")
    return urls

def download_urls(name, urls):
    print("Downloading images")
    working_dir = os.getcwd()
    print(working_dir)

    #if folder does not exist create
    if path.exists("sample-images/" + name) == False:
        print("Folder not found, creating one in sample-images")
        os.mkdir("sample-images/" + name)
        os.chdir("sample-images/" + name)
    
    counter = 0
    #Download images
    for url in urls:
        image = requests.get(url).content
        f = open(str(counter)+".jpg", 'wb')
        f.write(image)
        f.close()
        counter+=1

if __name__ == '__main__':
    if platform.system() == "Windows":
        print("Running webscraper for Windows")
        
        webdriver = webdriver.Chrome()
        urls = fetch_urls(sys.argv[1], webdriver)
        webdriver.close()
        download_urls(sys.argv[1], urls)
    else:
        print("Error running webscraper, please use a Windows machine")

