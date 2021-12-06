# Flower-Classification
Machine Learning Group 25 repository.  Flower classification for given types of flowers

# Run web-scraper
Install pip dependendencies
  1) pip install selenium
  2) pip install pillow
  3) pip install requests
Run python script
  1) python web-scraper.py {flower name}
  The program will take in the given arguement of flower name (e.g. orchid) and will create a folder in sample-images with that name and save the defined number of pictures into the folder
  2) python web-scraper delete-duplicates
  The program will delete all duplicate images in the folders in sample images. This is done by checking every single image so willt ake a while to run

If this does not work:
Download appropriate google chrome driver for your chrome version @ https://chromedriver.storage.googleapis.com/index.html and replace the chromedriver.exe in this file

Note:
If program stalls on image downlaod try change the search term (i.e. it was stalling on Lily, so i changed the term to lily-flower)

# Run KNN
Install dependencies 
  1) pip install scikit-learn
  2) pip install opencv-python
  3) python knn.py

# Run CNN
Install dependencies 
  1) pip install scikit-learn
  2) pip install opencv-python
  3) pip install tensorflow
  4) python model.py