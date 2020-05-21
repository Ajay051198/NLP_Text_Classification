#  NLP - Text Classification

This repo contains the files for an implementation of a depth wise separable convolutional neural network.\
Finally implemented a simple UI in tkinter to perform classifications

The structure of the project follows the guide from google about Text Classification which can be found at: <https://developers.google.com/machine-learning/practica/image-classification> 

---

### To use the application

- clone the repo: 

  ```python
  git clone https://github.com/Ajay051198/NLP_Text_Classification.git
  ```

- Follow instructions in  "instructions for data.txt" to experiment with training and tuning different models. 

- Run generate_h5.py to build and train the model.

  ```
  python generate_h5.py
  ```

- to use application (trained on the Twitter tweet vs weather data from Kaggle) run the following code from command line. (The UI takes time to initiate as it loads the model and the weight with the help of tensorflow 2 API)

  ```python
  python app.py
  ```

- Feel free to try out your entries of tweets !!

---

### Output examples: 

<img src='images\Capture1.JPG' width ="600">

<img src='images\Capture2.JPG' width ="600">

---

***A project by Ajaykumar Mudaliar***





