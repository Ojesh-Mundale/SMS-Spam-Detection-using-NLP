# Spam Classification Application  

## Overview  

The Spam Classification Application is a machine learning-based project that classifies messages as spam or not spam (ham) using a Naive Bayes algorithm. It provides a simple graphical user interface (GUI) built with Tkinter, allowing users to input text and receive instant feedback on whether the message is spam or not. The application employs a pre-trained Naive Bayes model based on a dataset of spam and ham messages.  

## Table of Contents  

- [Features](#features)  
- [Technologies Used](#technologies-used)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Training](#model-training)  
- [Contributions](#contributions)  
- [License](#license)  

## Features  

- Classifies messages as spam or ham.  
- Uses a pre-trained machine learning model.  
- Provides real-time predictions through a GUI.  
- Text-to-speech functionality to announce the prediction results.  

## Technologies Used  

- Python  
- pandas  
- NumPy  
- scikit-learn  
- Tkinter  
- Pickle (for model serialization)  
- Win32com (for text-to-speech)  

## Installation  

1. Clone the repository:  
    ```bash  
    git clone https://github.com/yourusername/spam-classification-app.git  
    cd spam-classification-app  
    ```  

2. Install the required libraries (you can create a virtual environment if you prefer):  
    ```bash  
    pip install pandas numpy scikit-learn tk pywin32  
    ```  

3. Ensure that the `spam.csv` dataset file is placed in the same directory as the script.  

## Usage  

1. Run the application:  
    ```bash  
    python spam_classification_app.py  
    ```  

2. A window will open with the title "Email Spam Classification Application."   

3. Enter your message in the input box and click the "Click" button.   

4. The application will announce and display whether the message is classified as spam or not.  

## Model Training  

If you would like to retrain the model, follow these steps:  

1. Prepare the "spam.csv" dataset, ensuring it contains two columns: `class` (spam/ham) and `message` (the text).  

2. Run the following code snippet to preprocess the data and train the model:  
    ```python  
    import pandas as pd  
    from sklearn.feature_extraction.text import CountVectorizer  
    from sklearn.model_selection import train_test_split  
    from sklearn.naive_bayes import MultinomialNB  
    import pickle  

    # Load data  
    data = pd.read_csv("spam.csv", encoding="latin-1")  
    data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)  
    data['class'] = data['class'].map({'ham': 0, 'spam': 1})  

    # Vectorization  
    cv = CountVectorizer()  
    X = cv.fit_transform(data['message'])  
    y = data['class']  
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

    # Model training  
    model = MultinomialNB()  
    model.fit(x_train, y_train)  

    # Save the model and vectorizer  
    pickle.dump(model, open('spam.pkl','wb'))  
    pickle.dump(cv, open('vectorizer.pkl', 'wb'))  
    ```  

## Contributions  

Contributions are welcome! Please feel free to submit pull requests or open issues.  

## License  

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.  
