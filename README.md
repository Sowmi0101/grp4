# grp4
# Automated News Classification System

## Project Overview
[cite_start]This project addresses the challenge of navigating the massive volume of news available online by developing a system that categorizes news articles into four distinct topics: **World, Sports, Business, and Science/Tech**[cite: 143, 179, 222]. [cite_start]By leveraging **Natural Language Processing (NLP)** and Machine Learning, the system provides an efficient way for users to obtain essential information quickly and effectively[cite: 179, 183].

## Technical Architecture & Workflow
[cite_start]The system follows a comprehensive NLP pipeline designed for high classification accuracy and efficiency[cite: 190, 226].

### 1. Data Preprocessing
[cite_start]Raw news text from the **AG News Classification Dataset** is cleaned and structured using the **NLTK** library[cite: 221, 224, 239].
* [cite_start]**Tokenization**: Breaking down text into individual units like words and symbols[cite: 205, 227].
* [cite_start]**Text Cleaning**: Lowercasing, punctuation removal, and digit removal[cite: 227].
* [cite_start]**Noise Reduction**: Removing stop words and extranious characters like URLs[cite: 201, 204, 227].
* [cite_start]**Lemmatization**: Reducing words to their base or dictionary form for consistent analysis[cite: 227].

### 2. Feature Extraction
* [cite_start]**TF-IDF Vectorization**: Preprocessed text is converted into high-value numerical feature vectors using `TfidfVectorizer`[cite: 228, 268].
* [cite_start]**Optimization**: Parameters like `max_features=10000` and `min_df=6` were selected to ensure the most informative features were extracted while maintaining model scalability[cite: 242, 272].

### 3. Machine Learning Modeling
[cite_start]Four distinct classifiers were trained and evaluated to identify the most effective architecture for this task[cite: 231, 235].
* **Multinomial Naive Bayes (MNB)**
* **Stochastic Gradient Descent (SGD)**
* **Decision Tree Classifier**
* **Gaussian Naive Bayes (GNB)**


## Key Results & Performance
[cite_start]The models were evaluated based on **F1-Score**, **Accuracy**, and **Confusion Matrices**[cite: 234, 244].
* [cite_start]**Top Performers**: **Multinomial Naive Bayes** and **Stochastic Gradient Descent** achieved the highest accuracy, with MNB reaching approximately **88.15%**[cite: 245, 249].
* [cite_start]**Error Analysis**: Confusion matrices were used to visualize misclassifications, particularly between closely related topics like Business and Science/Tech.

## Technologies Used
* **Languages**: Python
* **Libraries**: NLTK, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
* [cite_start]**Dataset**: [AG News Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) [cite: 224]

## Professional Challenges & Resolutions
* [cite_start]**Emoticon Handling**: Addressed issues with stop-word removal when emoticons were present in the news text[cite: 265, 266].
* [cite_start]**Overfitting**: Troubleshot and resolved overfitting issues specifically encountered during Decision Tree classifier training[cite: 278].

---
[cite_start]*Developed as part of the Master's program at the University of North Texas*[cite: 146, 154, 162].
