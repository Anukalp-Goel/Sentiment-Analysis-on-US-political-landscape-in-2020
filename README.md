
# README: Sentiment Analysis of US Election Tweets

## Overview
This project performs sentiment analysis on Twitter data related to the 2020 US Presidential Elections, with the goal of understanding public sentiment and its political implications. The project involves data cleaning, exploratory analysis, model training, and prediction using various machine learning techniques. The results provide insights into the US political landscape based on the sentiment of tweets.

## Files
- `code.ipynb`: The main Jupyter Notebook containing the implementation of the analysis.
- `sentiment_analysis.csv`: A dataset containing pre-labeled tweets with sentiment polarity.
- `US_Elections_2020.csv`: A dataset of tweets related to the 2020 US Presidential Elections, with labeled sentiment and, in some cases, reasons for negative sentiment.
  
## Project Components

### 1. Data Cleaning
The raw tweets contain unwanted noise, such as HTML tags, URLs, and stopwords. This step involves:
- Removing HTML tags and attributes.
- Converting all text to lowercase.
- Removing URLs and stopwords.
- Handling empty tweets post-cleaning.

### 2. Exploratory Analysis
A preliminary analysis was conducted to understand the data. This included:
- Categorizing tweets by political party based on relevant keywords and hashtags.
- Visualizing tweet distributions and sentiment using charts, graphs, and word clouds.
  
### 3. Model Preparation
The `sentiment_analysis.csv` dataset was split into training (70%) and test data (30%). Two sets of features were prepared for model training:
- **Bag of Words (BoW)**: Frequency of word occurrences.
- **TF-IDF**: Term Frequency-Inverse Document Frequency to represent the importance of words.

### 4. Model Implementation and Tuning
Various machine learning models were trained and tested, including:
- Logistic Regression
- k-Nearest Neighbors (k-NN)
- Naive Bayes
- Decision Trees
- Random Forest (Ensembles)
- XGBoost

Cross-validation and hyperparameter tuning were performed to optimize the models. The best-performing model was then used to predict sentiment on the `US_Elections_2020.csv` dataset.

### 5. Sentiment Prediction
The model with the highest accuracy was used to predict sentiment on the US Election tweets. The results were visualized and analyzed to determine how the public viewed each political party.

### 6. Multi-class Classification for Negative Reasons
Negative tweets were further analyzed to predict the reason for negativity. Three models were used to predict the reason for negative sentiment:
- Logistic Regression
- Naive Bayes
- XGBoost

Hyperparameter tuning was performed to achieve the best accuracy.

## Results
- Sentiment Analysis on US Elections data showed varying sentiment trends for both political parties.
- The multi-class classification on negative reasons identified common themes, though some reasons were more difficult to predict accurately.

## Future Improvements
- Exploring deep learning models for better performance.
- Applying more sophisticated feature engineering techniques like word embeddings.
  
## Instructions to Run the Project
1. Install the necessary Python libraries (e.g., `Numpy`, `Pandas`, `NLTK`, `Scikit-learn`).
2. Open `code.ipynb` in Jupyter Notebook or Google Colab.
3. Run the cells sequentially to execute the analysis.

## License
This project is for academic purposes related to the MIE 1624 course at the University of Toronto.
