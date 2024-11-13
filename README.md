## NCAA Basketball Prediction Model
A machine learning pipeline for predicting NCAA basketball outcomes using historical team performance data. Please note that this most likely is still a work in progress.

#### Features
- Data Processing & Engineering: Loads, preprocesses, and standardizes game data; calculates team scores, trends, and head-to-head stats.
- Machine Learning Models: Implements a Random Forest and Neural Network to classify game outcomes.
- PCA Analysis: Reduces dimensionality and visualizes feature importance for further feature engineering or model optimization.
- Odds collection using API calls
- Automated and manual linkage of database IDs to match odds data to prediction data

#### Setup
Currently only works on my proprietary dataset of NCAA match history. Will happily share this data if you're interested. I am currently working on making the original web scraping script that I used to collect the data public, but the code needs some work.

For the config-file in the predictions folder, set up appropriate data paths in the example config file provided and rename to config.yaml. 

#### Contact info
Email: zacharias@veiksaar.se
