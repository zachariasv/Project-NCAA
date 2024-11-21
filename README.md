## NCAA Basketball Prediction Model
*!! This project is a work in progress !!*
A machine learning pipeline for predicting NCAA basketball outcomes using historical team performance data. Please note that this most likely is still a work in progress.

#### Features
- Data Processing & Engineering: Loads, preprocesses, and standardizes game data; calculates team scores, trends, and head-to-head stats.
- Machine Learning Models: Implements a Random Forest and Neural Network to classify game outcomes.
- PCA Analysis: Reduces dimensionality and visualizes feature importance for further feature engineering or model optimization.
- Odds collection using API calls
- Automated and manual linkage of database IDs to match odds data to prediction data

#### Setup
The project is currently split among 3 folders with different goals. *Prediction* contains my original project file which was all contained in a single notebook, and serves as the basis for the entire directory. The current work involves generalising this notebook to a more robust program that can work in real time. 

The *The Odds API* folder covers the collection of real time odds data as well as an ongoing collection of team names. The odds data will be used for the real time prediction and "bet opportunity" search, while the team names are collected for matching between databases. This is necessary as team names aren't universally agreed upon. For example, the basketball team of the University of Pennsylvania is often referred to as Penn. In these cases, I need to be able to systematically match Pennsylvania to Penn. That is the database which is being built up in that folder along with the odds collection. 

The *Prediction* folder contains the original jupyter notebook where I first created the project, and will contain the live implementation as well. For the config-file in the predictions folder, set up appropriate data paths in the example config file provided and rename to config.yaml. 

#### Contact info
Email: zacharias@veiksaar.se
