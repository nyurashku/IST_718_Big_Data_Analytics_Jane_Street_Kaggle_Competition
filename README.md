# Jane Street Real-Time Market Data Forecasting

IST 718 class assignment which focuses on using big data analytics tools to tackle Wall Street challenges.

![image](https://github.com/user-attachments/assets/8d6a5fbe-2a41-4793-982e-49d79300eddd)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Data & Sources](#data--sources)
- [Exploratory Data Analysis](#exploratory--data--analysis)
- [Models & Methodology](#models--methodology)
- [Results & Analysis](#results--analysis)
- [Setup & Installation](#setup--installation)
- [Authors](#authors)

---

## Project Overview

Financial markets are notoriously complex and fiercely competitive. According to economic theory, any profit opportunity created by arbitrage is likely to be discovered quickly and eroded to zero as more market participants rush in to exploit it. In an era of unprecedented access to information, it can be argued that finding true alpha—an investing “edge”—has become nearly impossible without industrial-scale data mining and sophisticated predictive models. Wall Street quant shops such as Jane Street, Millennium, and Citadel dominate the CTA space by employing top PhDs and math Olympiad winners to sift through massive datasets in search of signals that yield elusive market advantages. In this project, despite not having access to the same level of cutting-edge infrastructure, we aim to develop a machine learning model to forecast responder_6, predicting its behavior up to six months into the future. Our goal is to demonstrate that even with more modest resources, thoughtful methodology and careful modeling can still provide valuable insights.
The dataset comes from a Kaggle competition hosted by Jane Street Group, a global proprietary trading firm. It consists of time-series financial data with 92 anonymized columns representing features and responders. The dataset has 47,127,338 rows. One of the biggest challenges of this project is the anonymization of responder_6, which forces our team to rely solely on data-driven insights. The actual Kaggle competition is also ongoing. The highest R² score on the leaderboard is only 0.012932. This low of an R² score demonstrates how difficult it is to find meaningful predictive patterns in financial data. This also reflects the reality of quantitative trading, where the smallest predictive advantage can lead to an edge over the competition.
Our primary project goal is to identify patterns within the Jane Street dataset, identify the key features influencing responder_6, and forecast responder_6 up to six months into the future. 


---

## Data & Sources

The data used for this project is from a Jane Street hosted Kaggle competition which can be found here: The data used [Kaggle: Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting).

All data resources have been downloaded and added to this github too in case the data will not longer be available after the competition ends. 

---
## Exploratory Data Analysis

For data cleaning and exploration, a 5% randomly sampled dataset was extracted from the full dataset of over 47 million rows. This was done to reduce the computation time required for cleaning, exploration, and modeling. This methodology assumes the randomly sampled data is representative of the full dataset. The original dataset contains 9 responders including the target, responder_6. All responders except for the target were dropped from the sampled dataset. Next, the data was checked for categorical data and missing data. If the missing data was more than 10% of the individual feature, the missing rows were removed. If the missing data was less than 10% of the feature, the missing values were imputed with the median value. The low-variance features were then removed. After cleaning was completed, data exploration began. To better understand the relationship between features and the target variable, a correlation analysis was conducted on the 5% sampled dataset. The graph below highlights the top 15 features with the strongest correlations to the target. Feature_06 had the highest correlation at -0.046, followed by Feature_04 at -0.032—both of which are negatively correlated with responder_6. However, despite being the strongest relationships, these correlation values are generally weak. This indicates that no single feature has a dominant impact on the target variable.

Exhibit 1.1

![image](https://github.com/user-attachments/assets/aaa0f5af-54d7-425f-a196-a920542e2df9)


A feature importance analysis was also conducted to identify the most influential features in predicting the target variable. The graph below shows the top 20 most important features, ranked by the most predictive. Feature_06 is the most significant predictor. Feature_04 ranked second, though its predictive influence is considerably lower than Feature_06. 

Exhibit 1.2

![image](https://github.com/user-attachments/assets/fcbeabcc-d15c-412e-a29b-a09cfa566146)


Exhibit 1.3

![image](https://github.com/user-attachments/assets/766db2af-8240-4289-840f-dabc79d5c6be)


The figure above illustrates the distribution of the top five most important features. It indicates the data for each feature follows a normal distribution pattern.
Further Analysis of Dataset
In an effort to understand the dataset’s underlying meaning, the team hypothesized that the feature values might represent a rate of return. Jane Street included columns for date and time; however, both were anonymized and reduced to simple integer values indicating a progression of time.
As shown in Exhibit 1.4, we can plot any symbol in the dataset against this anonymized time index to produce a time series. The resulting plot displays patterns reminiscent of a stock price or a volatility metric, reinforcing the idea that these feature values could indeed be related to returns or price movements.

Exhibit 1.4:

![image](https://github.com/user-attachments/assets/2a003dc5-17a9-4fc6-969d-7faeb40e7dea)

---

## Models & Methodology

---

## Results & Analysis

---

## Setup & Installation

---

## Authors

Keli Davis

Benjamin Tisinger

Nikolay Yurashku
