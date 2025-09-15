# Assignment 4 (GMM-Based Synthetic Sampling for Imbalanced Data)

## Student Info:
Name: Shashank Satish Adsule\
Roll no.: DA25M005

## Dataset Used
- [**Credit Card Fraud Detection**](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- this data set contains one CSV files: 
    - `creditcard.csv`
- The dataset contains 31 colume feature, with the target variable being `"class"`:
    - class -> '0' for valid user and '1' for fraud user
    - 28 PCA components-> v1..v28
    - amount -> transaction amount
    - time -> seconds 

## Observations
- since the dataset is highly imbalanced, Logistic Regression on the raw imbalanced dataset achieves very high accuracy and strong performance on the majority class, but fails to detect fraud effectively
- using GMM oversampling increases the representation of minority samples and improves both recall and F1-score for fraud detection.
- Adding CBU alongside GMM further improves precision and F1-score by reducing redundant majority samples.

![models score](./assests/model%20comparision%20(lbfgs_1000_0.3).png)

##  Python Dependencies
The following libraries were used in the analysis:

```bash
pandas             # data handling
numpy              # numerical computation
matplotlib         # data visualization
seaborn            # statistical visualization

scikit-learn       # train-test split, resampling, model building, and evaluation
    ├── model_selection (train_test_split)  
    ├── utils (resample)  
    ├── metrics (confusion_matrix, accuracy_score, classification_report)  
    ├── linear_model (LogisticRegression)  
    ├── mixture (GaussianMixture)  
    └── cluster (KMeans)

```

<!-- # ├── and └── -->

## Conclusion
- By applying Gaussian Mixture Model (GMM) based oversampling, the minority class was  augmented with synthetic samples, which improved recall and F1-score for fraud detection.
- Further combining GMM with Clustering-Based Undersampling (CBU) provided the best overall performance, as redundant majority samples were removed and minority samples were enhanced. 
- Overall, GMM + CBU proved to be the most effective strategy in this experiment, striking a better balance between precision and recall compared to the baseline model or GMM alone. 

<!--upload link: https://docs.google.com/forms/d/e/1FAIpQLSdkaNzY0fGkpD07-Hq2ke1kX92cXedSrnW-pLdzUgh0e_IPFg/viewform -->
