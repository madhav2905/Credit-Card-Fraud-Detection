# Credit Card Fraud Detection System 
A machine learning-powered fraud detection system with real-time monitoring dashboard built using Streamlit. This project implements and compares two powerful ensemble methods - Random Forest (RF) and XGBoost (XGB) - to detect fraudulent credit card transactions.

## 📊 Dataset 
Dataset Source: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The model is trained on credit card transaction data containing:
- **Features**: Transaction amount, time, and anonymized variables (V1-V28).
- **Target**: Binary classification (0: Normal, 1: Fraud).
- **Imbalance**: Highly imbalanced dataset typical of fraud detection scenarios.
- **Format**: CSV file with anonymized features for privacy protection.
### Data Description
- **Time**: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- **V1-V28**: PCA transformed anonymized features.
- **Amount**: Transaction amount (continuous).
- **Class**: Target variable (1 for fraud, 0 for normal transactions).

## 🚀 Live Dashboard

The application provides a real-time dashboard displaying:
- Model performance metrics comparison.
- Interactive visualizations of fraud detection results.
- Live monitoring of both Random Forest and XGBoost models.
- Transaction classification results with confidence scores.

The app can be viewed live at: https://credit-card-fraud-detection-uffpxd4enax7xytx9jxbeg.streamlit.app

## 🛠️ Technologies Used

- **Python 3.8+**.
- **NumPy** - Numerical computing.
- **Pandas** - Data manipulation and analysis.
- **Matplotlib/Plotly** - Data visualization.
- **Seaborn** - Statistical data visualization.
- **Scikit-learn** - Random Forest implementation.
- **XGBoost** - Gradient boosting framework.
- **Streamlit** - Web application framework.

## ✨ Features

- **Dual Model Architecture**: Implements both Random Forest and XGBoost algorithms.
- **Real-time Performance Monitoring**: Live dashboard showing model metrics.
- **Interactive Visualizations**: Charts and graphs for better data understanding.
- **Model Comparison**: Side-by-side performance analysis of RF vs XGB.
- **Fraud Probability Scoring**: Confidence scores for each prediction.
- **Responsive Web Interface**: Built with Streamlit for easy accessibility. 

## 📈 Model Performance

### Random Forest (RF)
- **Algorithm**: Ensemble of decision trees.
- **Advantages**: Robust to overfitting, handles missing values well.
- **Use Case**: Baseline model with interpretable results.

### XGBoost (XGB)
- **Algorithm**: Gradient boosting framework.
- **Advantages**: High performance, efficient training.
- **Use Case**: Advanced model for maximum accuracy.

## 🔍 Model Evaluation Metrics

The dashboard displays the following metrics for both models:

- **Accuracy**: Overall correctness of predictions.
- **Precision**: True positive rate for fraud detection.
- **Recall**: Sensitivity to actual fraud cases.
- **F1-Score**: Harmonic mean of precision and recall.
- **AUC-ROC**: Area under the ROC curve.
- **Confusion Matrix**: Detailed classification results.

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 🙏 Acknowledgments

- Dataset providers for credit card transaction data.
- Scikit-learn and XGBoost communities for excellent ML libraries.
- Streamlit team for the amazing web app framework.

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 
