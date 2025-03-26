# Customer Segmentation using Unsupervised Learning

## Overview
Customer segmentation is a process of dividing customers into groups based on their common characteristics. This project leverages unsupervised learning techniques, specifically **K-Means Clustering**, to segment customers based on their purchasing behavior and demographic data.

## Features
- Data preprocessing and cleaning
- Feature engineering and scaling
- Applying **K-Means Clustering** for customer segmentation
- Evaluating cluster performance using **Elbow Method & Silhouette Score**
- Visualizing customer segments using **Matplotlib & Seaborn**

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Dataset
The dataset used for this project consists of customer purchase records, including attributes such as:
- Customer ID
- Age
- Gender
- Annual Income
- Spending Score

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/customer-segmentation.git
   cd customer-segmentation
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the data preprocessing script:
   ```bash
   python preprocess.py
   ```
2. Perform customer segmentation:
   ```bash
   python segmentation.py
   ```
3. View visualizations:
   ```bash
   python visualize.py
   ```

## Results & Analysis
After applying **K-Means Clustering**, the customers are segmented into distinct groups based on their spending behavior and income levels. These insights can help businesses personalize marketing strategies and improve customer engagement.

## Future Improvements
- Experiment with **Hierarchical Clustering** and **DBSCAN** for comparison
- Integrate **PCA** for dimensionality reduction
- Implement a **web dashboard** to interactively explore segments

## License
This project is licensed under the MIT License. Feel free to use and modify it.

## Author
[Aashutosh Sah]

For any queries, feel free to reach out via email or GitHub issues.

