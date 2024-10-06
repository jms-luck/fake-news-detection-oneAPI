# fake-news-detection-oneAPI
# Overview:
  This project focuses on detecting fake news on social media using machine learning techniques. The AI model is optimized using Intel OneAPI's powerful libraries, such as oneDNN for deep learning model training and oneMKL for efficient data processing. The project can analyze text content from various social media platforms, flagging misinformation in real time.
# Project Structure
fake-news-detection-oneAPI/
├── data/                   # Directory for datasets
│   ├── raw/                # Raw social media posts or articles
│   └── processed/          # Processed and cleaned data
├── notebooks/              # Jupyter notebooks for experiments
├── src/                    # Source code for the project
│   ├── preprocessing.py    # Data preprocessing script
│   ├── model.py            # Model training and evaluation script
│   └── inference.py        # Script for making predictions
├── results/                # Directory to store results, models, and evaluation metrics
├── README.md               # Project documentation
└── requirements.txt        # List of dependencies
