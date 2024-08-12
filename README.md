
---

# 🎓 Student Performance Indicator: End-to-End Data Science Project

Welcome to the **Student Performance Indicator** project! This repository contains an end-to-end data science project aimed at predicting student exam performance based on various features.

## 🚀 Project Overview

This project leverages data science techniques to build a predictive model that estimates a student's exam performance. The project follows a structured data science workflow, including data collection, preprocessing, model building, evaluation, and deployment.

https://github.com/user-attachments/assets/58fd394d-d3a2-463f-bc83-664e13e0eaba

## 🗂️ Project Structure

The project is organized into the following directories and files:

### 📁 `artifacts/`
- **Description**: Contains all the raw and processed data files used in the project.
- **Files**:
  - `raw.csv`: The main dataset used for this project.
  - `test.csv`: This data is used to test the model.
  - `train.csv`: This data is used to train the model.
  - `model.pkl`: This is the best model on base of evaluation metrics.
  - `preprocesser.pkl`: This is model obtained taht can do preprocessing on dataset.

### 📁 `notebooks/`
- **Description**: Jupyter notebooks for data exploration, analysis, and model experimentation.
- **Files**:
  - `EDA_and_Model_training_student_perfomance_indicater.ipynb`: Initial data exploration and visualization.Experimentation with different machine learning models. Evaluation of model performance with different metrics.

### 📁 `src/`
- **Description**: Source code for the project, including data processing, model building, and utility functions.
- **Files**:
  - `data_ingestion.py`: Code for data ingestion,exploration.
  - `data_transformation.py`: Code for data cleaning, feature engineering, and splitting the data.
  - `model_trainer.py`: Contains code for building and training the machine learning models.
  - `prediction_pipeline.py`: Contains code for building and training the machine learning models that can make prediction


**Description**: Code for the Flask web app that serves the model.
- **Files**:
  - `app.py`: The main Flask application file.
  - `templates/`: HTML templates for the web interface.
    - `index.html`: The home page of the web app where users can input data and get predictions.
  - `static/`: Static files like CSS and JavaScript for the web app.


### 📁 `logs/`
- **Description**: Logs generated during the project’s execution, useful for tracking progress and debugging.
- **Files**:
  - `training.log`: Logs related to the model training process.
  - `app.log`: Logs from the Flask application.


### 📄 `requirements.txt`
- **Description**: A list of Python packages required to run the project. Install them using `pip install -r requirements.txt`.

### 📄 `README.md`
- **Description**: The file you're currently reading, providing an overview and explanation of the project.

### 📄 `LICENSE`
- **Description**: The license under which the project is distributed.

### 📄 `.gitignore`
- **Description**: Specifies which files and directories should be ignored by Git to avoid unnecessary clutter in the repository.

## 🔧 Tools and Technologies

- **Python**: Programming language used for data manipulation and model building.
- **Pandas & NumPy**: Libraries for data manipulation and analysis.
- **Scikit-learn**: Machine learning library for model building and evaluation.
- **Matplotlib & Seaborn**: Libraries for data visualization.
- **Flask**: Web framework used for deploying the model.
- **GitHub**: Version control and project collaboration.

## 🎯 Key Features

- **User-friendly Interface**: The web app provides an easy-to-use interface for predicting student performance.
- **Real-time Predictions**: Get instant predictions based on input data.
- **Comprehensive EDA**: In-depth analysis of data to uncover insights.
- **Robust Model**: A well-trained model that offers reliable predictions.

## 📜 How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/muhammadadilnaeem/Student-Performance-Indicater-End-To-End-Data-Science-Project.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd Student-Performance-Indicater-End-To-End-Data-Science-Project
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
5. **Run the main.py**:
   ```bash
   python main.py
   ```

6. **Run the web app**:
   ```bash
   python app.py
   ```

7. **Access the app**:
   Open your browser and go to `http://127.0.0.1:5000/` to interact with the model.

## 🛠️ Future Work

- **Add more features** to the model for better prediction accuracy.
- **Improve UI/UX** of the web app for a more interactive user experience.
- **Integrate advanced models** like deep learning for more complex predictions.

## 🙌 Contributing

Feel free to fork this repository and contribute by submitting a pull request. Contributions are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/muhammadadilnaeem/Student-Performance-Indicater-End-To-End-Data-Science-Project/blob/main/LICENSE) file for details.


MLFLOW_TRACKING_URI=[Mlflow and Dagshub Dashboard for this Project](https://dagshub.com/muhammadadilnaeem/Student-Performance-Indicater-End-To-End-Data-Science-Project.mlflow/#/experiments/0/runs/5a6a57d513a94584ae7761d7b4c4685b)

---
