# import os
# from pathlib import Path
# import logging

# logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

# project_name = "Machine-Learning-Poject" # Change it Acoording to requirements

# list_of_files = [
#     #".github/workflows/.gitkeep", # Activete if you need to 
#     f"src/{project_name}/__init__.py",
#     f"src/{project_name}/components/__init__.py",
#     f"src/{project_name}/components/data_ingestion.py",
#     f"src/{project_name}/components/data_transformation.py",
#     f"src/{project_name}/components/model_trainer.py",
#     f"src/{project_name}/components/model_monitoring.py",
#     f"src/{project_name}/pipelines/__init__.py",
#     f"src/{project_name}/pipelines/training_pipeline.py",
#     f"src/{project_name}/pipelines/prediction_pipeline.py",
#     f"src/{project_name}/exception.py",
#     f"src/{project_name}/logger.py",
#     f"src/{project_name}/utils.py",
#     "Dockerfile",
#     "requirements.txt",
#     "setup.py",
#     ".env",
#     "notebooks/experimentation.ipynb",
#     "app.py",
#     "static/style.css",
#     "templates/index.html",
# ]

# for file_path in list_of_files:
#     file_path = Path(file_path)
#     file_dir, file_name = os.path.split(file_path)
    
#     if file_dir != "":
#         os.makedirs(file_dir, exist_ok=True)
#         logging.info(f"Directory created: {file_dir} for the file {file_name}.")

#         if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
#             with open(file_path, "w") as f:
#                 pass
#             logging.info(f"File created: {file_path}.")
#         else:
#             logging.info(f"File already exists: {file_path}.")


import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

project_name = "mlproject"  # Change it according to requirements

list_of_files = [
    # ".github/workflows/.gitkeep", # Activate if you need to 
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_monitoring.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/training_pipeline.py",
    f"src/{project_name}/pipelines/prediction_pipeline.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    #".env",    # Activate if you need 
    "notebooks/experimentation.ipynb",
    "app.py",
    "static/style.css",
    "templates/index.html",
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Directory created: {file_dir} for the file {file_name}.")

    if not file_path.exists():
        with open(file_path, "w") as f:
            pass
        logging.info(f"File created: {file_path}.")
    else:
        logging.info(f"File already exists: {file_path}.")
