from setuptools import find_packages, setup
from typing import List


# define a function to install requirements.txt files automatically
def get_requirements(file_path:str)->List[str]:
    '''
    This Function will return list of requirements.
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="End to End Data Science Project", # Change this according to project name.
    version="0.0.1",
    author="Adil Naeem",
    author_email="madilnaeem0@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)