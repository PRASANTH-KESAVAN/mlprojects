from setuptools import find_packages, setup
from typing import List
HYPHEN_E_DOT= "-e ."

def get_requirements(file_path: str)-> List[str]:
    req=[]
    with open(file_path) as obj:
        req= obj.readlines()
        req = [re.replace("/n", "" ) for re in req]
        
        if HYPHEN_E_DOT in req:
            req.remove(HYPHEN_E_DOT)
    return req
            
            
setup(
    name="mlprojects",
    version='0.0.1',
    author="prasanth",
    author_email="prasanthk3022@gmail.com",
    packages= find_packages(),
    install_requires=get_requirements("requirements.txt"),
    
    
    
    
)