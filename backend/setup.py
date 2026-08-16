from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    """Reads requirements.txt and strips editable install flag if present."""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='agri-shield',
    version='0.0.1',
    author='A.G.R.I.-S.H.I.E.L.D. Team',
    packages=find_packages(),
    description="Automated Geospatial Risk Identification & Spatio-Temporal Hotspot Interactive Ecosystem for Localized Defense",
    install_requires=get_requirements('requirements.txt')
)
