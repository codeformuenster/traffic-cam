from glob import glob
from setuptools import setup, find_packages


setup(
    name="traffic_cam",
    version="0.1.0",
    license="MIT",
    packages=find_packages(),
    scripts=glob("bin/*"),
    install_requires=[
        "tensorflow==2.1.1",
        "Pillow",
        "opencv-python",
        "numpy",
        # "yolo34py",  # requires pkg-config, e.g. 'sudo apt install pkg-config#'
        "matplotlib",
        # dev
        "black",
        "flake8",
        "jupyter",
        "rope",
    ],
)
