from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="easy_viz_cnn",
    version="0.1.0",
    author=[
        "Sadegh Yazdani",
        "Hossein ShahAbadi",
        "Mojtaba ZarrinPour",
    ],
    author_email=[
        "silverstar10@gmail.com",
    ],
    description="A simple tool for visualizing CNN architectures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AvidDevelopers/EasyVizCNN",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "matplotlib>=3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3 License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
