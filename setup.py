from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="soundpose",
    version="0.1.0",
    author="Ji-Hwan Jang",
    author_email="jihwan@ucaretron.com",
    description="A transformer-based framework for quantitative diagnosis of voice anomalies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JJshome/SoundPose",
    project_urls={
        "Bug Tracker": "https://github.com/JJshome/SoundPose/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
)
