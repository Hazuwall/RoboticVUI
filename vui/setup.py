import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vui",
    version="0.0.1",
    author="Pavel Seleznev",
    author_email="p.seleznyov2005@yandex.ru",
    description="Trainable voice user interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hazuwall/RoboticVUI",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=["matplotlib", "numpy", "tensorflow>=2.2.0", "keras",
                      "coolname", "pyrubberband", "h5py", "soundfile", "playsound" "termcolor", "pydot_ng"]
)
