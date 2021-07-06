import setuptools

setuptools.setup(
    name="styleformer",
    version="0.1",
    author="Prithiviraj Damodaran",
    author_email="",
    description="Styleformer",
    long_description="A Neural Language Style Transfer framework to transfer natural language text smoothly between fine-grained language styles like formal/casual, active/passive, and many more. Created by Prithiviraj Damodaran. Open to pull requests and other forms of collaboration.",
    url="https://github.com/PrithivirajDamodaran/Styleformer.git",
    packages=setuptools.find_packages(),
    install_requires=['transformers', 'sentencepiece', 'python-Levenshtein', 'fuzzywuzzy'],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
)

