import setuptools

long_description = ''

install_requires = [
    'numpy',
    'scikit-learn',
    'tqdm',
    'torch >= 1.6.0',
    'torchvision',
]

extras_requires = None

setuptools.setup(
    name="pt_ranking",
    version="0.1",
    author="II-Research",
    author_email="yuhaitao@slis.tsukuba.ac.jp",
    description="A library of scalable and extendable implementations of typical learning-to-rank methods based on PyTorch.",
    license="MIT License",
    keywords=['Learning-to-rank', 'PyTorch'],
    url="https://pt-ranking.github.io",
    package_dir={"": "pt_ranking"},
    packages=setuptools.find_packages(where="pt_ranking"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    extras_require=extras_requires
)