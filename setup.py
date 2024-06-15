from setuptools import setup, find_packages

setup(
    name='PsoriasisClassification',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'tensorflow',
        'matplotlib',
        'numpy',
        'pandas',
        'scikit-learn',
        'seaborn',
    ],
)
