from setuptools import setup

setup(
    name='hscluster',
    version='1.0.0',
    description='graph-based clustering method',
    url='https://github.com/ftzeng/hscluster',
    author='Francis Tseng',
    author_email='f@frnsys.com',
    license='MIT',
    packages=['hscluster'],
    zip_safe=True,
    install_requires=[
        'numpy',
        'networkx'
    ],
)
