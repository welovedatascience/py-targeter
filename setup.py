from setuptools import setup

VERSION = '0.0.1'

with open("README.md", 'r') as f:
    long_description = f.read()

# https://www.reddit.com/r/Python/comments/3uzl2a/comment/cxk3z76/
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
   name='pytargeter',
   version='1.0',
   description='Efficient Visual Targets Exploration',
   license="MIT",
   author='WeLoveDataScience',
   author_email='eric.lecoutre@welovedatascience.com',
   packages=['targeter'],  #same as name
   install_requires=requirements, #external packages as dependencies
)