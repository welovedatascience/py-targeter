from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

# https://www.reddit.com/r/Python/comments/3uzl2a/comment/cxk3z76/
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
   name='py-targeter',
   version='1.0',
   description='Efficient Visual Targets Exploration',
   license="MIT",
   author='WeLoveDataScience',
   author_email='eric.lecoutre@welovedatascience.com',
   packages=['py-targeter'],  #same as name
   install_requires=requirements, #external packages as dependencies
)