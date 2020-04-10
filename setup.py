from setuptools import find_packages, setup
import os


def package_files(directory):

    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.sep.join(["resources", path.replace(directory, ""), filename]))

    return paths


extra_files = package_files('src/resources')

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Food Panda Challenge',
    package_data={"": extra_files},
    author='Damien Marlier',
)
