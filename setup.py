from setuptools import setup, find_packages


def load_requirements():
    with open("requirements.txt", "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    requirements = []
    for line in lines:
        if line:
            requirements.append(line)

    return requirements


test_packages = ["pytest>=6.2.3", "black>=20.8b1", "flake8>=3.9.0"]

util_packages = ["pre-commit>=2.12.0"]

dev_packages = util_packages + test_packages

setup(
    name="pytorch_gan_trainer",
    licence="MIT",
    version="0.1.0",
    url="https://github.com/kad99kev/pytorch_gan_trainer",
    author="Kevlyn Kadamala",
    author_email="kevlyn@gmail.com",
    description="A simple module for you to directly import \
        and start training different GAN models.",
    packages=find_packages(),
    install_requires=load_requirements(),
    extras_require={"dev": dev_packages, "test": test_packages},
)
