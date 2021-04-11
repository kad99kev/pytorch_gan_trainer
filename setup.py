from setuptools import setup, find_packages


def load_requirements():
    with open("requirements.txt", "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    requirements = []
    for line in lines:
        if line:
            requirements.append(line)

    return requirements


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
)
