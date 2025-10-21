from setuptools import setup, find_packages

setup(
    name="stable-diffusion",
    version="0.0.1",
    description="",
    packages=find_packages(),
    install_requires=[
        # Keep minimal deps here — full deps installed via pip separately
        "torch",
        "numpy",
        "tqdm",
    ],
)

