from setuptools import setup, find_packages

setup(
    name="gdf",
    packages=find_packages(),
    description="Generic Diffusion Framework",
    version="0.1.8",
    url="https://github.com/WARP-AI/gdf",
    author="Pablo Pernías",
    author_email="pablo@pernias.com",
    keywords=["pip", "pytorch", "tools", "diffusion", "generative", "models", "edm"],
    zip_safe=False,
    install_requires=[
        "torch>=1.6",
        "numpy>=1.0",
    ],
    package_data={},
    include_package_data=True,
)
