from setuptools import setup, find_packages

setup(
    name="transformers_distiller",
    version="0.1.0",
    description="A Hugging Face model distillation trainer",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Dhiraj Patil",
    author_email="patildhiraj1197@gmail.com",
    python_requires=">=3.9",
    license="Apache-2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers==4.55.2",
        "datasets==4.0.0",
        "accelerate==1.10.0",
        "bitsandbytes==0.47.0",
        "huggingface-hub==0.34.4",
        "safetensors==0.6.2",
        "numpy>=2.1.2",
        "pandas>=2.3.1",
        "tqdm>=4.67.1"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
    zip_safe=False,
)