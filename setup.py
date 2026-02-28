# setup.py
from setuptools import setup, find_packages
from pathlib import Path

README = (Path(__file__).parent / "README.md")
long_description = README.read_text(encoding="utf-8") if README.exists() else ""

setup(
    name="rhi-lr-ik",           
    version="0.1.0",
    description="Robotic IK training utilities (robot, utils, model, data, logger)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="desc",
    python_requires=">=3.9",

    packages=find_packages(include=["robot*", "utils*", "model*", "data*", "logger*"]),

    install_requires=[
        "numpy>=1.24",
        "matplotlib>=3.7",
    ],
    extras_require={
        # "gpu": ["torch>=2.3.0"],
    },

    include_package_data=True,   

    scripts=[
        "train_rhi_ik.py",
        "train_lr_ik.py",
        "train_combine.py",
        "train_fine_tune.py",
        "test_dynamic_gt.py",
        "test_dynamic_two_stage.py",
    ],
)
