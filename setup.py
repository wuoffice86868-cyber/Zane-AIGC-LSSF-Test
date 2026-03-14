from setuptools import setup, find_packages

setup(
    name="prompt-evaluator",
    version="0.1.0",
    description="Prompt evaluation and optimization system for AI video generation pipelines",
    author="Zane Jenkins",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "pydantic>=2.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
)
