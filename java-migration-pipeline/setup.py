from setuptools import setup, find_packages

setup(
    name="java-migration-pipeline",
    version="1.0.0",
    description="Automated Java 11 to 21 migration pipeline with AI assistance",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-anthropic>=0.1.0",
        "langchain-openai>=0.1.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "pandas>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "java-migrate=migration_orchestrator:main",
        ],
    },
    python_requires=">=3.9",
)
