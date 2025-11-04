from setuptools import setup, find_packages
from pathlib import Path

this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="airbnb_price_predictor",
    version="0.1.0",
    description="Airbnb Paris Price Prediction - FastAPI + Streamlit App",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Emanuele Torrisi & Katherine Stewart",
    license="MIT",
    python_requires=">=3.12",
    packages=find_packages(exclude=("tests", "docs")),
    include_package_data=True,
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.2",
        "scikit-learn==1.5.2",
        "xgboost==2.1.1",
        "joblib==1.4.2",
        "SQLAlchemy==2.0.35",
        "psycopg2-binary==2.9.9",
        "python-dotenv==1.0.1",
    ],
    extras_require={
        "api": [
            "fastapi==0.115.0",
            "uvicorn[standard]==0.32.0",
            "pydantic==2.9.2",
            "pydantic-settings==2.4.0",
        ],
        "streamlit": [
            "streamlit==1.39.0",
            "requests==2.32.3",
        ],
        "dev": [
            "pytest==8.3.3",
            "black==24.8.0",
            "isort==5.13.2",
        ],
    },
    entry_points={
        "console_scripts": [
            "predict-airbnb=scripts.predict:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Framework :: FastAPI",
        "Framework :: Streamlit",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
