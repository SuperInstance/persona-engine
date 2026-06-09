from setuptools import setup, find_packages

setup(
    name="persona-engine",
    version="0.1.0",
    description="Decompose, compose, and vibe-code personalities as vector databases with rhythmic TTS rendering",
    author="SuperInstance Fleet",
    author_email="superinstance@users.noreply.github.com",
    url="https://github.com/SuperInstance/persona-engine",
    packages=find_packages(include=["persona_engine", "persona_engine.*"]),
    python_requires=">=3.11",
    install_requires=[
        "pydantic>=2.9.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "tts": ["piper-tts>=1.0.0"],
        "decompose": ["opensmile>=2.0.0", "whisper"],
    },
)
