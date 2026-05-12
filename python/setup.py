from setuptools import setup, find_packages

setup(
    name="echomap-client",
    version="0.1.0",
    description="Gym-compatible Python SDK for EchoMap simulation",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "websocket-client",
    ],
    extras_require={
        "llm": ["anthropic"],
    },
    entry_points={
        "console_scripts": [
            "echomap-boxing=echomap_client.cli:main",
        ],
    },
)
