import importlib.metadata
import importlib.util
import os
import re
from typing import List

from setuptools import find_packages, setup


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _is_torch_npu_available() -> bool:
    return _is_package_available("torch_npu")


def _is_torch_available() -> bool:
    return _is_package_available("torch")


def _is_torch_cuda_available() -> bool:
    if _is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False


def get_version() -> str:
    with open(os.path.join("lingbotvla", "__init__.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("__version__")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> List[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        lines = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
        return lines

BASE_REQUIRE = []

def main():
    # Update install_requires and extras_require
    install_requires = list(set(BASE_REQUIRE + get_requires()))

    setup(
        name="lingbotvla",
        version=get_version(),
        python_requires=">=3.8.0",
        packages=find_packages(exclude=["scripts", "tasks", "tests"]),
        url="https://www.robbyant.com",
        license="Apache 2.0",
        author="Robbyant Team",
        description="From Foundation to Application: Improving VLA Models in Practice",
        install_requires=install_requires,
        include_package_data=False,
    )


if __name__ == "__main__":
    main()
