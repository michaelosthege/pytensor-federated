import os
import pathlib
import re

import setuptools

__packagename__ = "pytensor_federated"
ROOT = pathlib.Path(__file__).parent


def package_files(directory):
    assert os.path.exists(directory)
    fp_typed = pathlib.Path(__packagename__, "py.typed")
    fp_typed.touch()
    paths = [str(fp_typed.absolute())]
    for (path, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


def get_version():
    VERSIONFILE = os.path.join(__packagename__, "__init__.py")
    initfile_lines = open(VERSIONFILE, "rt").readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError(f"Unable to find version string in {VERSIONFILE}.")


__version__ = get_version()


setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version=__version__,
    description="This package helps to reduce the amount of boilerplate code when creating Airflow DAGs from Python callables.",
    long_description=open(ROOT / "README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/michaelosthege/pytensor-federated",
    download_url=f"https://github.com/michaelosthege/pytensor-federated/tarball/{__version__}",
    author="Michael Osthege",
    author_email="michael.osthege@outlook.com",
    license="GNU Affero General Public License v3",
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU Affero General Public License v3",
    ],
    install_requires=open(pathlib.Path(ROOT, "requirements.txt")).readlines(),
    package_data={
        __packagename__: package_files(
            str(pathlib.Path(pathlib.Path(__file__).parent, __packagename__).absolute())
        )
    },
)
