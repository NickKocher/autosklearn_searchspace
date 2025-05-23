from typing import List, Optional, Union, no_type_check

import importlib
from distutils.version import LooseVersion

import pkg_resources

from util import RE_PATTERN


def verify_packages(packages: Optional[Union[str, List[str]]]) -> None:
    if not packages:
        return
    if isinstance(packages, str):
        packages = packages.splitlines()

    for package in packages:
        if not package:
            continue

        # Ignore comments
        if package.startswith("#"):
            continue

        match = RE_PATTERN.match(package)
        if match:
            name = match.group("name")
            operation = match.group("operation1")
            version = match.group("version1")
            _verify_package(name, operation, version)
        else:
            raise ValueError("Unable to read requirement: %s" % package)


# Module has no attribute __version__ wa
@no_type_check
def _verify_package(name: str, operation: Optional[str], version: str) -> None:
    try:
        module_dist = pkg_resources.get_distribution(name)
        installed_version = LooseVersion(module_dist.version)
    except pkg_resources.DistributionNotFound:
        try:
            module = importlib.import_module(name)  # type: ignore
            installed_version = LooseVersion(module.__version__)
        except ImportError:
            raise MissingPackageError(name)

    if not operation:
        return

    required_version = LooseVersion(version)

    if operation == "==":
        check = required_version == installed_version
    elif operation == ">":
        check = installed_version > required_version
    elif operation == "<":
        check = installed_version < required_version
    elif operation == ">=":
        check = (
            installed_version > required_version
            or installed_version == required_version
        )
    else:
        raise NotImplementedError("operation '%s' is not supported" % operation)
    if not check:
        raise IncorrectPackageVersionError(
            name, installed_version, operation, required_version
        )


class MissingPackageError(Exception):
    error_message = "Mandatory package '{name}' not found!"

    def __init__(self, package_name: str):
        self.package_name = package_name
        super(MissingPackageError, self).__init__(
            self.error_message.format(name=package_name)
        )


class IncorrectPackageVersionError(Exception):
    error_message = (
        "found '{name}' version {installed_version} but requires {name} version "
        "{operation}{required_version}"
    )

    def __init__(
        self,
        package_name: str,
        installed_version: Union[str, LooseVersion],
        operation: Optional[str],
        required_version: Union[str, LooseVersion],
    ):
        self.package_name = package_name
        self.installed_version = installed_version
        self.operation = operation
        self.required_version = required_version
        message = self.error_message.format(
            name=package_name,
            installed_version=installed_version,
            operation=operation,
            required_version=required_version,
        )
        super(IncorrectPackageVersionError, self).__init__(message)
