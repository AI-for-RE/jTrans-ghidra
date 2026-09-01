# TAKEN FROM BINSIM ANALYZER
# 25/08/2026

from dataclasses import dataclass, fields
from typing import Any, TextIO
import os

@dataclass
class Variant:
    compiler: str
    compiler_version: str
    optimization: str
    arch: str

    def __str__(self) -> str:
        return f"{self.compiler}_{self.compiler_version}_{self.arch}_{self.optimization}"

    def to_jtrans(self) -> str:
        # we don't have a notion of bitness either in jtrans
        return f"{self.compiler}-{self.compiler_version}-{self.arch}-{self.optimization}"

    def __eq__(self, other) -> bool:
        return str(self) == str(other)

def get_info(path):
    basename = os.path.basename(os.path.normpath(path))
    lib_info, separator, _ = basename.rpartition("-")
    head, compiler_name, compiler_version, arch, optimisation_level = lib_info.rsplit('-', 4)
    lib_name, _, version_name = head.partition('-')
    oup_variant = Variant(compiler_name, compiler_version, optimisation_level, arch)
    return lib_name, version_name, oup_variant

def get_comp_string(path):
    lib_name, version_name, variant = get_info(path)
    comp_string = '-'.join([version_name, variant.compiler, variant.compiler_version, variant.arch, variant.optimization])
    return comp_string