# CPU (OpenMP) port of faster_muons_torch's field-map transport: python3 setup.py build_ext --inplace
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(name="cpu_muons",
      ext_modules=[CppExtension("cpu_muons", ["cpu_muons.cpp"],
                                extra_compile_args=["-O3", "-march=native", "-fopenmp"], extra_link_args=["-fopenmp"])],
      cmdclass={"build_ext": BuildExtension})
