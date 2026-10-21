project(
  'fortran-nlp-kernels',
  ['cpp', 'fortran'],
  version: '0.1.0b1',
  meson_version: '>=1.3.0',
  default_options: [
    'cpp_std=c++17',
    'buildtype=release',
    'warning_level=2',
  ],
)

python = import('python').find_installation(
  pure: false,
)

python.install_sources(
  'src/fortran_nlp/__init__.py',
  subdir: 'fortran_nlp',
)

python.install_sources(
  [
    'src/fortran_nlp/lexical/__init__.py',
    'src/fortran_nlp/lexical/index.py',
    'src/fortran_nlp/lexical/preprocessing.py',
    'src/fortran_nlp/lexical/reference.py',
  ],
  subdir: 'fortran_nlp/lexical',
)

python.install_sources(
  [
    'src/fortran_nlp/vector_search/__init__.py',
    'src/fortran_nlp/vector_search/flat.py',
    'src/fortran_nlp/vector_search/normalization.py',
    'src/fortran_nlp/vector_search/reference.py',
  ],
  subdir: 'fortran_nlp/vector_search',
)

nanobind_dep = dependency(
  'nanobind',
  required: true,
)

openmp_fortran_dep = dependency(
  'openmp',
  language: 'fortran',
  required: true,
)

fortran_compiler = meson.get_compiler('fortran')

openblas_dep = fortran_compiler.find_library(
  'openblas',
  required: true,
)

native_include = include_directories(
  'src/native/include',
)

cpu_extension = python.extension_module(
  '_cpu',
  sources: [
    'src/native/cpu/lexical/lexical_core.f90',
    'src/native/cpu/lexical/lexical_c_api.f90',
    'src/native/cpu/vector_search/topk_core.f90',
    'src/native/cpu/vector_search/vector_search_core.f90',
    'src/native/cpu/vector_search/vector_search_blas.f90',
    'src/native/cpu/vector_search/vector_search_c_api.f90',
    'src/native/bindings/cpu_bindings.cpp',
  ],
  include_directories: native_include,
  dependencies: [
    nanobind_dep,
    openmp_fortran_dep,
    openblas_dep,
  ],
  fortran_args: [
    '-O3',
    '-fopenmp',
  ],
  cpp_args: [
    '-O3',
  ],
  link_args: [
    '-fopenmp',
  ],
  install: true,
  subdir: 'fortran_nlp',
)