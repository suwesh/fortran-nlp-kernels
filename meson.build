project(
  'fortran-nlp-kernels',
  ['cpp', 'fortran'],
  version: '0.1.0',
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

nanobind_dep = dependency(
  'nanobind',
  required: true,
)

openmp_fortran_dep = dependency(
  'openmp',
  language: 'fortran',
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
    'src/native/bindings/cpu_bindings.cpp',
  ],
  include_directories: native_include,
  dependencies: [
    nanobind_dep,
    openmp_fortran_dep,
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