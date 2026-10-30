project(
  'fortran-nlp-kernels',
  ['cpp', 'fortran'],
  version: '0.1.1dev',
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
  [
    'src/fortran_nlp/__init__.py',
    'src/fortran_nlp/auto.py',
    'src/fortran_nlp/profile.py',
    'src/fortran_nlp/runtime.py',
    'src/fortran_nlp/calibration.py',
  ],
  subdir: 'fortran_nlp',
)

python.install_sources(
  [
    'src/fortran_nlp/policies/__init__.py',
  ],
  subdir: 'fortran_nlp/policies',
)

install_data(
  'src/fortran_nlp/policies/cpu_fallback_v1.json',
  install_dir: join_paths(
    python.get_install_dir(),
    'fortran_nlp',
    'policies',
  ),
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

if get_option('fnlp_cuda')
  bash = find_program(
    'bash',
    required: true,
  )

  nvfortran = find_program(
    '/opt/nvidia/hpc_sdk/Linux_x86_64/26.9/compilers/bin/nvfortran',
    required: true,
  )

  cuda_builder = find_program(
    'tools/build_cuda_extension.sh',
    required: true,
  )

  extension_suffix_result = run_command(
    python,
    '-c',
    'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))',
    check: true,
  )

  extension_suffix = extension_suffix_result.stdout().strip()
  cuda_extension_name = '_cuda' + extension_suffix

  cuda_environment = environment()

  cuda_environment.set(
    'NVHPC_ROOT',
    '/opt/nvidia/hpc_sdk/Linux_x86_64/26.9',
  )

  cuda_environment.set(
    'CUDA_ARCH',
    get_option('fnlp_cuda_arch'),
  )

  cuda_environment.set(
    'CUDA_TOOLKIT',
    get_option('fnlp_cuda_toolkit'),
  )

  cuda_extension = custom_target(
    'fnlp_cuda_extension',
    input: [
      'src/native/cuda/lexical/lexical_cuda_core.cuf',
      'src/native/cuda/lexical/lexical_cuda_c_api.cuf',
      'src/native/include/fortran_nlp_cuda_lexical.h',
      'src/native/bindings/cuda_bindings.cpp',
      'tools/build_cuda_extension.sh',
    ],
    output: cuda_extension_name,
    command: [
      bash,
      cuda_builder,
      '@BUILD_ROOT@',
      '@OUTPUT@',
      '@PRIVATE_DIR@',
    ],
    env: cuda_environment,
    build_by_default: true,
    install: true,
    install_dir: join_paths(
      python.get_install_dir(),
      'fortran_nlp',
    ),
  )

  message(
    'CUDA Fortran extension enabled: '
    + cuda_extension_name
  )

  message(
    'CUDA target: '
    + get_option('fnlp_cuda_arch')
    + ', '
    + get_option('fnlp_cuda_toolkit')
  )
endif