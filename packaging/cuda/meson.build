project(
  'fortran-nlp-kernels-cuda',
  version: '0.1.2',
  meson_version: '>=1.3.0',
  default_options: [
    'buildtype=release',
  ],
)

python = import('python').find_installation(
  pure: false,
)

bash = find_program(
    'bash',
    required: true,
)

nvhpc_root = get_option('fnlp_nvhpc_root')

nvfortran = find_program(
    join_paths(nvhpc_root, 'compilers', 'bin', 'nvfortran'),
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
    get_option('fnlp_nvhpc_root'),
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
    'fnlp_cuda_companion_extension',
    input: [
        'src/native/cpu/vector_search/topk_core.f90',
        'src/native/cuda/lexical/lexical_cuda_core.cuf',
        'src/native/cuda/lexical/lexical_cuda_c_api.cuf',
        'src/native/include/fortran_nlp_cuda_lexical.h',
        'src/native/cuda/vector_search/vector_search_cuda_core.cuf',
        'src/native/cuda/vector_search/vector_search_cublas.cuf',
        'src/native/cuda/vector_search/vector_search_cuda_c_api.cuf',
        'src/native/include/fortran_nlp_cuda_vector_search.h',
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
    'CUDA companion extension enabled: '
    + cuda_extension_name
)

message(
    'CUDA companion target: '
    + get_option('fnlp_cuda_arch')
    + ', '
    + get_option('fnlp_cuda_toolkit')
)

message(
    'NVIDIA HPC SDK root: '
    + get_option('fnlp_nvhpc_root')
)