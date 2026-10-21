program test_vector_search_exact
    use, intrinsic :: iso_fortran_env, only: real32, int64

    use fnlp_vector_search_core, only: &
        l2_search_serial, &
        l2_search_openmp, &
        ip_search_serial, &
        ip_search_openmp

    implicit none

    integer, parameter :: dimension = 3
    integer, parameter :: vector_count = 4
    integer, parameter :: k = 2

    real(real32), parameter :: tolerance = 1.0e-6_real32

    real(real32) :: query_vector(dimension)
    real(real32) :: database_vectors(dimension, vector_count)

    real(real32) :: l2_serial_values(k)
    real(real32) :: l2_openmp_values(k)
    integer(int64) :: l2_serial_indices(k)
    integer(int64) :: l2_openmp_indices(k)

    real(real32) :: ip_serial_values(k)
    real(real32) :: ip_openmp_values(k)
    integer(int64) :: ip_serial_indices(k)
    integer(int64) :: ip_openmp_indices(k)

    real(real32) :: expected_l2_values(k)
    integer(int64) :: expected_l2_indices(k)

    real(real32) :: expected_ip_values(k)
    integer(int64) :: expected_ip_indices(k)


    query_vector = [ &
        1.0_real32, &
        2.0_real32, &
        3.0_real32 &
    ]

    database_vectors(:, 1) = [ &
        1.0_real32, &
        2.0_real32, &
        3.0_real32 &
    ]

    database_vectors(:, 2) = [ &
        4.0_real32, &
        5.0_real32, &
        6.0_real32 &
    ]

    database_vectors(:, 3) = [ &
        0.0_real32, &
        0.0_real32, &
        0.0_real32 &
    ]

    database_vectors(:, 4) = [ &
        -1.0_real32, &
        2.0_real32, &
        1.0_real32 &
    ]


    ! Complete squared-L2 scores:
    ! [0, 27, 14, 8]
    !
    ! The two smallest values are:
    ! value 0 at zero-based ID 0
    ! value 8 at zero-based ID 3

    expected_l2_values = [ &
        0.0_real32, &
        8.0_real32 &
    ]

    expected_l2_indices = [ &
        0_int64, &
        3_int64 &
    ]


    ! Complete inner-product scores:
    ! [14, 32, 0, 6]
    !
    ! The two largest values are:
    ! value 32 at zero-based ID 1
    ! value 14 at zero-based ID 0

    expected_ip_values = [ &
        32.0_real32, &
        14.0_real32 &
    ]

    expected_ip_indices = [ &
        1_int64, &
        0_int64 &
    ]


    call l2_search_serial( &
        query_vector, &
        database_vectors, &
        k, &
        l2_serial_values, &
        l2_serial_indices &
    )

    call l2_search_openmp( &
        query_vector, &
        database_vectors, &
        k, &
        l2_openmp_values, &
        l2_openmp_indices &
    )

    call ip_search_serial( &
        query_vector, &
        database_vectors, &
        k, &
        ip_serial_values, &
        ip_serial_indices &
    )

    call ip_search_openmp( &
        query_vector, &
        database_vectors, &
        k, &
        ip_openmp_values, &
        ip_openmp_indices &
    )


    if (any( &
        abs(l2_serial_values - expected_l2_values) > tolerance &
    )) then
        error stop "Serial L2 top-k values are incorrect."
    end if

    if (any(l2_serial_indices /= expected_l2_indices)) then
        error stop "Serial L2 top-k indices are incorrect."
    end if

    if (any( &
        abs(l2_openmp_values - expected_l2_values) > tolerance &
    )) then
        error stop "OpenMP L2 top-k values are incorrect."
    end if

    if (any(l2_openmp_indices /= expected_l2_indices)) then
        error stop "OpenMP L2 top-k indices are incorrect."
    end if


    if (any( &
        abs(ip_serial_values - expected_ip_values) > tolerance &
    )) then
        error stop "Serial IP top-k values are incorrect."
    end if

    if (any(ip_serial_indices /= expected_ip_indices)) then
        error stop "Serial IP top-k indices are incorrect."
    end if

    if (any( &
        abs(ip_openmp_values - expected_ip_values) > tolerance &
    )) then
        error stop "OpenMP IP top-k values are incorrect."
    end if

    if (any(ip_openmp_indices /= expected_ip_indices)) then
        error stop "OpenMP IP top-k indices are incorrect."
    end if


    print *, "Serial L2 top-k values:  ", l2_serial_values
    print *, "Serial L2 top-k IDs:     ", l2_serial_indices

    print *, "OpenMP L2 top-k values:  ", l2_openmp_values
    print *, "OpenMP L2 top-k IDs:     ", l2_openmp_indices

    print *

    print *, "Serial IP top-k values:  ", ip_serial_values
    print *, "Serial IP top-k IDs:     ", ip_serial_indices

    print *, "OpenMP IP top-k values:  ", ip_openmp_values
    print *, "OpenMP IP top-k IDs:     ", ip_openmp_indices

    print *
    print *, "Exact vector-search correctness: PASS"
    print *, "Serial/OpenMP search equivalence: PASS"

end program test_vector_search_exact