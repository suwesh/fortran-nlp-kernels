program test_vector_search_blas
    use, intrinsic :: iso_fortran_env, only: real32, int64

    use fnlp_vector_search_core, only: &
        l2_search_serial, &
        ip_search_serial

    use fnlp_vector_search_blas, only: &
        l2_search_blas, &
        ip_search_blas

    implicit none

    integer, parameter :: dimension = 3
    integer, parameter :: vector_count = 4
    integer, parameter :: k = 2

    real(real32), parameter :: tolerance = 1.0e-5_real32

    real(real32) :: query_vector(dimension)
    real(real32) :: database_vectors(dimension, vector_count)

    real(real32) :: l2_serial_values(k)
    integer(int64) :: l2_serial_indices(k)

    real(real32) :: l2_blas_values(k)
    integer(int64) :: l2_blas_indices(k)

    real(real32) :: ip_serial_values(k)
    integer(int64) :: ip_serial_indices(k)

    real(real32) :: ip_blas_values(k)
    integer(int64) :: ip_blas_indices(k)


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


    call l2_search_serial( &
        query_vector, &
        database_vectors, &
        k, &
        l2_serial_values, &
        l2_serial_indices &
    )

    call l2_search_blas( &
        query_vector, &
        database_vectors, &
        k, &
        l2_blas_values, &
        l2_blas_indices &
    )

    call ip_search_serial( &
        query_vector, &
        database_vectors, &
        k, &
        ip_serial_values, &
        ip_serial_indices &
    )

    call ip_search_blas( &
        query_vector, &
        database_vectors, &
        k, &
        ip_blas_values, &
        ip_blas_indices &
    )


    if (any( &
        abs(l2_blas_values - l2_serial_values) > tolerance &
    )) then
        error stop "BLAS and handwritten L2 values differ."
    end if

    if (any(l2_blas_indices /= l2_serial_indices)) then
        error stop "BLAS and handwritten L2 indices differ."
    end if

    if (any( &
        abs(ip_blas_values - ip_serial_values) > tolerance &
    )) then
        error stop "BLAS and handwritten IP values differ."
    end if

    if (any(ip_blas_indices /= ip_serial_indices)) then
        error stop "BLAS and handwritten IP indices differ."
    end if


    print *, "Handwritten L2 values:", l2_serial_values
    print *, "BLAS L2 va*ues:       ", l2_blas_values
    print *, "BLAS L2 IDs:          ", l2_blas_indices

    print *

    print *, "Handwritten IP values:", ip_serial_values
    print *, "BLAS I* values:       ", ip_blas_values
    print *, "BLAS IP IDs:          ", ip_blas_indices

    print *
    print *, "BLAS vector-search correctness: PASS"
    print *, "BLAS/handwritten equivalence: PASS"

end program test_vector_search_blas