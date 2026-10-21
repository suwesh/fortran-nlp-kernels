program test_vector_search_blas_batch
    use, intrinsic :: iso_fortran_env, only: real32, int64

    use fnlp_vector_search_blas, only: &
        ip_search_batch_blas, &
        l2_search_batch_blas

    implicit none

    integer, parameter :: dimension = 3
    integer, parameter :: vector_count = 4
    integer, parameter :: query_count = 2
    integer, parameter :: k = 2

    real(real32), parameter :: tolerance = 1.0e-5_real32

    real(real32) :: query_vectors(dimension, query_count)
    real(real32) :: database_vectors(dimension, vector_count)

    real(real32) :: ip_values(k, query_count)
    integer(int64) :: ip_indices(k, query_count)

    real(real32) :: l2_values(k, query_count)
    integer(int64) :: l2_indices(k, query_count)

    real(real32) :: expected_ip_values(k, query_count)
    integer(int64) :: expected_ip_indices(k, query_count)

    real(real32) :: expected_l2_values(k, query_count)
    integer(int64) :: expected_l2_indices(k, query_count)


    query_vectors(:, 1) = [ &
        1.0_real32, &
        2.0_real32, &
        3.0_real32 &
    ]

    query_vectors(:, 2) = [ &
        0.0_real32, &
        1.0_real32, &
        0.0_real32 &
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


    ! Query 1 inner products:
    ! [14, 32, 0, 6]
    !
    ! Query 2 inner products:
    ! [2, 5, 0, 2]

    expected_ip_values(:, 1) = [ &
        32.0_real32, &
        14.0_real32 &
    ]

    expected_ip_indices(:, 1) = [ &
        1_int64, &
        0_int64 &
    ]

    expected_ip_values(:, 2) = [ &
        5.0_real32, &
        2.0_real32 &
    ]

    expected_ip_indices(:, 2) = [ &
        1_int64, &
        0_int64 &
    ]


    ! Query 1 squared L2:
    ! [0, 27, 14, 8]
    !
    ! Query 2 squared L2:
    ! [11, 66, 1, 3]

    expected_l2_values(:, 1) = [ &
        0.0_real32, &
        8.0_real32 &
    ]

    expected_l2_indices(:, 1) = [ &
        0_int64, &
        3_int64 &
    ]

    expected_l2_values(:, 2) = [ &
        1.0_real32, &
        3.0_real32 &
    ]

    expected_l2_indices(:, 2) = [ &
        2_int64, &
        3_int64 &
    ]


    call ip_search_batch_blas( &
        query_vectors, &
        database_vectors, &
        k, &
        ip_values, &
        ip_indices &
    )

    call l2_search_batch_blas( &
        query_vectors, &
        database_vectors, &
        k, &
        l2_values, &
        l2_indices &
    )


    if (any( &
        abs(ip_values - expected_ip_values) > tolerance &
    )) then
        error stop "Batched BLAS IP values are incorrect."
    end if

    if (any(ip_indices /= expected_ip_indices)) then
        error stop "Batched BLAS IP indices are incorrect."
    end if

    if (any( &
        abs(l2_values - expected_l2_values) > tolerance &
    )) then
        error stop "Batched BLAS L2 values are incorrect."
    end if

    if (any(l2_indices /= expected_l2_indices)) then
        error stop "Batched BLAS L2 indices are incorrect."
    end if


    print *, "Batched IP values:"
    print *, ip_values

    print *, "Batched IP IDs:"
    print *, ip_indices

    print *

    print *, "Batched L2 values:"
    print *, l2_values

    print *, "Batched L2 IDs:"
    print *, l2_indices

    print *
    print *, "BLAS SGEMM batched-search correctness: PASS"

end program test_vector_search_blas_batch