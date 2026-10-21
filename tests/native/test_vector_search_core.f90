program test_vector_search_core
    use, intrinsic :: iso_fortran_env, only: real32

    use fnlp_vector_search_core, only: &
        squared_l2_distance, &
        inner_product, &
        l2_database_serial, &
        l2_database_openmp, &
        ip_database_serial, &
        ip_database_openmp

    implicit none

    integer, parameter :: dimension = 3
    integer, parameter :: vector_count = 4

    real(real32), parameter :: tolerance = 1.0e-6_real32

    real(real32) :: query_vector(dimension)

    real(real32) :: database_vectors( &
        dimension, &
        vector_count &
    )

    real(real32) :: pairwise_l2
    real(real32) :: pairwise_ip

    real(real32) :: serial_l2(vector_count)
    real(real32) :: openmp_l2(vector_count)

    real(real32) :: serial_ip(vector_count)
    real(real32) :: openmp_ip(vector_count)

    real(real32) :: expected_l2(vector_count)
    real(real32) :: expected_ip(vector_count)


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


    ! Squared L2:
    !
    ! Vector 1:
    ! (1-1)^2 + (2-2)^2 + (3-3)^2 = 0
    !
    ! Vector 2:
    ! (1-4)^2 + (2-5)^2 + (3-6)^2
    ! = 9 + 9 + 9 = 27
    !
    ! Vector 3:
    ! 1^2 + 2^2 + 3^2 = 14
    !
    ! Vector 4:
    ! (1+1)^2 + (2-2)^2 + (3-1)^2
    ! = 4 + 0 + 4 = 8

    expected_l2 = [ &
        0.0_real32, &
        27.0_real32, &
        14.0_real32, &
        8.0_real32 &
    ]


    ! Inner product:
    !
    ! Vector 1:
    ! 1*1 + 2*2 + 3*3 = 14
    !
    ! Vector 2:
    ! 1*4 + 2*5 + 3*6 = 32
    !
    ! Vector 3:
    ! 0
    !
    ! Vector 4:
    ! 1*(-1) + 2*2 + 3*1 = 6

    expected_ip = [ &
        14.0_real32, &
        32.0_real32, &
        0.0_real32, &
        6.0_real32 &
    ]


    call squared_l2_distance( &
        query_vector, &
        database_vectors(:, 2), &
        pairwise_l2 &
    )

    call inner_product( &
        query_vector, &
        database_vectors(:, 2), &
        pairwise_ip &
    )

    if (abs(pairwise_l2 - 27.0_real32) > tolerance) then
        error stop "Pairwise squared L2 test failed."
    end if

    if (abs(pairwise_ip - 32.0_real32) > tolerance) then
        error stop "Pairwise inner-product test failed."
    end if


    call l2_database_serial( &
        query_vector, &
        database_vectors, &
        serial_l2 &
    )

    call l2_database_openmp( &
        query_vector, &
        database_vectors, &
        openmp_l2 &
    )

    call ip_database_serial( &
        query_vector, &
        database_vectors, &
        serial_ip &
    )

    call ip_database_openmp( &
        query_vector, &
        database_vectors, &
        openmp_ip &
    )


    if (any(abs(serial_l2 - expected_l2) > tolerance)) then
        error stop "Serial squared L2 results are incorrect."
    end if

    if (any(abs(openmp_l2 - expected_l2) > tolerance)) then
        error stop "OpenMP squared L2 results are incorrect."
    end if

    if (any(abs(serial_ip - expected_ip) > tolerance)) then
        error stop "Serial inner-product results are incorrect."
    end if

    if (any(abs(openmp_ip - expected_ip) > tolerance)) then
        error stop "OpenMP inner-product results are incorrect."
    end if


    print *, "Expected squared L2:", expected_l2
    print *, "Serial squared L2:  ", serial_l2
    print *, "OpenMP squared L2:  ", openmp_l2

    print *

    print *, "Expected inner product:", expected_ip
    print *, "Serial inner product:  ", serial_ip
    print *, "OpenMP inner product:  ", openmp_ip

    print *
    print *, "Vector-search core correctness: PASS"
    print *, "Serial/OpenMP equivalence: PASS"

end program test_vector_search_core