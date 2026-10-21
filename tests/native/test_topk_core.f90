program test_topk_core
    use, intrinsic :: iso_fortran_env, only: real32, int64

    use fnlp_topk_core, only: &
        topk_smallest, &
        topk_largest

    implicit none

    integer, parameter :: value_count = 4
    integer, parameter :: k = 2

    real(real32), parameter :: tolerance = 1.0e-6_real32

    real(real32) :: values(value_count)

    real(real32) :: smallest_values(k)
    integer(int64) :: smallest_indices(k)

    real(real32) :: largest_values(k)
    integer(int64) :: largest_indices(k)

    real(real32) :: expected_smallest_values(k)
    integer(int64) :: expected_smallest_indices(k)

    real(real32) :: expected_largest_values(k)
    integer(int64) :: expected_largest_indices(k)


    values = [ &
        14.0_real32, &
        32.0_real32, &
        0.0_real32, &
        6.0_real32 &
    ]

    expected_smallest_values = [ &
        0.0_real32, &
        6.0_real32 &
    ]

    expected_smallest_indices = [ &
        2_int64, &
        3_int64 &
    ]

    expected_largest_values = [ &
        32.0_real32, &
        14.0_real32 &
    ]

    expected_largest_indices = [ &
        1_int64, &
        0_int64 &
    ]


    call topk_smallest( &
        values, &
        k, &
        smallest_values, &
        smallest_indices &
    )

    call topk_largest( &
        values, &
        k, &
        largest_values, &
        largest_indices &
    )


    if (any( &
        abs(smallest_values - expected_smallest_values) > tolerance &
    )) then
        error stop "The smallest top-k values are incorrect."
    end if

    if (any( &
        smallest_indices /= expected_smallest_indices &
    )) then
        error stop "The smallest top-k indices are incorrect."
    end if

    if (any( &
        abs(largest_values - expected_largest_values) > tolerance &
    )) then
        error stop "The largest top-k values are incorrect."
    end if

    if (any( &
        largest_indices /= expected_largest_indices &
    )) then
        error stop "The largest top-k indices are incorrect."
    end if


    print *, "Input values:         ", values

    print *
    print *, "Smallest top-k values:", smallest_values
    print *, "Smallest top-k IDs:   ", smallest_indices

    print *
    print *, "Largest top-k values: ", largest_values
    print *, "Largest top-k IDs:    ", largest_indices

    print *
    print *, "Exact top-k correctness: PASS"

end program test_topk_core