module fnlp_topk_core
    use, intrinsic :: iso_fortran_env, only: real32, int64
    implicit none
    private
    public :: topk_smallest
    public :: topk_largest

contains
    ! Select the k smallest values. Used for squared L2 distance.
    subroutine topk_smallest( &
        values, &
        k, &
        topk_values, &
        topk_indices &
    )
        real(real32), intent(in) :: values(:)
        integer, intent(in) :: k
        real(real32), intent(out) :: topk_values(k)
        integer(int64), intent(out) :: topk_indices(k)

        logical, allocatable :: selected(:)

        integer(int64) :: value_count
        integer(int64) :: value_index
        integer :: result_index
        real(real32) :: best_value
        integer(int64) :: best_index

        value_count = size(values, kind=int64)

        if (k < 1) then
            error stop "k must be at least 1."
        end if

        if (int(k, int64) > value_count) then
            error stop "k cannot exceed the number of values."
        end if

        allocate(selected(value_count))
        selected = .false.

        do result_index = 1, k
            best_value = huge(0.0_real32)
            best_index = 0_int64
            do value_index = 1_int64, value_count
                if (selected(value_index)) then
                    cycle
                end if
                if (values(value_index) < best_value) then
                    best_value = values(value_index)
                    best_index = value_index
                end if
            end do
            selected(best_index) = .true.
            topk_values(result_index) = best_value
            ! Return zero-based IDs for Python compatibility.
            topk_indices(result_index) = best_index - 1_int64
        end do
        deallocate(selected)
    end subroutine topk_smallest

    ! Select the k largest values. Used for inner product and cosine similarity.
    subroutine topk_largest( &
        values, &
        k, &
        topk_values, &
        topk_indices &
    )
        real(real32), intent(in) :: values(:)
        integer, intent(in) :: k
        real(real32), intent(out) :: topk_values(k)
        integer(int64), intent(out) :: topk_indices(k)

        logical, allocatable :: selected(:)

        integer(int64) :: value_count
        integer(int64) :: value_index
        integer :: result_index
        real(real32) :: best_value
        integer(int64) :: best_index

        value_count = size(values, kind=int64)

        if (k < 1) then
            error stop "k must be at least 1."
        end if

        if (int(k, int64) > value_count) then
            error stop "k cannot exceed the number of values."
        end if

        allocate(selected(value_count))
        selected = .false.

        do result_index = 1, k
            best_value = -huge(0.0_real32)
            best_index = 0_int64
            do value_index = 1_int64, value_count
                if (selected(value_index)) then
                    cycle
                end if
                if (values(value_index) > best_value) then
                    best_value = values(value_index)
                    best_index = value_index
                end if
            end do
            selected(best_index) = .true.
            topk_values(result_index) = best_value
            ! Return zero-based IDs for Python compatibility.
            topk_indices(result_index) = best_index - 1_int64
        end do
        deallocate(selected)
    end subroutine topk_largest

end module fnlp_topk_core