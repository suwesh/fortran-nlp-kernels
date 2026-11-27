program benchmark_ip
    use, intrinsic :: iso_fortran_env, only: real32
    use fnlp_vector_search_core
    implicit none

    integer, parameter :: d = 768
    integer, parameter :: niter = 10000000
    real(real32), allocatable :: x(:), y(:)
    real(real32) :: sim
    integer :: i

    allocate(x(d), y(d))

    do i = 1, d
        x(i) = real(mod(17 * i, 101), real32) / 101.0_real32
        y(i) = real(mod(19 * i, 103), real32) / 103.0_real32
    end do

    do i = 1, niter
        call inner_product(x, y, sim)
    end do

    print *, sim
end program benchmark_ip
