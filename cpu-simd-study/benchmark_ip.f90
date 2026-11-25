program benchmark_ip
    use, intrinsic :: iso_fortran_env, only: real32
    use fnlp_vector_search_core
    implicit none

    integer, parameter :: d = 768
    integer, parameter :: niter = 100000000
    real(real32), allocatable :: x(:), y(:)
    real(real32) :: sim
    integer :: i

    allocate(x(d), y(d))

    call random_number(x)
    call random_number(y)

    do i = 1, niter
        call inner_product(x, y, sim)
    end do

    print *, sim
end program benchmark_ip
