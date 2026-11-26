program benchmark_nativesgemm
    use, intrinsic :: iso_fortran_env, only: real32
    implicit none

    integer, parameter :: n = 512
    integer, parameter :: niter = 500
    real(real32), allocatable :: A(:,:), B(:,:), C(:,:)
    integer :: iter
    integer :: i
    integer :: j
    integer :: k

    allocate(A(n, n))
    allocate(B(n, n))
    allocate(C(n, n))

    call random_number(A)
    call random_number(B)

    do iter = 1, niter
        C = 0.0_real32
        do j = 1, n
            do k = 1, n
                do i = 1, n
                    C(i, j) = C(i, j) + A(i, k) * B(k, j)
                end do
            end do
        end do
    end do 

    print *, C(1, 1)
end program benchmark_nativesgemm