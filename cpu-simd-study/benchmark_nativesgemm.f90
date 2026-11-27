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

    do j = 1, n
        do i = 1, n
            A(i, j) = real(mod(17 * i + 13 * j, 101), real32) / 101.0_real32
            B(i, j) = real(mod(19 * i + 11 * j, 103), real32) / 103.0_real32
        end do
    end do

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

    print *, sum(C)
end program benchmark_nativesgemm