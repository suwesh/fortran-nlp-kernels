program benchmark_sgemm
    use , intrinsic :: iso_fortran_env, only: real32
    implicit none

    interface
        subroutine sgemm( &
            transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            import :: real32
            character(len=1), intent(in) :: transa
            character(len=1), intent(in) :: transb
            integer, intent(in) :: m
            integer, intent(in) :: n
            integer, intent(in) :: k
            real(real32), intent(in) :: alpha
            integer, intent(in) :: lda
            real(real32), intent(in) :: a(lda, *)
            integer, intent(in) :: ldb
            real(real32), intent(in) :: b(ldb, *)
            real(real32), intent(in) :: beta
            integer, intent(in) :: ldc
            real(real32), intent(inout) :: c(ldc, *)
        end subroutine sgemm
    end interface

    integer, parameter :: d = 768
    integer, parameter :: n = 768
    integer, parameter :: k = 768
    integer, parameter :: niter = 10000
    real(real32), allocatable :: A(:,:), B(:,:), C(:,:)
    integer :: i

    allocate(A(m, k))
    allocate(B(k, n))
    allocate(C(m, n))

    call random_number(A)
    call random_number(B)

    C = 0.0_real32

    do i = 1, niter
        call sgemm('N', 'N', m, n, k, 1.0_real32, A, m, B, k, 0.0_real32, C, m)
    end do

    print *, C(1, 1)
end program benchmark_sgemm
