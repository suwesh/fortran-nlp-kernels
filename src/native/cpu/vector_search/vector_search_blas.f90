module fnlp_vector_search_blas
    use, intrinsic :: iso_fortran_env, only: real32, int64
    use fnlp_topk_core, only: topk_largest, topk_smallest
    implicit none
    private
    public :: ip_search_blas
    public :: l2_search_blas

    interface
        subroutine sgemv( &
            trans, &
            m, &
            n, &
            alpha, &
            matrix, &
            leading_dimension, &
            vector, &
            vector_increment, &
            beta, &
            result, &
            result_increment &
        )
            import :: real32
            character(len=1), intent(in) :: trans
            integer, intent(in) :: m
            integer, intent(in) :: n
            real(real32), intent(in) :: alpha
            integer, intent(in) :: leading_dimension
            real(real32), intent(in) :: matrix(leading_dimension, *)
            real(real32), intent(in) :: vector(*)
            integer, intent(in) :: vector_increment
            real(real32), intent(in) :: beta
            real(real32), intent(inout) :: result(*)
            integer, intent(in) :: result_increment
        end subroutine sgemv
    end interface

contains
    ! Exact inner-product search using BLAS SGEMV.
    ! similarities = database_vectors**T x query_vector
    subroutine ip_search_blas( &
        query_vector, &
        database_vectors, &
        k, &
        topk_similarities, &
        topk_indices &
    )
        real(real32), intent(in) :: query_vector(:)
        real(real32), intent(in) :: database_vectors(:,:)
        integer, intent(in) :: k
        real(real32), intent(out) :: topk_similarities(k)
        integer(int64), intent(out) :: topk_indices(k)

        real(real32), allocatable :: similarities(:)
        integer :: dimension
        integer :: vector_count

        dimension = size(query_vector)
        vector_count = size(database_vectors, dim=2)

        call validate_search_inputs( &
            query_vector, &
            database_vectors, &
            k &
        )
        allocate(similarities(vector_count))
        call sgemv( &
            "T", &
            dimension, &
            vector_count, &
            1.0_real32, &
            database_vectors, &
            dimension, &
            query_vector, &
            1, &
            0.0_real32, &
            similarities, &
            1 &
        )
        call topk_largest( &
            similarities, &
            k, &
            topk_similarities, &
            topk_indices &
        )
        deallocate(similarities)
    end subroutine ip_search_blas

    ! Exact squared-L2 search using BLAS SGEMV.
    ! ||q - x||^2 =
    ! ||q||^2 + ||x||^2 - 2(q**T x)
    subroutine l2_search_blas( &
        query_vector, &
        database_vectors, &
        k, &
        topk_distances, &
        topk_indices &
    )
        real(real32), intent(in) :: query_vector(:)
        real(real32), intent(in) :: database_vectors(:,:)
        integer, intent(in) :: k
        real(real32), intent(out) :: topk_distances(k)
        integer(int64), intent(out) :: topk_indices(k)

        real(real32), allocatable :: inner_products(:)
        real(real32), allocatable :: distances(:)
        real(real32) :: query_norm_squared
        real(real32) :: database_norm_squared
        integer :: dimension
        integer :: vector_count
        integer :: vector_index

        dimension = size(query_vector)
        vector_count = size(database_vectors, dim=2)

        call validate_search_inputs( &
            query_vector, &
            database_vectors, &
            k &
        )
        allocate(inner_products(vector_count))
        allocate(distances(vector_count))
        call sgemv( &
            "T", &
            dimension, &
            vector_count, &
            1.0_real32, &
            database_vectors, &
            dimension, &
            query_vector, &
            1, &
            0.0_real32, &
            inner_products, &
            1 &
        )
        query_norm_squared = sum(query_vector * query_vector)
        do vector_index = 1, vector_count
            database_norm_squared = sum( &
                database_vectors(:, vector_index) * &
                database_vectors(:, vector_index) &
            )
            distances(vector_index) = &
                query_norm_squared + &
                database_norm_squared - &
                2.0_real32 * inner_products(vector_index)
            ! Protect against tiny negative values from rounding.
            distances(vector_index) = max( &
                distances(vector_index), &
                0.0_real32 &
            )
        end do
        call topk_smallest( &
            distances, &
            k, &
            topk_distances, &
            topk_indices &
        )

        deallocate(inner_products)
        deallocate(distances)

    end subroutine l2_search_blas

    subroutine validate_search_inputs( &
        query_vector, &
        database_vectors, &
        k &
    )
        real(real32), intent(in) :: query_vector(:)
        real(real32), intent(in) :: database_vectors(:, :)
        integer, intent(in) :: k

        integer(int64) :: dimension
        integer(int64) :: vector_count

        dimension = size(query_vector, kind=int64)

        vector_count = size( &
            database_vectors, &
            dim=2, &
            kind=int64 &
        )

        if (dimension < 1_int64) then
            error stop &
                "Vector dimension must be greater than zero."
        end if

        if (vector_count < 1_int64) then
            error stop &
                "Database must contain at least one vector."
        end if

        if ( &
            size(database_vectors, dim=1, kind=int64) &
            /= dimension &
        ) then
            error stop &
                "Query and database vectors must have equal dimensions."
        end if

        if (k < 1) then
            error stop "k must be at least 1."
        end if
        if (int(k, int64) > vector_count) then
            error stop "k cannot exceed the database size."
        end if
    end subroutine validate_search_inputs

end module fnlp_vector_search_blas