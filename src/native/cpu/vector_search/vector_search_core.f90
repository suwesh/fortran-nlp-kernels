module fnlp_vector_search_core
    use, intrinsic :: iso_fortran_env, only: real32, int64
    use fnlp_topk_core, only: topk_smallest, topk_largest
    implicit none
    private
    public :: squared_l2_distance
    public :: inner_product
    public :: l2_database_serial
    public :: ip_database_serial
    public :: l2_database_openmp
    public :: ip_database_openmp
    public :: l2_search_serial
    public :: l2_search_openmp
    public :: ip_search_serial
    public :: ip_search_openmp

contains
    ! One query vector x one database vector.
    subroutine squared_l2_distance( &
        query_vector, &
        database_vector, &
        distance &
    )
        real(real32), intent(in) :: query_vector(:)
        real(real32), intent(in) :: database_vector(:)
        real(real32), intent(out) :: distance

        integer(int64) :: dimension
        integer(int64) :: dimension_index
        real(real32) :: difference

        dimension = size(query_vector, kind=int64)

        if (size(database_vector, kind=int64) /= dimension) then
            error stop "Query and database vectors must have equal dimensions."
        end if

        distance = 0.0_real32

        do dimension_index = 1_int64, dimension
            difference = &
                query_vector(dimension_index) - &
                database_vector(dimension_index)
            distance = distance + difference * difference
        end do
    end subroutine squared_l2_distance

    ! One query vector x one database vector.
    subroutine inner_product( &
        query_vector, &
        database_vector, &
        similarity &
    )
        real(real32), intent(in) :: query_vector(:)
        real(real32), intent(in) :: database_vector(:)
        real(real32), intent(out) :: similarity

        integer(int64) :: dimension
        integer(int64) :: dimension_index

        dimension = size(query_vector, kind=int64)

        if (size(database_vector, kind=int64) /= dimension) then
            error stop "Query and database vectors must have equal dimensions."
        end if

        similarity = 0.0_real32

        do dimension_index = 1_int64, dimension
            similarity = similarity + &
                query_vector(dimension_index) * &
                database_vector(dimension_index)
        end do
    end subroutine inner_product

    ! One query vector x complete database using squared L2 distance - Serial
    subroutine l2_database_serial( &
        query_vector, &
        database_vectors, &
        distances &
    )
        real(real32), intent(in) :: query_vector(:)
        real(real32), intent(in) :: database_vectors(:,:)
        real(real32), intent(out) :: distances(size(database_vectors, dim=2))

        integer(int64) :: dimension
        integer(int64) :: vector_count
        integer(int64) :: vector_index

        dimension = size(query_vector, kind=int64)
        vector_count = size(database_vectors, dim=2, kind=int64)

        if (size(database_vectors, dim=1, kind=int64) /= dimension) then
            error stop "Query and database vectors must have equal dimensions."
        end if

        do vector_index = 1_int64, vector_count
            call squared_l2_distance( &
                query_vector, &
                database_vectors(:, vector_index), &
                distances(vector_index) &
            )
        end do
    end subroutine l2_database_serial

    ! One query vector x complete database using inner product. - Serial
    subroutine ip_database_serial( &
        query_vector, &
        database_vectors, &
        similarities &
    )
        real(real32), intent(in) :: query_vector(:)
        real(real32), intent(in) :: database_vectors(:,:)
        real(real32), intent(out) :: similarities(size(database_vectors, dim=2))

        integer(int64) :: dimension
        integer(int64) :: vector_count
        integer(int64) :: vector_index

        dimension = size(query_vector, kind=int64)
        vector_count = size(database_vectors, dim=2, kind=int64)

        if (size(database_vectors, dim=1, kind=int64) /= dimension) then
            error stop "Query and database vectors must have equal dimensions."
        end if

        do vector_index = 1_int64, vector_count
            call inner_product( &
                query_vector, &
                database_vectors(:, vector_index), &
                similarities(vector_index) &
            )
        end do
    end subroutine ip_database_serial

    ! One query vector x complete database using squared L2 distance - Parallel
    subroutine l2_database_openmp( &
        query_vector, &
        database_vectors, &
        distances &
    )
        real(real32), intent(in) :: query_vector(:)
        real(real32), intent(in) :: database_vectors(:,:)
        real(real32), intent(out) :: distances(size(database_vectors, dim=2))

        integer(int64) :: dimension
        integer(int64) :: vector_count
        integer(int64) :: vector_index

        dimension = size(query_vector, kind=int64)
        vector_count = size(database_vectors, dim=2, kind=int64)

        if (size(database_vectors, dim=1, kind=int64) /= dimension) then
            error stop "Query and database vectors must have equal dimensions."
        end if
        
        !$omp parallel do default(none) &
        !$omp shared(query_vector, database_vectors, distances, vector_count) &
        !$omp private(vector_index) &
        !$omp schedule(static)

        do vector_index = 1_int64, vector_count
            call squared_l2_distance( &
                query_vector, &
                database_vectors(:, vector_index), &
                distances(vector_index) &
            )
        end do
        !$omp end parallel do
    end subroutine l2_database_openmp

    ! One query vector x complete database using inner product. - Parallel
    subroutine ip_database_openmp( &
        query_vector, &
        database_vectors, &
        similarities &
    )
        real(real32), intent(in) :: query_vector(:)
        real(real32), intent(in) :: database_vectors(:,:)
        real(real32), intent(out) :: similarities(size(database_vectors, dim=2))

        integer(int64) :: dimension
        integer(int64) :: vector_count
        integer(int64) :: vector_index

        dimension = size(query_vector, kind=int64)
        vector_count = size(database_vectors, dim=2, kind=int64)

        if (size(database_vectors, dim=1, kind=int64) /= dimension) then
            error stop "Query and database vectors must have equal dimensions."
        end if

        !$omp parallel do default(none) &
        !$omp shared(query_vector, database_vectors, similarities, vector_count) &
        !$omp private(vector_index) &
        !$omp schedule(static)

        do vector_index = 1_int64, vector_count
            call inner_product( &
                query_vector, &
                database_vectors(:, vector_index), &
                similarities(vector_index) &
            )
        end do
        !$omp end parallel do
    end subroutine ip_database_openmp

    ! Exact squared L2 search using serial database scoring.
    subroutine l2_search_serial( &
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

        real(real32), allocatable :: distances(:)
        integer(int64) :: vector_count

        vector_count = size(database_vectors, dim=2, kind=int64)
        if (k < 1) then
            error stop "k must be at least 1."
        end if
        if (int(k, int64) > vector_count) then
            error stop "k cannot exceed the database size."
        end if
        
        allocate(distances(vector_count))

        call l2_database_serial( &
            query_vector, &
            database_vectors, &
            distances &
        )
        call topk_smallest( &
            distances, &
            k, &
            topk_distances, &
            topk_indices &
        )

        deallocate(distances)
    end subroutine l2_search_serial

    ! Exact squared L2 search using parallel database scoring.
    subroutine l2_search_openmp( &
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

        real(real32), allocatable :: distances(:)
        integer(int64) :: vector_count

        vector_count = size(database_vectors, dim=2, kind=int64)
        if (k < 1) then
            error stop "k must be at least 1."
        end if
        if (int(k, int64) > vector_count) then
            error stop "k cannot exceed the database size."
        end if
        
        allocate(distances(vector_count))

        call l2_database_openmp( &
            query_vector, &
            database_vectors, &
            distances &
        )
        call topk_smallest( &
            distances, &
            k, &
            topk_distances, &
            topk_indices &
        )

        deallocate(distances)
    end subroutine l2_search_openmp

    ! Exact inner-product search using serial database scoring.
    subroutine ip_search_serial( &
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
        integer(int64) :: vector_count

        vector_count = size(database_vectors, dim=2, kind=int64)
        if (k < 1) then
            error stop "k must be at least 1."
        end if
        if (int(k, int64) > vector_count) then
            error stop "k cannot exceed the database size."
        end if

        allocate(similarities(vector_count))

        call ip_database_serial( &
            query_vector, &
            database_vectors, &
            similarities &
        )
        call topk_largest( &
            similarities, &
            k, &
            topk_similarities, &
            topk_indices &
        )

        deallocate(similarities)
    end subroutine ip_search_serial

    ! Exact inner-product search using parallel database scoring.
    subroutine ip_search_openmp( &
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
        integer(int64) :: vector_count

        vector_count = size(database_vectors, dim=2, kind=int64)
        if (k < 1) then
            error stop "k must be at least 1."
        end if
        if (int(k, int64) > vector_count) then
            error stop "k cannot exceed the database size."
        end if

        allocate(similarities(vector_count))

        call ip_database_openmp( &
            query_vector, &
            database_vectors, &
            similarities &
        )
        call topk_largest( &
            similarities, &
            k, &
            topk_similarities, &
            topk_indices &
        )

        deallocate(similarities)
    end subroutine ip_search_openmp

end module fnlp_vector_search_core