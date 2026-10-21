module fnlp_vector_search_c_api
    use, intrinsic :: iso_c_binding, only: c_float, c_int, c_int64_t
    use fnlp_vector_search_core, only: l2_search_serial, l2_search_openmp, ip_search_serial, ip_search_openmp
    use fnlp_vector_search_blas, only: l2_search_blas, ip_search_blas
    implicit none
    private
    public :: fnlp_l2_search_serial
    public :: fnlp_l2_search_openmp
    public :: fnlp_ip_search_serial
    public :: fnlp_ip_search_openmp
    public :: fnlp_l2_search_blas
    public :: fnlp_ip_search_blas

contains
    subroutine fnlp_l2_search_serial( &
        query_vector, &
        database_vectors, &
        dimension, &
        vector_count, &
        k, &
        topk_distances, &
        topk_indices &
    ) bind(C, name="fnlp_l2_search_serial")
        integer(c_int), value, intent(in) :: dimension
        integer(c_int), value, intent(in) :: vector_count
        integer(c_int), value, intent(in) :: k
        real(c_float), intent(in) :: query_vector(dimension)
        real(c_float), intent(in) :: database_vectors(dimension, vector_count)
        real(c_float), intent(out) :: topk_distances(k)
        integer(c_int64_t), intent(out) :: topk_indices(k)

        call l2_search_serial( &
            query_vector, &
            database_vectors, &
            k, &
            topk_distances, &
            topk_indices &
        )
    end subroutine fnlp_l2_search_serial

    subroutine fnlp_l2_search_openmp( &
        query_vector, &
        database_vectors, &
        dimension, &
        vector_count, &
        k, &
        topk_distances, &
        topk_indices &
    ) bind(C, name="fnlp_l2_search_openmp")
        integer(c_int), value, intent(in) :: dimension
        integer(c_int), value, intent(in) :: vector_count
        integer(c_int), value, intent(in) :: k
        real(c_float), intent(in) :: query_vector(dimension)
        real(c_float), intent(in) :: database_vectors(dimension, vector_count)
        real(c_float), intent(out) :: topk_distances(k)
        integer(c_int64_t), intent(out) :: topk_indices(k)

        call l2_search_openmp( &
            query_vector, &
            database_vectors, &
            k, &
            topk_distances, &
            topk_indices &
        )
    end subroutine fnlp_l2_search_openmp

    subroutine fnlp_ip_search_serial( &
        query_vector, &
        database_vectors, &
        dimension, &
        vector_count, &
        k, &
        topk_similarities, &
        topk_indices &
    ) bind(C, name="fnlp_ip_search_serial")
        integer(c_int), value, intent(in) :: dimension
        integer(c_int), value, intent(in) :: vector_count
        integer(c_int), value, intent(in) :: k
        real(c_float), intent(in) :: query_vector(dimension)
        real(c_float), intent(in) :: database_vectors(dimension, vector_count)
        real(c_float), intent(out) :: topk_similarities(k)
        integer(c_int64_t), intent(out) :: topk_indices(k)

        call ip_search_serial( &
            query_vector, &
            database_vectors, &
            k, &
            topk_similarities, &
            topk_indices &
        )
    end subroutine fnlp_ip_search_serial

    subroutine fnlp_ip_search_openmp( &
        query_vector, &
        database_vectors, &
        dimension, &
        vector_count, &
        k, &
        topk_similarities, &
        topk_indices &
    ) bind(C, name="fnlp_ip_search_openmp")
        integer(c_int), value, intent(in) :: dimension
        integer(c_int), value, intent(in) :: vector_count
        integer(c_int), value, intent(in) :: k
        real(c_float), intent(in) :: query_vector(dimension)
        real(c_float), intent(in) :: database_vectors(dimension, vector_count)
        real(c_float), intent(out) :: topk_similarities(k)
        integer(c_int64_t), intent(out) :: topk_indices(k)

        call ip_search_openmp( &
            query_vector, &
            database_vectors, &
            k, &
            topk_similarities, &
            topk_indices &
        )
    end subroutine fnlp_ip_search_openmp

    ! BLAS implementation of l2 search
    subroutine fnlp_l2_search_blas( &
        query_vector, &
        database_vectors, &
        dimension, &
        vector_count, &
        k, &
        topk_distances, &
        topk_indices &
    ) bind(C, name="fnlp_l2_search_blas")
        integer(c_int), value, intent(in) :: dimension
        integer(c_int), value, intent(in) :: vector_count
        integer(c_int), value, intent(in) :: k
        real(c_float), intent(in) :: query_vector(dimension)
        real(c_float), intent(in) :: database_vectors(dimension, vector_count)
        real(c_float), intent(out) :: topk_distances(k)
        integer(c_int64_t), intent(out) :: topk_indices(k)

        call l2_search_blas( &
            query_vector, &
            database_vectors, &
            k, &
            topk_distances, &
            topk_indices &
        )
    end subroutine fnlp_l2_search_blas

    ! BLAS implementation of ip search
    subroutine fnlp_ip_search_blas( &
        query_vector, &
        database_vectors, &
        dimension, &
        vector_count, &
        k, &
        topk_similarities, &
        topk_indices &
    ) bind(C, name="fnlp_ip_search_blas")
        integer(c_int), value, intent(in) :: dimension
        integer(c_int), value, intent(in) :: vector_count
        integer(c_int), value, intent(in) :: k
        real(c_float), intent(in) :: query_vector(dimension)
        real(c_float), intent(in) :: database_vectors(dimension, vector_count)
        real(c_float), intent(out) :: topk_similarities(k)
        integer(c_int64_t), intent(out) :: topk_indices(k)

        call ip_search_blas( &
            query_vector, &
            database_vectors, &
            k, &
            topk_similarities, &
            topk_indices &
        )
    end subroutine fnlp_ip_search_blas

end module fnlp_vector_search_c_api