Yes, passing:

1_int64


preserves the original single-query behavior exactly.

The generalized helper calculates:

query_start = (
    query_vector_index - 1_int64
) * dimension


For single query:

query_vector_index = 1
query_start = (1 - 1) × dimension = 0


Therefore:

query_element = 0 + dimension_index


which is identical to the previous:

query_vector(dimension_index)


Your passing tests confirm there was no behavior change.

Keep single-query kernels separate

Do not extend l2_database_kernel() and ip_database_kernel() with batch parameters.

Keep:

l2_database_kernel
ip_database_kernel


optimized and readable for one query.

Add separate batch kernels:

l2_database_batch_kernel
ip_database_batch_kernel


Both sets will reuse the generalized device helpers.

The structure becomes:

Device helpers:
- squared_l2_distance_device
- inner_product_device

Single-query kernels:
- l2_database_kernel
- ip_database_kernel

Batch kernels:
- l2_database_batch_kernel
- ip_database_batch_kernel


This avoids forcing single-query execution to carry unused query_count, pair-index calculations, division, and modulo operations.

Next: L2 batch kernel

Add this after ip_database_kernel():

    ! Score every query vector against every database vector
    ! using squared L2 distance.
    !
    ! One CUDA thread processes one query-database pair.
    attributes(global) subroutine l2_database_batch_kernel( &
        query_vectors, &
        database_vectors, &
        dimension, &
        query_count, &
        database_vector_count, &
        distances &
    )
        real(real32), device, intent(in) :: query_vectors(*)
        real(real32), device, intent(in) :: database_vectors(*)

        integer(int64), value, intent(in) :: dimension
        integer(int64), value, intent(in) :: query_count
        integer(int64), value, intent(in) :: &
            database_vector_count

        real(real32), device, intent(out) :: distances(*)

        integer(int64) :: pair_count
        integer(int64) :: pair_index
        integer(int64) :: query_vector_index
        integer(int64) :: database_vector_index

        pair_count = query_count * database_vector_count

        pair_index = int( &
            (blockIdx%x - 1) * blockDim%x + threadIdx%x, &
            kind=int64 &
        )

        if (pair_index > pair_count) then
            return
        end if

        query_vector_index = &
            (pair_index - 1_int64) &
            / database_vector_count &
            + 1_int64

        database_vector_index = modulo( &
            pair_index - 1_int64, &
            database_vector_count &
        ) + 1_int64

        call squared_l2_distance_device( &
            query_vectors, &
            database_vectors, &
            dimension, &
            query_vector_index, &
            database_vector_index, &
            distances(pair_index) &
        )
    end subroutine l2_database_batch_kernel

Pair mapping

Suppose:

query_count = 2
database_vector_count = 3


Then:

pair_index 1 -> query 1, database vector 1
pair_index 2 -> query 1, database vector 2
pair_index 3 -> query 1, database vector 3
pair_index 4 -> query 2, database vector 1
pair_index 5 -> query 2, database vector 2
pair_index 6 -> query 2, database vector 3


The flattened output order corresponds to the Fortran matrix:

distances(database_vector_count, query_count)


because the first dimension varies fastest:

all database vectors for query 1
all database vectors for query 2
...

Why separate kernels are cleaner

Single query:

thread index directly equals database_vector_index


Batch:

thread index equals flattened pair_index
pair_index must be decoded into:
- query_vector_index
- database_vector_index


They share the actual pairwise L2 arithmetic, but keep different work-mapping logic.

After this compiles, write the matching:

ip_database_batch_kernel


with the same pair mapping and a call to:

inner_product_device(...)