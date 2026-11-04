Yes. We’ll first implement the two private GPU kernels that make the cuBLAS L2 path genuinely GPU-native:

squared_norms_kernel
l2_from_inner_products_kernel


These kernels surround cuBLAS:

squared_norms_kernel
        ↓
cuBLAS SGEMV / SGEMM
        ↓
l2_from_inner_products_kernel


No CPU norm calculation or CPU L2 combination is performed. Only final top-k remains on the CPU.

1. squared_norms_kernel
Purpose

Given a matrix:

vectors(dimension, vector_count)


calculate:

squared_norms(vector_index)
=
sum(vectors(:, vector_index) ** 2)


One CUDA thread processes one complete vector.

Add this after contains:

    ! Calculate one squared L2 norm for each vector.
    !
    ! vectors is logically shaped as:
    !     (dimension, vector_count)
    !
    ! One CUDA thread processes one vector.
    attributes(global) subroutine squared_norms_kernel( &
        vectors, &
        dimension, &
        vector_count, &
        squared_norms &
    )
        real(real32), device, intent(in) :: vectors(*)

        integer(int64), value, intent(in) :: dimension
        integer(int64), value, intent(in) :: vector_count

        real(real32), device, intent(out) :: squared_norms(*)

        integer(int64) :: vector_index
        integer(int64) :: vector_start
        integer(int64) :: dimension_index
        integer(int64) :: vector_element

        real(real32) :: value
        real(real32) :: squared_norm

Thread assignment
        vector_index = int( &
            (blockIdx%x - 1) * blockDim%x + threadIdx%x, &
            kind=int64 &
        )

        if (vector_index > vector_count) then
            return
        end if


This maps:

thread 1 -> vector 1
thread 2 -> vector 2
thread 3 -> vector 3

Find the assigned vector
        vector_start = &
            (vector_index - 1_int64) * dimension


For:

dimension = 3
vector_index = 2


the flattened storage is:

Vector 1: positions 1, 2, 3
Vector 2: positions 4, 5, 6


Therefore:

vector_start = 3


and the second vector begins at:

vector_start + 1 = 4

Calculate the squared norm
        squared_norm = 0.0_real32

        do dimension_index = 1_int64, dimension
            vector_element = &
                vector_start + dimension_index

            value = vectors(vector_element)

            squared_norm = squared_norm + value * value
        end do


This calculates:

∥x∥2=∑dxd2\|x\|^2 = \sum_d x_d^2
Store the result
        squared_norms(vector_index) = squared_norm
    end subroutine squared_norms_kernel

Complete first kernel
    ! Calculate one squared L2 norm for each vector.
    !
    ! vectors is logically shaped as:
    !     (dimension, vector_count)
    !
    ! One CUDA thread processes one vector.
    attributes(global) subroutine squared_norms_kernel( &
        vectors, &
        dimension, &
        vector_count, &
        squared_norms &
    )
        real(real32), device, intent(in) :: vectors(*)

        integer(int64), value, intent(in) :: dimension
        integer(int64), value, intent(in) :: vector_count

        real(real32), device, intent(out) :: squared_norms(*)

        integer(int64) :: vector_index
        integer(int64) :: vector_start
        integer(int64) :: dimension_index
        integer(int64) :: vector_element

        real(real32) :: value
        real(real32) :: squared_norm

        vector_index = int( &
            (blockIdx%x - 1) * blockDim%x + threadIdx%x, &
            kind=int64 &
        )

        if (vector_index > vector_count) then
            return
        end if

        vector_start = &
            (vector_index - 1_int64) * dimension

        squared_norm = 0.0_real32

        do dimension_index = 1_int64, dimension
            vector_element = &
                vector_start + dimension_index

            value = vectors(vector_element)

            squared_norm = squared_norm + value * value
        end do

        squared_norms(vector_index) = squared_norm
    end subroutine squared_norms_kernel

Difference from the CPU BLAS code

CPU:

database_norm_squared = sum( &
    database_vectors(:, vector_index) * &
    database_vectors(:, vector_index) &
)


GPU:

one thread is assigned to each vector
each thread computes one squared norm
all vector norms execute concurrently


This kernel can process either:

database_vectors(dimension, database_vector_count)


or:

query_vectors(dimension, query_count)


It is intentionally generic.

2. l2_from_inner_products_kernel
Purpose

cuBLAS will produce the inner-product matrix:

inner_products(database_vector_count, query_count)


This kernel converts it into squared-L2 distances:

∥qj−xi∥2=∥xi∥2+∥qj∥2−2xiTqj\|q_j-x_i\|^2 = \|x_i\|^2 + \|q_j\|^2 - 2x_i^Tq_j

One CUDA thread processes one database-query pair.

Add this after squared_norms_kernel:

    ! Convert inner products and squared norms into squared L2
    ! distances.
    !
    ! inner_products and distances are logically shaped as:
    !     (database_vector_count, query_count)
    !
    ! One CUDA thread processes one query-database pair.
    attributes(global) subroutine l2_from_inner_products_kernel( &
        inner_products, &
        database_squared_norms, &
        query_squared_norms, &
        database_vector_count, &
        query_count, &
        distances &
    )

Inputs and output
        real(real32), device, intent(in) :: inner_products(*)

        real(real32), device, intent(in) :: &
            database_squared_norms(*)

        real(real32), device, intent(in) :: &
            query_squared_norms(*)

        integer(int64), value, intent(in) :: &
            database_vector_count

        integer(int64), value, intent(in) :: query_count

        real(real32), device, intent(out) :: distances(*)


The inputs are:

inner_products(pair_index)
    qᵀx from SGEMV or SGEMM

database_squared_norms(database_vector_index)
    ||x||²

query_squared_norms(query_vector_index)
    ||q||²

Local indexes
        integer(int64) :: pair_count
        integer(int64) :: pair_index
        integer(int64) :: query_vector_index
        integer(int64) :: database_vector_index

        real(real32) :: distance

Map one thread to one pair
        pair_count = &
            database_vector_count * query_count

        pair_index = int( &
            (blockIdx%x - 1) * blockDim%x + threadIdx%x, &
            kind=int64 &
        )

        if (pair_index > pair_count) then
            return
        end if


This uses the same flattened ordering as handwritten batched scoring.

Recover query and database indexes
        query_vector_index = &
            (pair_index - 1_int64) &
            / database_vector_count &
            + 1_int64

        database_vector_index = modulo( &
            pair_index - 1_int64, &
            database_vector_count &
        ) + 1_int64


Example:

database_vector_count = 3
pair_index = 5


produces:

query_vector_index = 2
database_vector_index = 2

Apply norm expansion
        distance = &
            database_squared_norms(database_vector_index) &
            + query_squared_norms(query_vector_index) &
            - 2.0_real32 * inner_products(pair_index)


This is the GPU equivalent of the CPU code:

distances(vector_index) = &
    query_norm_squared + &
    database_norm_squared - &
    2.0_real32 * inner_products(vector_index)

Clamp rounding artifacts

The CPU BLAS implementation does:

distance = max(distance, 0.0_real32)


Keep the same behavior:

        distances(pair_index) = max( &
            distance, &
            0.0_real32 &
        )


Why? Floating-point rounding can produce values such as:

-0.000000238


for vectors that should have exact squared distance zero.

Close the kernel
    end subroutine l2_from_inner_products_kernel

Complete second kernel
    ! Convert inner products and squared norms into squared L2
    ! distances.
    !
    ! inner_products and distances are logically shaped as:
    !     (database_vector_count, query_count)
    !
    ! One CUDA thread processes one query-database pair.
    attributes(global) subroutine l2_from_inner_products_kernel( &
        inner_products, &
        database_squared_norms, &
        query_squared_norms, &
        database_vector_count, &
        query_count, &
        distances &
    )
        real(real32), device, intent(in) :: inner_products(*)

        real(real32), device, intent(in) :: &
            database_squared_norms(*)

        real(real32), device, intent(in) :: &
            query_squared_norms(*)

        integer(int64), value, intent(in) :: &
            database_vector_count

        integer(int64), value, intent(in) :: query_count

        real(real32), device, intent(out) :: distances(*)

        integer(int64) :: pair_count
        integer(int64) :: pair_index
        integer(int64) :: query_vector_index
        integer(int64) :: database_vector_index

        real(real32) :: distance

        pair_count = &
            database_vector_count * query_count

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

        distance = &
            database_squared_norms(database_vector_index) &
            + query_squared_norms(query_vector_index) &
            - 2.0_real32 * inner_products(pair_index)

        ! Protect against tiny negative values from rounding.
        distances(pair_index) = max( &
            distance, &
            0.0_real32 &
        )
    end subroutine l2_from_inner_products_kernel

Why this differs from handwritten L2
Handwritten kernel
for each query-database pair:
    loop over dimensions
    calculate (q[d] - x[d])²
    accumulate distance directly

cuBLAS L2 path
squared_norms_kernel:
    calculate ||q||² and ||x||²

cuBLAS:
    calculate qᵀx

l2_from_inner_products_kernel:
    calculate ||q||² + ||x||² - 2qᵀx


The cuBLAS design decomposes L2 scoring because cuBLAS is optimized for dot-product matrix operations, not direct pairwise difference-squared calculations.

Compile-check

Temporarily keep the module without undefined public declarations, then compile:

nvfortran \
  -c \
  -cuda \
  -gpu=cc86,cuda13.3 \
  -O3 \
  src/native/cuda/vector_search/vector_search_cublas.cuf \
  -I build/cuda/vector_search \
  -module build/cuda/vector_search \
  -o build/cuda/vector_search/vector_search_cublas.o


Once these compile, the next procedure is not another kernel. It is the first cuBLAS host wrapper:

ip_database_cublas


That wrapper will teach:

cuBLAS handle creation;
device allocation;
SGEMV transpose mode;
matrix dimensions;
leading dimension;
alpha and beta;
cuBLAS status checking;
copying the similarity array back to the host.
====

## Maths:

Yes, exactly. Let’s ignore CUDA and code for a moment and understand the maths using ordinary 2D points.

1. Direct squared-L2 distance

Consider two points:

A=(x1,y1)A=(x_1,y_1) B=(x2,y2)B=(x_2,y_2)

The ordinary Euclidean distance is:

(x2−x1)2+(y2−y1)2\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}

Vector search usually uses squared L2 distance, which removes the square root:

(x2−x1)2+(y2−y1)2(x_2-x_1)^2+(y_2-y_1)^2

Removing the square root does not change the result ordering because square root is increasing:

distance² = 4  -> distance = 2
distance² = 9  -> distance = 3


The smaller squared distance still represents the nearer point.

Example

Let:

A=(1,2)A=(1,2) B=(4,6)B=(4,6)

Then:

∥A−B∥2=(4−1)2+(6−2)2\|A-B\|^2=(4-1)^2+(6-2)^2 =32+42=3^2+4^2 =9+16=25=9+16=25

This is what the handwritten CUDA kernel calculates directly:

difference = query_value - database_value
distance = distance + difference * difference


For two dimensions:

difference x = 1 - 4 = -3
difference y = 2 - 6 = -4

squared distance = (-3)² + (-4)²
                 = 9 + 16
                 = 25

2. Expanding the same formula

Start with:

(x2−x1)2+(y2−y1)2(x_2-x_1)^2+(y_2-y_1)^2

Using:

(a−b)2=a2+b2−2ab(a-b)^2=a^2+b^2-2ab

expand both terms:

(x22+x12−2x1x2)+(y22+y12−2y1y2)(x_2^2+x_1^2-2x_1x_2) + (y_2^2+y_1^2-2y_1y_2)

Rearrange:

(x12+y12)+(x22+y22)−2(x1x2+y1y2)(x_1^2+y_1^2) + (x_2^2+y_2^2) - 2(x_1x_2+y_1y_2)

Now these three sections have familiar meanings.

First section: squared norm of A
x12+y12=∥A∥2x_1^2+y_1^2=\|A\|^2
Second section: squared norm of B
x22+y22=∥B∥2x_2^2+y_2^2=\|B\|^2
Third section: inner product
x1x2+y1y2=ATBx_1x_2+y_1y_2=A^TB

Therefore:

∥A−B∥2=∥A∥2+∥B∥2−2ATB\boxed{ \|A-B\|^2 = \|A\|^2+\|B\|^2-2A^TB }

It is exactly the same distance formula, only rearranged.

3. Verify using the same example

We had:

A=(1,2),B=(4,6)A=(1,2),\qquad B=(4,6)
A’s squared norm
∥A∥2=12+22=1+4=5\|A\|^2=1^2+2^2=1+4=5
B’s squared norm
∥B∥2=42+62=16+36=52\|B\|^2=4^2+6^2=16+36=52
Inner product
ATB=(1×4)+(2×6)A^TB=(1\times4)+(2\times6) =4+12=16=4+12=16
Combine
∥A−B∥2=5+52−2(16)\|A-B\|^2=5+52-2(16) =57−32=25=57-32=25

Same answer as the direct method:

(4−1)2+(6−2)2=25(4-1)^2+(6-2)^2=25
4. Why cuBLAS needs the expanded formula

Your handwritten kernel can directly calculate:

(qd−xd)2(q_d-x_d)^2

for every dimension.

But cuBLAS SGEMV and SGEMM are built to perform multiplication and accumulation efficiently. They naturally calculate the inner-product portion:

qTxq^Tx

They do not directly calculate:

∑d(qd−xd)2\sum_d(q_d-x_d)^2

So we break L2 into pieces that fit GPU operations:

squared_norms_kernel:
    calculate ||q||²
    calculate ||x||²

cuBLAS:
    calculate qᵀx

l2_from_inner_products_kernel:
    calculate ||q||² + ||x||² - 2qᵀx

5. Database example

Suppose the query is:

q=(1,2)q=(1,2)

and the database contains:

x1=(4,6)x_1=(4,6) x2=(2,2)x_2=(2,2) x3=(0,0)x_3=(0,0)
GPU squared-norm kernel

The query norm:

∥q∥2=12+22=5\|q\|^2=1^2+2^2=5

Database norms:

∥x1∥2=42+62=52\|x_1\|^2=4^2+6^2=52 ∥x2∥2=22+22=8\|x_2\|^2=2^2+2^2=8 ∥x3∥2=0\|x_3\|^2=0

So the GPU produces:

query_squared_norms   = [5]
database_squared_norms = [52, 8, 0]

cuBLAS SGEMV

cuBLAS calculates all query-database inner products:

qTx1=(1×4)+(2×6)=16q^Tx_1=(1\times4)+(2\times6)=16 qTx2=(1×2)+(2×2)=6q^Tx_2=(1\times2)+(2\times2)=6 qTx3=0q^Tx_3=0

So:

inner_products = [16, 6, 0]

L2-combination kernel

For database vector 1:

5+52−2(16)=255+52-2(16)=25

For database vector 2:

5+8−2(6)=15+8-2(6)=1

For database vector 3:

5+0−2(0)=55+0-2(0)=5

Final distances:

[25, 1, 5]


Therefore, the nearest vector is:

x2 with squared distance 1

6. What the two private kernels do
squared_norms_kernel

One GPU thread receives one vector:

Thread 1 -> calculate ||vector 1||²
Thread 2 -> calculate ||vector 2||²
Thread 3 -> calculate ||vector 3||²


For a vector:

(a,b,c)(a,b,c)

the thread calculates:

a2+b2+c2a^2+b^2+c^2
l2_from_inner_products_kernel

One GPU thread receives one query-database pair and combines already-calculated values:

database norm
+ query norm
- 2 × inner product


For pair (xi,qj)(x_i,q_j):

distancei,j=database_normi+query_normj−2×inner_producti,jdistance_{i,j} = database\_norm_i + query\_norm_j - 2\times inner\_product_{i,j}

This second kernel does not loop over dimensions. All dimension-wise work has already been completed by:

the squared-norm kernel;
cuBLAS SGEMV or SGEMM.
7. Handwritten CUDA versus cuBLAS L2
Handwritten CUDA

Each thread calculates directly:

(q1−x1)2+(q2−x2)2+⋯(q_1-x_1)^2+(q_2-x_2)^2+\cdots
query and database vectors
→ direct difference-squared loop
→ distance

cuBLAS path

The work is decomposed:

query and database vectors
→ squared norms
→ cuBLAS dot products
→ norm expansion
→ distance


Both produce mathematically equivalent answers:

∑d(qd−xd)2=∥q∥2+∥x∥2−2qTx\sum_d(q_d-x_d)^2 = \|q\|^2+\|x\|^2-2q^Tx

The cuBLAS approach is useful because SGEMV and especially SGEMM are highly optimized for producing many dot products efficiently.

So those two private kernels are essentially the preparation and finishing steps around cuBLAS:

squared_norms_kernel
    = prepare ||q||² and ||x||²

cuBLAS
    = calculate qᵀx

l2_from_inner_products_kernel
    = combine all three into squared L2

=== 
Exactly. You’ve mapped it correctly.

1. Squared norms

For:

A=(x1,y1)A=(x_1,y_1) B=(x2,y2)B=(x_2,y_2)

these terms:

x12+y12=∥A∥2x_1^2+y_1^2=\|A\|^2 x22+y22=∥B∥2x_2^2+y_2^2=\|B\|^2

are calculated by our custom:

squared_norms_kernel


The same kernel works for both collections:

query vectors
database vectors


It produces:

query_squared_norms
database_squared_norms

2. Inner product

This term:

x1x2+y1y2=ATBx_1x_2+y_1y_2=A^TB

is computed by cuBLAS, not by one of our custom kernels.

For a single query, we call:

cublasSgemv


Conceptually:

inner_products = transpose(database_vectors) × query_vector


For a query batch, we call:

cublasSgemm


Conceptually:

inner_products =
    transpose(database_vectors) × query_vectors


So yes, the inner-product operation is prebuilt and highly optimized inside cuBLAS. We only allocate the input/output device arrays and invoke the library routine. cuBLAS internally selects and launches its GPU kernel implementation.

3. The factor of two

Small but important correction: the inner product itself is:

ATB=x1x2+y1y2A^TB=x_1x_2+y_1y_2

That corresponds to:

inner_products(pair_index)


The full norm-expansion formula requires:

−2ATB-2A^TB

That corresponds to:

-2.0_real32 * inner_products(pair_index)


So:

cuBLAS produces:
AᵀB

our L2 combination kernel applies:
-2 × AᵀB

Full mapping

The equation:

∥A−B∥2=∥A∥2+∥B∥2−2ATB\|A-B\|^2 = \|A\|^2+\|B\|^2-2A^TB

maps to:

distance = &
    query_squared_norms(query_vector_index) &
    + database_squared_norms(database_vector_index) &
    - 2.0_real32 * inner_products(pair_index)


And the ownership is:

||A||²
    -> squared_norms_kernel

||B||²
    -> squared_norms_kernel

AᵀB
    -> cuBLAS SGEMV or SGEMM

||A||² + ||B||² - 2AᵀB
    -> l2_from_inner_products_kernel

Example

For:

A=(1,2)A=(1,2) B=(4,6)B=(4,6)

The norm kernel produces:

∥A∥2=12+22=5\|A\|^2=1^2+2^2=5 ∥B∥2=42+62=52\|B\|^2=4^2+6^2=52

cuBLAS produces:

ATB=(1×4)+(2×6)=16A^TB=(1\times4)+(2\times6)=16

The L2 combination kernel produces:

5+52−2(16)=255+52-2(16)=25

So the pipeline is:

Custom norm kernel:
    [5], [52]

cuBLAS:
    [16]

Custom L2 combination kernel:
    5 + 52 - 2 × 16 = 25


You have written the preparation kernel and the finishing kernel. The part between them, SGEMV or SGEMM, is the optimized prebuilt inner-product engine we implement next.
===
Exactly. use cublas only makes the prebuilt cuBLAS API available to the Fortran module. It does not perform any multiplication until we explicitly call cublasSgemv or cublasSgemm.

What we have written
1. squared_norms_kernel

For each vector, this computes:

∥A∥2=x12+y12\|A\|^2=x_1^2+y_1^2

and:

∥B∥2=x22+y22\|B\|^2=x_2^2+y_2^2

For higher-dimensional vectors:

∥A∥2=∑dAd2\|A\|^2=\sum_d A_d^2

The same kernel can process both the query collection and database collection:

squared_norms_kernel(query_vectors)
→ query_squared_norms

squared_norms_kernel(database_vectors)
→ database_squared_norms

2. l2_from_inner_products_kernel

This combines the norms with the inner products:

∥A−B∥2=∥A∥2+∥B∥2−2ATB\|A-B\|^2 = \|A\|^2+\|B\|^2-2A^TB

In code:

distance = &
    database_squared_norms(database_vector_index) &
    + query_squared_norms(query_vector_index) &
    - 2.0_real32 * inner_products(pair_index)


But inner_products must first be filled by cuBLAS.

What remains between those kernels

The pipeline currently has a missing middle:

squared_norms_kernel
    ↓
query and database squared norms available

cuBLAS SGEMV/SGEMM
    ↓
inner_products available

l2_from_inner_products_kernel
    ↓
final squared-L2 distances available


We have written the first and third operations. Next we write the host wrapper that explicitly invokes the prebuilt cuBLAS operation.

use cublas versus calling cuBLAS

This:

use cublas


is comparable to importing a Python library:

import numpy as np


Importing NumPy does not multiply anything. You still call:

np.matmul(a, b)


Similarly:

use cublas


provides access to items such as:

cuBLAS handle types
cuBLAS status values
transpose constants
cublasCreate
cublasDestroy
cublasSgemv
cublasSgemm


We still need to call:

status = cublasSgemv(...)


or:

status = cublasSgemm(...)


cuBLAS then internally chooses and launches NVIDIA’s optimized GPU kernel. cuBLAS provides GPU implementations of BLAS Level 2 matrix-vector and Level 3 matrix-matrix operations.

Single-query path

For one query:

DTqD^Tq

we will call:

SGEMV


It produces:

inner_products(database_vector_count)


Then the L2 combination kernel produces:

distances(database_vector_count)

Batched path

For multiple queries:

DTQD^TQ

we will call:

SGEMM


It produces:

inner_products(
    database_vector_count,
    query_count
)


Then the L2 combination kernel produces:

distances(
    database_vector_count,
    query_count
)

Is this kernel optimization?

Yes. More precisely, it is algorithm decomposition and heterogeneous kernel composition.

Instead of writing one kernel that directly calculates:

∑d(qd−xd)2\sum_d(q_d-x_d)^2

we decompose the calculation into operations that match efficient GPU building blocks:

custom CUDA kernel:
    vector squared norms

cuBLAS:
    optimized bulk dot products

custom CUDA kernel:
    final elementwise L2 combination


That is a core HPC skill: reshaping the mathematics so most of the work can use an optimized library primitive, then writing only the surrounding specialized kernels yourself.

The next step is implementing:

ip_database_cublas


This will teach the actual SGEMV call and produce the currently missing inner_products array.