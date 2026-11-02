#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "fortran_nlp_cuda_lexical.h"
#include "fortran_nlp_cuda_vector_search.h"

namespace nb = nanobind;

using Int64Array = nb::ndarray<
    nb::numpy,
    const std::int64_t,
    nb::ndim<1>,
    nb::c_contig,
    nb::device::cpu
>;
using Float32Vector = nb::ndarray<
    nb::numpy,
    const float,
    nb::ndim<1>,
    nb::c_contig,
    nb::device::cpu
>;
using Float32MatrixF = nb::ndarray<
    nb::numpy,
    const float,
    nb::ndim<2>,
    nb::f_contig,
    nb::device::cpu
>;

std::int64_t checked_int64_count(
    std::size_t count,
    const char *label
)
{
    if (
        count > static_cast<std::size_t>(
            std::numeric_limits<std::int64_t>::max()
        )
    ) {
        throw std::overflow_error(
            std::string(label) + " exceeds the C int64_t range"
        );
    }
    return static_cast<std::int64_t>(count);
}

int checked_int_count(std::size_t count, const char *label)
{
    if (
        count > static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::overflow_error(
            std::string(label) + " exceeds the C int range"
        );
    }
    return static_cast<int>(count);
}

using VectorSearchFunction = void (*)(
    const float *query_vector,
    const float *database_vectors,
    int dimension,
    int vector_count,
    int k,
    float *topk_values,
    std::int64_t *topk_indices
);

using VectorBatchSearchFunction = void (*)(
    const float *query_vectors,
    const float *database_vectors,
    int dimension,
    int query_count,
    int vector_count,
    int k,
    float *topk_values,
    std::int64_t *topk_indices
);

nb::tuple run_jaccard_cuda(
    const Int64Array &candidate,
    const Int64Array &corpus,
    const Int64Array &document_offsets
)
{
    if (document_offsets.shape(0) < 1) {
        throw std::invalid_argument(
            "document_offsets must contain at least one element"
        );
    }
    const auto *document_offset_data = document_offsets.data();
    if (document_offset_data[0] != 0) {
        throw std::invalid_argument(
            "document_offsets must begin with zero"
        );
    }
    const std::size_t document_count_size = document_offsets.shape(0) - 1;
    const int document_count = checked_int_count(document_count_size, "document count");
    const std::int64_t candidate_count = checked_int64_count(candidate.shape(0), "candidate shingle count");
    const std::int64_t corpus_count = checked_int64_count(corpus.shape(0), "corpus shingle count");
    if (
        document_offset_data[document_count] != corpus_count
    ) {
        throw std::invalid_argument(
            "final document offset must equal corpus length"
        );
    }
    for (
        int index = 1;
        index <= document_count; ++index
    ) {
        if (
            document_offset_data[index] < document_offset_data[index - 1]
        ) {
            throw std::invalid_argument(
                "document_offsets must be nondecreasing"
            );
        }
    }
    nb::module_ numpy = nb::module_::import_("numpy");

    nb::object intersections_object = numpy.attr("empty")(document_count, "int64");
    
    nb::object unions_object = numpy.attr("empty")(document_count, "int64");

    nb::object scores_object = numpy.attr("empty")(document_count, "float64");

    auto intersections = nb::cast<
        nb::ndarray<
            nb::numpy,
            std::int64_t,
            nb::ndim<1>,
            nb::c_contig,
            nb::device::cpu
        >
    >(intersections_object);

    auto unions = nb::cast<
        nb::ndarray<
            nb::numpy,
            std::int64_t,
            nb::ndim<1>,
            nb::c_contig,
            nb::device::cpu
        >
    >(unions_object);

    auto scores = nb::cast<
        nb::ndarray<
            nb::numpy,
            double,
            nb::ndim<1>,
            nb::c_contig,
            nb::device::cpu
        >
    >(scores_object);

    {
        /*
         * The CUDA Fortran wrapper performs allocation, transfers,
         * kernel execution, synchronization, and result transfer.
         * Python's GIL is not required during that native work.
         */
        nb::gil_scoped_release release;
        fnlp_jaccard_corpus_cuda(
            candidate.data(),
            candidate_count,
            corpus.data(),
            corpus_count,
            document_offsets.data(),
            document_count,
            intersections.data(),
            unions.data(),
            scores.data()
        );
    }
    return nb::make_tuple(
        intersections_object,
        unions_object,
        scores_object
    );
}

nb::tuple run_jaccard_batch_cuda(
    const Int64Array &candidate_shingles,
    const Int64Array &candidate_offsets,
    const Int64Array &corpus_shingles,
    const Int64Array &document_offsets
)
{
    if (candidate_offsets.shape(0) < 2){
        throw std::invalid_argument(
            "candidate_offsets must describe at least one candidate"
        );
    }
    if (document_offsets.shape(0) < 2){
        throw std::invalid_argument(
            "document_offsets must describe at least one document"
        );
    }

    const auto *candidate_offset_data = candidate_offsets.data();
    const auto *document_offset_data = document_offsets.data();

    if (candidate_offset_data[0] != 0) {
        throw std::invalid_argument(
            "candidate_offsets must begin with zero"
        );
    }
    if (document_offset_data[0] != 0) {
        throw std::invalid_argument(
            "document_offsets must begin with zero"
        );
    }

    const std::size_t candidate_count_size = candidate_offsets.shape(0) - 1;
    const std::size_t document_count_size = document_offsets.shape(0) - 1;
    const int candidate_count = checked_int_count(candidate_count_size, "candidate count");
    const int document_count = checked_int_count(document_count_size, "document count");
    const std::int64_t candidate_shingle_count = checked_int64_count(candidate_shingles.shape(0), "candidate shingle count");
    const std::int64_t corpus_shingle_count = checked_int64_count(corpus_shingles.shape(0), "corpus shingle count");

    if (
        candidate_offset_data[candidate_count]
        != candidate_shingle_count
    ) {
        throw std::invalid_argument(
            "final candidate offset must equal candidate shingles length"
        );
    }
    if (
        document_offset_data[document_count]
        != corpus_shingle_count
    ) {
        throw std::invalid_argument(
            "final document offset must equal corpus length"
        );
    }

    for (
        int index = 1;
        index <= candidate_count;
        ++index
    ) {
        if (
            candidate_offset_data[index]
            < candidate_offset_data[index - 1]
        ) {
            throw std::invalid_argument(
                "candidate_offsets must be nondecreasing"
            );
        }
    }
    for (
        int index = 1;
        index <= document_count;
        ++index
    ) {
        if (
            document_offset_data[index]
            < document_offset_data[index - 1]
        ) {
            throw std::invalid_argument(
                "document_offsets must be nondecreasing"
            );
        }
    }

    const std::size_t candidate_count_unsigned = static_cast<std::size_t>(candidate_count);
    const std::size_t document_count_unsigned = static_cast<std::size_t>(document_count);

    if (
        candidate_count_unsigned != 0
        && document_count_unsigned >
            std::numeric_limits<std::size_t>::max()
            / candidate_count_unsigned
    ) {
        throw std::overflow_error(
            "candidate-document result size exceeds the platform size range"
        );
    }

    const std::size_t result_count =
        candidate_count_unsigned
        * document_count_unsigned;

    if (
        result_count >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::overflow_error(
            "candidate-document result count exceeds the supported CUDA grid range"
        );
    }

    nb::module_ numpy = nb::module_::import_("numpy");

    /*
     * The native Fortran output is shaped as:
     *
     *     (document_count, candidate_count)
     *
     * A Fortran-contiguous NumPy array preserves the exact flattened
     * storage layout produced through the C ABI.
     */
    const nb::tuple output_shape =
        nb::make_tuple(
            document_count,
            candidate_count
        );
    
    nb::object intersections_object =
        numpy.attr("empty")(
            output_shape,
            "int64",
            "F"
        );

    nb::object unions_object =
        numpy.attr("empty")(
            output_shape,
            "int64",
            "F"
        );

    nb::object scores_object =
        numpy.attr("empty")(
            output_shape,
            "float64",
            "F"
        );
    
    auto intersections = nb::cast<
        nb::ndarray<
            nb::numpy,
            std::int64_t,
            nb::ndim<2>,
            nb::f_contig,
            nb::device::cpu
        >
    >(intersections_object);

    auto unions = nb::cast<
        nb::ndarray<
            nb::numpy,
            std::int64_t,
            nb::ndim<2>,
            nb::f_contig,
            nb::device::cpu
        >
    >(unions_object);

    auto scores = nb::cast<
        nb::ndarray<
            nb::numpy,
            double,
            nb::ndim<2>,
            nb::f_contig,
            nb::device::cpu
        >
    >(scores_object);

    {
        nb::gil_scoped_release release;

        fnlp_jaccard_batch_corpus_cuda(
            candidate_shingles.data(),
            candidate_shingle_count,
            candidate_offsets.data(),
            candidate_count,
            corpus_shingles.data(),
            corpus_shingle_count,
            document_offsets.data(),
            document_count,
            intersections.data(),
            unions.data(),
            scores.data()
        );
    }
    return nb::make_tuple(
        intersections_object,
        unions_object,
        scores_object
    );
}

nb::tuple run_vector_search_single(
    const Float32Vector &query_vector,
    const Float32MatrixF &database_vectors,
    int k,
    VectorSearchFunction search_function
)
{
    const std::size_t dimension_size =
        query_vector.shape(0);

    const std::size_t database_dimension_size =
        database_vectors.shape(0);

    const std::size_t vector_count_size =
        database_vectors.shape(1);

    if (dimension_size == 0) {
        throw std::invalid_argument(
            "vector dimension must be greater than zero"
        );
    }

    if (vector_count_size == 0) {
        throw std::invalid_argument(
            "database must contain at least one vector"
        );
    }

    if (database_dimension_size != dimension_size) {
        throw std::invalid_argument(
            "query and database vectors must have "
            "equal dimensions"
        );
    }

    if (k < 1) {
        throw std::invalid_argument(
            "k must be at least 1"
        );
    }

    if (
        static_cast<std::size_t>(k)
        > vector_count_size
    ) {
        throw std::invalid_argument(
            "k cannot exceed the database size"
        );
    }

    const int dimension = checked_int_count(
        dimension_size,
        "vector dimension"
    );

    const int vector_count = checked_int_count(
        vector_count_size,
        "database vector count"
    );

    nb::module_ numpy = nb::module_::import_("numpy");

    nb::object values_object = numpy.attr("empty")(
        k,
        "float32"
    );

    nb::object indices_object = numpy.attr("empty")(
        k,
        "int64"
    );

    auto values = nb::cast<
        nb::ndarray<
            nb::numpy,
            float,
            nb::ndim<1>,
            nb::c_contig,
            nb::device::cpu
        >
    >(values_object);

    auto indices = nb::cast<
        nb::ndarray<
            nb::numpy,
            std::int64_t,
            nb::ndim<1>,
            nb::c_contig,
            nb::device::cpu
        >
    >(indices_object);

    {
        /*
         * The native CUDA wrapper performs allocation, transfers,
         * GPU execution, synchronization, result transfer, and
         * deterministic CPU top-k selection.
         */
        nb::gil_scoped_release release;

        search_function(
            query_vector.data(),
            database_vectors.data(),
            dimension,
            vector_count,
            k,
            values.data(),
            indices.data()
        );
    }

    return nb::make_tuple(
        values_object,
        indices_object
    );
}

nb::tuple run_vector_search_batch(
    const Float32MatrixF &query_vectors,
    const Float32MatrixF &database_vectors,
    int k,
    VectorBatchSearchFunction search_function
)
{
    const std::size_t dimension_size =
        query_vectors.shape(0);

    const std::size_t query_count_size =
        query_vectors.shape(1);

    const std::size_t database_dimension_size =
        database_vectors.shape(0);

    const std::size_t vector_count_size =
        database_vectors.shape(1);

    if (dimension_size == 0) {
        throw std::invalid_argument(
            "vector dimension must be greater than zero"
        );
    }

    if (query_count_size == 0) {
        throw std::invalid_argument(
            "query batch must contain at least one vector"
        );
    }

    if (vector_count_size == 0) {
        throw std::invalid_argument(
            "database must contain at least one vector"
        );
    }

    if (database_dimension_size != dimension_size) {
        throw std::invalid_argument(
            "queries and database vectors must have "
            "equal dimensions"
        );
    }

    if (k < 1) {
        throw std::invalid_argument(
            "k must be at least 1"
        );
    }

    if (
        static_cast<std::size_t>(k)
        > vector_count_size
    ) {
        throw std::invalid_argument(
            "k cannot exceed the database size"
        );
    }

    const int dimension = checked_int_count(
        dimension_size,
        "vector dimension"
    );

    const int query_count = checked_int_count(
        query_count_size,
        "query count"
    );

    const int vector_count = checked_int_count(
        vector_count_size,
        "database vector count"
    );

    nb::module_ numpy = nb::module_::import_("numpy");

    /*
     * Native batch output shape:
     *
     *     (k, query_count)
     *
     * Use Fortran-contiguous NumPy arrays to preserve the exact
     * native storage layout.
     */
    const nb::tuple output_shape = nb::make_tuple(
        k,
        query_count
    );

    nb::object values_object = numpy.attr("empty")(
        output_shape,
        "float32",
        "F"
    );

    nb::object indices_object = numpy.attr("empty")(
        output_shape,
        "int64",
        "F"
    );

    auto values = nb::cast<
        nb::ndarray<
            nb::numpy,
            float,
            nb::ndim<2>,
            nb::f_contig,
            nb::device::cpu
        >
    >(values_object);

    auto indices = nb::cast<
        nb::ndarray<
            nb::numpy,
            std::int64_t,
            nb::ndim<2>,
            nb::f_contig,
            nb::device::cpu
        >
    >(indices_object);

    {
        nb::gil_scoped_release release;

        search_function(
            query_vectors.data(),
            database_vectors.data(),
            dimension,
            query_count,
            vector_count,
            k,
            values.data(),
            indices.data()
        );
    }

    return nb::make_tuple(
        values_object,
        indices_object
    );
}


NB_MODULE(_cuda, module)
{
    module.doc() = 
        "Native CUDA kernels for fortran-nlp-kernels";
    
    module.def(
        "jaccard_corpus_cuda",
        [](const Int64Array &candidate,
           const Int64Array &corpus,
           const Int64Array &document_offsets) {
            return run_jaccard_cuda(
                candidate,
                corpus,
                document_offsets
            );
        },
        nb::arg("candidate_shingles"),
        nb::arg("corpus_shingles"),
        nb::arg("document_offsets")
    );

    module.def(
        "jaccard_batch_corpus_cuda",
        [](const Int64Array &candidate_shingles,
           const Int64Array &candidate_offsets,
           const Int64Array &corpus_shingles,
           const Int64Array &document_offsets) {
            return run_jaccard_batch_cuda(
                candidate_shingles,
                candidate_offsets,
                corpus_shingles,
                document_offsets
            );
        },
        nb::arg("candidate_shingles"),
        nb::arg("candidate_offsets"),
        nb::arg("corpus_shingles"),
        nb::arg("document_offsets")
    );

    module.def(
        "l2_search_cuda",
        [](const Float32Vector &query_vector,
           const Float32MatrixF &database_vectors,
           int k) {
            return run_vector_search_single(
                query_vector,
                database_vectors,
                k,
                fnlp_l2_search_cuda
            );
        },
        nb::arg("query_vector"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "l2_search_batch_cuda",
        [](const Float32MatrixF &query_vectors,
           const Float32MatrixF &database_vectors,
           int k) {
            return run_vector_search_batch(
                query_vectors,
                database_vectors,
                k,
                fnlp_l2_search_batch_cuda
            );
        },
        nb::arg("query_vectors"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "ip_search_batch_cuda",
        [](const Float32MatrixF &query_vectors,
           const Float32MatrixF &database_vectors,
           int k) {
            return run_vector_search_batch(
                query_vectors,
                database_vectors,
                k,
                fnlp_ip_search_batch_cuda
            );
        },
        nb::arg("query_vectors"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "l2_search_cublas",
        [](const Float32Vector &query_vector,
           const Float32MatrixF &database_vectors,
           int k) {
            return run_vector_search_single(
                query_vector,
                database_vectors,
                k,
                fnlp_l2_search_cublas
            );
        },
        nb::arg("query_vector"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "ip_search_cublas",
        [](const Float32Vector &query_vector,
           const Float32MatrixF &database_vectors,
           int k) {
            return run_vector_search_single(
                query_vector,
                database_vectors,
                k,
                fnlp_ip_search_cublas
            );
        },
        nb::arg("query_vector"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "l2_search_batch_cublas",
        [](const Float32MatrixF &query_vectors,
           const Float32MatrixF &database_vectors,
           int k) {
            return run_vector_search_batch(
                query_vectors,
                database_vectors,
                k,
                fnlp_l2_search_batch_cublas
            );
        },
        nb::arg("query_vectors"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "ip_search_batch_cublas",
        [](const Float32MatrixF &query_vectors,
           const Float32MatrixF &database_vectors,
           int k) {
            return run_vector_search_batch(
                query_vectors,
                database_vectors,
                k,
                fnlp_ip_search_batch_cublas
            );
        },
        nb::arg("query_vectors"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );
}