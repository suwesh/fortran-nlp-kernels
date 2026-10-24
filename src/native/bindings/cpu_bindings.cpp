#include <cstdint>
#include <limits>
#include <stdexcept>

#include <omp.h>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "fortran_nlp_lexical.h"
#include "fortran_nlp_vector_search.h"

extern "C" {
    int openblas_get_num_threads();
    void openblas_set_num_threads(int thread_count);
}

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
using Float32Database = nb::ndarray<
    nb::numpy,
    const float,
    nb::ndim<2>,
    nb::f_contig,
    nb::device::cpu
>;
using Float32QueryBatch = nb::ndarray<
    nb::numpy,
    const float,
    nb::ndim<2>,
    nb::f_contig,
    nb::device::cpu
>;

nb::tuple run_jaccard(
    const Int64Array &candidate,
    const Int64Array &corpus,
    const Int64Array &offsets,
    bool use_openmp
)
{
    if (offsets.shape(0) < 1) {
        throw std::invalid_argument(
            "document_offsets must contain at least one element"
        );
    }

    const auto *offset_data = offsets.data();

    if (offset_data[0] != 0) {
        throw std::invalid_argument(
            "document_offsets must begin with zero"
        );
    }

    const std::size_t document_count_size =
        offsets.shape(0) - 1;

    if (
        document_count_size >
        static_cast<std::size_t>(std::numeric_limits<int>::max())
    ) {
        throw std::overflow_error(
            "document count exceeds the C int range"
        );
    }

    const int document_count =
        static_cast<int>(document_count_size);

    if (
        offset_data[document_count] !=
        static_cast<std::int64_t>(corpus.shape(0))
    ) {
        throw std::invalid_argument(
            "final document offset must equal corpus length"
        );
    }

    for (int index = 1; index <= document_count; ++index) {
        if (offset_data[index] < offset_data[index - 1]) {
            throw std::invalid_argument(
                "document_offsets must be nondecreasing"
            );
        }
    }

    nb::module_ numpy = nb::module_::import_("numpy");

    nb::object intersections_object =
        numpy.attr("empty")(document_count, "int64");

    nb::object unions_object =
        numpy.attr("empty")(document_count, "int64");

    nb::object scores_object =
        numpy.attr("empty")(document_count, "float64");

    auto intersections =
        nb::cast<nb::ndarray<nb::numpy, std::int64_t>>(
            intersections_object
        );

    auto unions =
        nb::cast<nb::ndarray<nb::numpy, std::int64_t>>(
            unions_object
        );

    auto scores =
        nb::cast<nb::ndarray<nb::numpy, double>>(
            scores_object
        );

    {
        nb::gil_scoped_release release;

        if (use_openmp) {
            fnlp_jaccard_corpus_openmp(
                candidate.data(),
                static_cast<std::int64_t>(candidate.shape(0)),
                corpus.data(),
                static_cast<std::int64_t>(corpus.shape(0)),
                offsets.data(),
                document_count,
                intersections.data(),
                unions.data(),
                scores.data()
            );
        } else {
            fnlp_jaccard_corpus_serial(
                candidate.data(),
                static_cast<std::int64_t>(candidate.shape(0)),
                corpus.data(),
                static_cast<std::int64_t>(corpus.shape(0)),
                offsets.data(),
                document_count,
                intersections.data(),
                unions.data(),
                scores.data()
            );
        }
    }

    return nb::make_tuple(
        intersections_object,
        unions_object,
        scores_object
    );
}

nb::tuple run_jaccard_batch(
    const Int64Array &candidate_shingles,
    const Int64Array &candidate_offsets,
    const Int64Array &corpus_shingles,
    const Int64Array &document_offsets
)
{
    if (candidate_offsets.shape(0) < 2) {
        throw std::invalid_argument(
            "candidate_offsets must describe at least one candidate"
        );
    }

    if (document_offsets.shape(0) < 2) {
        throw std::invalid_argument(
            "document_offsets must describe at least one document"
        );
    }

    const auto *candidate_offset_data =
        candidate_offsets.data();

    const auto *document_offset_data =
        document_offsets.data();

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

    const std::size_t candidate_count_size =
        candidate_offsets.shape(0) - 1;

    const std::size_t document_count_size =
        document_offsets.shape(0) - 1;

    if (
        candidate_count_size >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::overflow_error(
            "candidate count exceeds the C int range"
        );
    }

    if (
        document_count_size >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::overflow_error(
            "document count exceeds the C int range"
        );
    }

    const int candidate_count =
        static_cast<int>(candidate_count_size);

    const int document_count =
        static_cast<int>(document_count_size);

    if (
        candidate_offset_data[candidate_count] !=
        static_cast<std::int64_t>(
            candidate_shingles.shape(0)
        )
    ) {
        throw std::invalid_argument(
            "final candidate offset must equal "
            "candidate shingles length"
        );
    }

    if (
        document_offset_data[document_count] !=
        static_cast<std::int64_t>(
            corpus_shingles.shape(0)
        )
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
            candidate_offset_data[index] <
            candidate_offset_data[index - 1]
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
            document_offset_data[index] <
            document_offset_data[index - 1]
        ) {
            throw std::invalid_argument(
                "document_offsets must be nondecreasing"
            );
        }
    }

    nb::module_ numpy =
        nb::module_::import_("numpy");

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

    auto intersections =
        nb::cast<
            nb::ndarray<nb::numpy, std::int64_t>
        >(
            intersections_object
        );

    auto unions =
        nb::cast<
            nb::ndarray<nb::numpy, std::int64_t>
        >(
            unions_object
        );

    auto scores =
        nb::cast<
            nb::ndarray<nb::numpy, double>
        >(
            scores_object
        );

    {
        nb::gil_scoped_release release;

        fnlp_jaccard_batch_corpus(
            candidate_shingles.data(),
            static_cast<std::int64_t>(
                candidate_shingles.shape(0)
            ),
            candidate_offsets.data(),
            candidate_count,
            corpus_shingles.data(),
            static_cast<std::int64_t>(
                corpus_shingles.shape(0)
            ),
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

nb::tuple run_vector_search(
    const Float32Vector &query,
    const Float32Database &database,
    int k,
    bool use_l2,
    bool use_openmp
)
{
    const std::size_t dimension_size = query.shape(0);
    const std::size_t vector_count_size = database.shape(1);

    if (database.shape(0) != dimension_size) {
        throw std::invalid_argument(
            "query and database vectors must have equal dimensions"
        );
    }

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

    if (k < 1) {
        throw std::invalid_argument(
            "k must be at least 1"
        );
    }

    if (
        static_cast<std::size_t>(k) >
        vector_count_size
    ) {
        throw std::invalid_argument(
            "k cannot exceed the database size"
        );
    }

    if (
        dimension_size >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::overflow_error(
            "vector dimension exceeds the C int range"
        );
    }

    if (
        vector_count_size >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::overflow_error(
            "database size exceeds the C int range"
        );
    }

    const int dimension =
        static_cast<int>(dimension_size);

    const int vector_count =
        static_cast<int>(vector_count_size);

    nb::module_ numpy = nb::module_::import_("numpy");

    nb::object values_object =
        numpy.attr("empty")(k, "float32");

    nb::object indices_object =
        numpy.attr("empty")(k, "int64");

    auto values =
        nb::cast<nb::ndarray<nb::numpy, float>>(
            values_object
        );

    auto indices =
        nb::cast<nb::ndarray<nb::numpy, std::int64_t>>(
            indices_object
        );

    {
        nb::gil_scoped_release release;

        if (use_l2 && use_openmp) {
            fnlp_l2_search_openmp(
                query.data(),
                database.data(),
                dimension,
                vector_count,
                k,
                values.data(),
                indices.data()
            );
        } else if (use_l2) {
            fnlp_l2_search_serial(
                query.data(),
                database.data(),
                dimension,
                vector_count,
                k,
                values.data(),
                indices.data()
            );
        } else if (use_openmp) {
            fnlp_ip_search_openmp(
                query.data(),
                database.data(),
                dimension,
                vector_count,
                k,
                values.data(),
                indices.data()
            );
        } else {
            fnlp_ip_search_serial(
                query.data(),
                database.data(),
                dimension,
                vector_count,
                k,
                values.data(),
                indices.data()
            );
        }
    }

    return nb::make_tuple(
        values_object,
        indices_object
    );
}

nb::tuple run_vector_search_blas(
    const Float32Vector &query,
    const Float32Database &database,
    int k,
    bool use_l2
)
{
    const int dimension =
        static_cast<int>(query.shape(0));

    const int vector_count =
        static_cast<int>(database.shape(1));

    if (database.shape(0) != query.shape(0)) {
        throw std::invalid_argument(
            "query and database vectors must have equal dimensions"
        );
    }

    if (dimension < 1 || vector_count < 1) {
        throw std::invalid_argument(
            "query and database must not be empty"
        );
    }

    if (k < 1 || k > vector_count) {
        throw std::invalid_argument(
            "k must be between 1 and the database size"
        );
    }

    nb::module_ numpy = nb::module_::import_("numpy");

    nb::object values_object =
        numpy.attr("empty")(k, "float32");

    nb::object indices_object =
        numpy.attr("empty")(k, "int64");

    auto values =
        nb::cast<nb::ndarray<nb::numpy, float>>(
            values_object
        );

    auto indices =
        nb::cast<nb::ndarray<nb::numpy, std::int64_t>>(
            indices_object
        );

    {
        nb::gil_scoped_release release;

        if (use_l2) {
            fnlp_l2_search_blas(
                query.data(),
                database.data(),
                dimension,
                vector_count,
                k,
                values.data(),
                indices.data()
            );
        } else {
            fnlp_ip_search_blas(
                query.data(),
                database.data(),
                dimension,
                vector_count,
                k,
                values.data(),
                indices.data()
            );
        }
    }

    return nb::make_tuple(
        values_object,
        indices_object
    );
}

nb::tuple run_vector_search_batch_blas(
    const Float32QueryBatch &queries,
    const Float32Database &database,
    int k,
    bool use_l2
)
{
    const std::size_t dimension_size = queries.shape(0);
    const std::size_t query_count_size = queries.shape(1);
    const std::size_t vector_count_size = database.shape(1);

    if (database.shape(0) != dimension_size) {
        throw std::invalid_argument(
            "queries and database vectors must have equal dimensions"
        );
    }

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

    if (k < 1) {
        throw std::invalid_argument(
            "k must be at least 1"
        );
    }

    if (
        static_cast<std::size_t>(k) >
        vector_count_size
    ) {
        throw std::invalid_argument(
            "k cannot exceed the database size"
        );
    }

    if (
        dimension_size >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::overflow_error(
            "vector dimension exceeds the C int range"
        );
    }

    if (
        query_count_size >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::overflow_error(
            "query count exceeds the C int range"
        );
    }

    if (
        vector_count_size >
        static_cast<std::size_t>(
            std::numeric_limits<int>::max()
        )
    ) {
        throw std::overflow_error(
            "database size exceeds the C int range"
        );
    }

    const int dimension =
        static_cast<int>(dimension_size);

    const int query_count =
        static_cast<int>(query_count_size);

    const int vector_count =
        static_cast<int>(vector_count_size);

    nb::module_ numpy = nb::module_::import_("numpy");

    nb::object values_object =
        numpy.attr("empty")(
            nb::make_tuple(k, query_count),
            "float32",
            "F"
        );

    nb::object indices_object =
        numpy.attr("empty")(
            nb::make_tuple(k, query_count),
            "int64",
            "F"
        );

    auto values =
        nb::cast<nb::ndarray<nb::numpy, float>>(
            values_object
        );

    auto indices =
        nb::cast<
            nb::ndarray<nb::numpy, std::int64_t>
        >(
            indices_object
        );

    {
        nb::gil_scoped_release release;

        if (use_l2) {
            fnlp_l2_search_batch_blas(
                queries.data(),
                database.data(),
                dimension,
                query_count,
                vector_count,
                k,
                values.data(),
                indices.data()
            );
        } else {
            fnlp_ip_search_batch_blas(
                queries.data(),
                database.data(),
                dimension,
                query_count,
                vector_count,
                k,
                values.data(),
                indices.data()
            );
        }
    }

    return nb::make_tuple(
        values_object,
        indices_object
    );
}


NB_MODULE(_cpu, module)
{
    module.doc() =
        "Native CPU kernels for fortran-nlp-kernels";
    
    module.def(
        "get_openmp_max_threads",
        []() {
            return omp_get_max_threads();
        }
    );
    module.def(
        "set_openmp_num_threads",
        [](int thread_count) {
            if (thread_count < 1) {
                throw std::invalid_argument("thread_count must be at least 1");
            }
            omp_set_num_threads(thread_count);
        },
        nb::arg("thread_count")
    );

    module.def(
        "get_openblas_num_threads",
        []() {
            return openblas_get_num_threads();
        }
    );
    module.def(
        "set_openblas_num_threads",
        [](int thread_count) {
            if (thread_count < 1) {
                throw std::invalid_argument("thread_count must be at least 1");
            }
            openblas_set_num_threads(thread_count);
        },
        nb::arg("thread_count")
    );

    module.def(
        "jaccard_corpus_serial",
        [](const Int64Array &candidate,
           const Int64Array &corpus,
           const Int64Array &offsets) {
            return run_jaccard(
                candidate,
                corpus,
                offsets,
                false
            );
        },
        nb::arg("candidate_shingles"),
        nb::arg("corpus_shingles"),
        nb::arg("document_offsets")
    );

    module.def(
        "jaccard_corpus_openmp",
        [](const Int64Array &candidate,
           const Int64Array &corpus,
           const Int64Array &offsets) {
            return run_jaccard(
                candidate,
                corpus,
                offsets,
                true
            );
        },
        nb::arg("candidate_shingles"),
        nb::arg("corpus_shingles"),
        nb::arg("document_offsets")
    );

    module.def(
        "jaccard_batch_corpus",
        [](const Int64Array &candidate_shingles,
           const Int64Array &candidate_offsets,
           const Int64Array &corpus_shingles,
           const Int64Array &document_offsets) {
            return run_jaccard_batch(
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
        "l2_search_serial",
        [](const Float32Vector &query,
           const Float32Database &database,
           int k) {
            return run_vector_search(
                query, database, k, true, false);
        },
        nb::arg("query_vector"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );
    
    module.def(
        "l2_search_openmp",
        [](const Float32Vector &query,
           const Float32Database &database,
           int k) {
            return run_vector_search(
                query, database, k, true, true);
        },
        nb::arg("query_vector"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "ip_search_serial",
        [](const Float32Vector &query,
           const Float32Database &database,
           int k) {
            return run_vector_search(
                query, database, k, false, false);
        },
        nb::arg("query_vector"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "ip_search_openmp",
        [](const Float32Vector &query,
           const Float32Database &database,
           int k) {
            return run_vector_search(
                query, database, k, false, true);
        },
        nb::arg("query_vector"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "l2_search_blas",
        [](const Float32Vector &query,
           const Float32Database &database,
           int k) {
            return run_vector_search_blas(
                query, database, k, true);
        },
        nb::arg("query_vector"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "ip_search_blas",
        [](const Float32Vector &query,
           const Float32Database &database,
           int k) {
            return run_vector_search_blas(
                query, database, k, false);
        },
        nb::arg("query_vector"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "l2_search_batch_blas",
        [](const Float32QueryBatch &queries,
           const Float32Database &database,
           int k) {
            return run_vector_search_batch_blas(
                queries, database, k, true);
        },
        nb::arg("query_vectors"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );

    module.def(
        "ip_search_batch_blas",
        [](const Float32QueryBatch &queries,
           const Float32Database &database,
           int k) {
            return run_vector_search_batch_blas(
                queries, database, k, false);
        },
        nb::arg("query_vectors"),
        nb::arg("database_vectors"),
        nb::arg("k")
    );
}