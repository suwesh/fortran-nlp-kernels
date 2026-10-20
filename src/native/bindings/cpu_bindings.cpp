#include <cstdint>
#include <limits>
#include <stdexcept>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "fortran_nlp_lexical.h"

namespace nb = nanobind;

using Int64Array = nb::ndarray<
    nb::numpy,
    const std::int64_t,
    nb::ndim<1>,
    nb::c_contig,
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


NB_MODULE(_cpu, module)
{
    module.doc() =
        "Native CPU kernels for fortran-nlp-kernels";

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
}