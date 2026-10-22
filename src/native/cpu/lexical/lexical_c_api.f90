module fnlp_lexical_c_api
    use, intrinsic :: iso_c_binding, only: c_int, c_int64_t, c_double
    use fnlp_lexical_core, only: jaccard_corpus_serial, jaccard_corpus_openmp, jaccard_batch_corpus
    implicit none
    private
    public :: fnlp_jaccard_corpus_serial
    public :: fnlp_jaccard_corpus_openmp
    public :: fnlp_jaccard_batch_corpus

contains
    subroutine fnlp_jaccard_corpus_serial( &
        candidate_shingles, &
        candidate_count, &
        corpus_shingles, &
        corpus_count, &
        document_offsets, &
        document_count, &
        intersection_counts, &
        union_counts, &
        jaccard_scores &
    ) bind(C, name="fnlp_jaccard_corpus_serial")
        integer(c_int64_t), intent(in) :: candidate_shingles(*)
        integer(c_int64_t), value, intent(in) :: candidate_count
        integer(c_int64_t), intent(in) :: corpus_shingles(*)
        integer(c_int64_t), value, intent(in) :: corpus_count
        integer(c_int64_t), intent(in) :: document_offsets(*)
        integer(c_int), value, intent(in) :: document_count
        integer(c_int64_t), intent(out) :: intersection_counts(*)
        integer(c_int64_t), intent(out) :: union_counts(*)
        real(c_double), intent(out) :: jaccard_scores(*)

        integer :: document_count_fortran
        document_count_fortran = int(document_count)
        
        call jaccard_corpus_serial( &
            candidate_shingles(1:candidate_count), &
            corpus_shingles(1:corpus_count), &
            document_offsets(1:document_count_fortran + 1), &
            document_count_fortran, &
            intersection_counts(1:document_count_fortran), &
            union_counts(1:document_count_fortran), &
            jaccard_scores(1:document_count_fortran) &
        )
    end subroutine fnlp_jaccard_corpus_serial

    subroutine fnlp_jaccard_corpus_openmp( &
        candidate_shingles, &
        candidate_count, &
        corpus_shingles, &
        corpus_count, &
        document_offsets, &
        document_count, &
        intersection_counts, &
        union_counts, &
        jaccard_scores &
    ) bind(C, name="fnlp_jaccard_corpus_openmp")
        integer(c_int64_t), intent(in) :: candidate_shingles(*)
        integer(c_int64_t), value, intent(in) :: candidate_count
        integer(c_int64_t), intent(in) :: corpus_shingles(*)
        integer(c_int64_t), value, intent(in) :: corpus_count
        integer(c_int64_t), intent(in) :: document_offsets(*)
        integer(c_int), value, intent(in) :: document_count
        integer(c_int64_t), intent(out) :: intersection_counts(*)
        integer(c_int64_t), intent(out) :: union_counts(*)
        real(c_double), intent(out) :: jaccard_scores(*)

        integer :: document_count_fortran
        document_count_fortran = int(document_count)

        call jaccard_corpus_openmp( &
            candidate_shingles(1:candidate_count), &
            corpus_shingles(1:corpus_count), &
            document_offsets(1:document_count_fortran + 1), &
            document_count_fortran, &
            intersection_counts(1:document_count_fortran), &
            union_counts(1:document_count_fortran), &
            jaccard_scores(1:document_count_fortran) &
        )
    end subroutine fnlp_jaccard_corpus_openmp

    ! C ABI wrapper for OpenMP batched Jaccard corpus scoring.
    ! Multiple packed candidate documents are compared against every packed corpus document.
    subroutine fnlp_jaccard_batch_corpus( &
        candidate_shingles, &
        candidate_shingle_count, &
        candidate_offsets, &
        candidate_count, &
        corpus_shingles, &
        corpus_shingle_count, &
        document_offsets, &
        document_count, &
        intersection_counts, &
        union_counts, &
        jaccard_scores &
    ) bind(C, name="fnlp_jaccard_batch_corpus")
        ! Flattened shingles across all candidate documents.
        integer(c_int64_t), intent(in) :: candidate_shingles(*)
        ! Total number of shingles across all candidates.
        integer(c_int64_t), value, intent(in) :: candidate_shingle_count
        ! Zero-based, end-exclusive candidate boundaries. Length = candidate_count + 1.
        integer(c_int64_t), intent(in) :: candidate_offsets(*)
        ! Number of candidate documents.
        integer(c_int), value, intent(in) :: candidate_count
        ! Flattened shingles across all corpus documents.
        integer(c_int64_t), intent(in) :: corpus_shingles(*)
        ! Total number of shingles across the entire corpus.
        integer(c_int64_t), value, intent(in) :: corpus_shingle_count
        ! Zero-based, end-exclusive corpus-document boundaries. Length = document_count + 1.
        integer(c_int64_t), intent(in) :: document_offsets(*)
        ! Number of corpus documents.
        integer(c_int), value, intent(in) :: document_count
        ! Output matrices flattened through the C ABI.
        ! Native Fortran shape: (document_count, candidate_count)
        integer(c_int64_t), intent(out) :: intersection_counts(*)
        integer(c_int64_t), intent(out) :: union_counts(*)
        real(c_double), intent(out) :: jaccard_scores(*)

        integer :: candidate_count_fortran
        integer :: document_count_fortran

        candidate_count_fortran = int(candidate_count, kind=kind(candidate_count_fortran))
        document_count_fortran = int(document_count, kind=kind(document_count_fortran))

        call jaccard_batch_corpus( &
            candidate_shingles(1:candidate_shingle_count), &
            candidate_offsets(1:candidate_count_fortran + 1), &
            candidate_count_fortran, &
            corpus_shingles(1:corpus_shingle_count), &
            document_offsets(1:document_count_fortran + 1), &
            document_count_fortran, &
            intersection_counts(1:document_count_fortran * candidate_count_fortran), &
            union_counts(1:document_count_fortran * candidate_count_fortran), &
            jaccard_scores(1:document_count_fortran * candidate_count_fortran) &
        )
    end subroutine fnlp_jaccard_batch_corpus

end module fnlp_lexical_c_api