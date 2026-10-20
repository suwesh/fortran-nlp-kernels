module fnlp_lexical_core
    use, intrinsic :: iso_fortran_env, only: int64, real64
    implicit none
    private
    public :: jaccard_corpus_serial
    public :: jaccard_corpus_openmp

contains
    ! candidate × one existing document - serial
    subroutine jaccard_sorted( &
        candidate_shingles, &
        existing_shingles, &
        intersection_count, &
        union_count, &
        jaccard_score &
    )
        ! parameterized variables with format <type, attributes :: variable_name> :
        integer(int64), intent(in) :: candidate_shingles(:)
        integer(int64), intent(in) :: existing_shingles(:)
        integer(int64), intent(out) :: intersection_count
        integer(int64), intent(out) :: union_count
        real(real64), intent(out) :: jaccard_score
        
        ! local variables with format <type :: variable_name> :
        integer(int64) :: candidate_index
        integer(int64) :: existing_index
        integer(int64) :: candidate_count
        integer(int64) :: existing_count

        candidate_count = size(candidate_shingles, kind=int64)
        existing_count = size(existing_shingles, kind=int64)
        candidate_index = 1_int64
        existing_index = 1_int64
        intersection_count = 0_int64

        do while (candidate_index <= candidate_count .and. existing_index <= existing_count)
            if (candidate_shingles(candidate_index) < existing_shingles(existing_index)) then
                candidate_index = candidate_index + 1_int64
            else if (candidate_shingles(candidate_index) > existing_shingles(existing_index)) then
                existing_index = existing_index + 1_int64
            else 
                intersection_count = intersection_count + 1_int64
                candidate_index = candidate_index + 1_int64
                existing_index = existing_index + 1_int64
            end if
        end do
        
        ! Union and Jaccard calculations.
        ! The union size follows: ∣A∪B∣=∣A∣+∣B∣−∣A∩B∣
        union_count = candidate_count + existing_count - intersection_count
        if (union_count == 0_int64) then 
            jaccard_score = 0.0_real64
        else 
            ! formulae: J(A,B)= ∣A∩B∣ / ∣A∪B∣
            jaccard_score = real(intersection_count, kind=real64) / real(union_count, kind=real64)
        end if
    end subroutine jaccard_sorted

    ! candidate × all existing documents - serial
    subroutine jaccard_corpus_serial( &
        candidate_shingles, &
        corpus_shingles, &
        document_offsets, &
        document_count, &
        intersection_counts, &
        union_counts, &
        jaccard_scores &
    )
        ! Inputs.
        integer(int64), intent(in) :: candidate_shingles(:)
        integer(int64), intent(in) :: corpus_shingles(:)
        integer(int64), intent(in) :: document_offsets(:)
        integer, intent(in) :: document_count
        ! One output value for each existing document. each output parameter dim = no of documents in corpus 
        integer(int64), intent(out) :: intersection_counts(document_count)
        integer(int64), intent(out) :: union_counts(document_count)
        real(real64), intent(out) :: jaccard_scores(document_count)

        ! Local variables
        integer :: document_index
        integer(int64) :: start_index
        integer(int64) :: end_index

        do document_index = 1, document_count
            ! Python offsets are zero-based and end-exclusive.
            ! Fortran indices are one-based and end-inclusive.
            start_index = document_offsets(document_index) + 1_int64
            end_index = document_offsets(document_index + 1)
            call jaccard_sorted(&
                candidate_shingles, &
                corpus_shingles(start_index:end_index), &
                intersection_counts(document_index), &
                union_counts(document_index), &
                jaccard_scores(document_index) &
            )
        end do
    end subroutine jaccard_corpus_serial

    ! Candidate document x all existing documents - parallel using OpenMP.
    subroutine jaccard_corpus_openmp( &
        candidate_shingles, &
        corpus_shingles, &
        document_offsets, &
        document_count, &
        intersection_counts, &
        union_counts, &
        jaccard_scores &
    )
        ! Inputs shared among all OpenMP threads.
        integer(int64), intent(in) :: candidate_shingles(:)
        integer(int64), intent(in) :: corpus_shingles(:)
        integer(int64), intent(in) :: document_offsets(:)
        integer, intent(in) :: document_count
        ! Each loop iteration writes to a different array element.
        integer(int64), intent(out) :: intersection_counts(document_count)
        integer(int64), intent(out) :: union_counts(document_count)
        real(real64), intent(out) :: jaccard_scores(document_count)

        ! Local loop and document-boundary variables
        integer :: document_index
        integer(int64) :: start_index
        integer(int64) :: end_index

        !$omp parallel do default(none) &
        !$omp shared(candidate_shingles, corpus_shingles, document_offsets, document_count) &
        !$omp shared(intersection_counts, union_counts, jaccard_scores) &
        !$omp private(document_index, start_index, end_index) &
        !$omp schedule(static)

        do document_index = 1, document_count
            start_index = document_offsets(document_index) + 1_int64
            end_index = document_offsets(document_index + 1)
            call jaccard_sorted(&
                candidate_shingles, &
                corpus_shingles(start_index:end_index), &
                intersection_counts(document_index), &
                union_counts(document_index), &
                jaccard_scores(document_index) &
            )
        end do
        !$omp end parallel do
    end subroutine jaccard_corpus_openmp

end module fnlp_lexical_core