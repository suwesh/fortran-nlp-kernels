program test_lexical_batch_core
    use, intrinsic :: iso_fortran_env, only: int64, real64

    use fnlp_lexical_core, only: &
        jaccard_corpus_serial, &
        jaccard_batch_corpus

    implicit none

    integer, parameter :: candidate_count = 2
    integer, parameter :: document_count = 3

    real(real64), parameter :: tolerance = 1.0e-12_real64

    integer(int64) :: candidate_shingles(5)
    integer(int64) :: candidate_offsets(candidate_count + 1)

    integer(int64) :: corpus_shingles(7)
    integer(int64) :: document_offsets(document_count + 1)

    integer(int64) :: batch_intersections( &
        document_count, &
        candidate_count &
    )

    integer(int64) :: batch_unions( &
        document_count, &
        candidate_count &
    )

    real(real64) :: batch_scores( &
        document_count, &
        candidate_count &
    )

    integer(int64) :: repeated_intersections(document_count)
    integer(int64) :: repeated_unions(document_count)
    real(real64) :: repeated_scores(document_count)

    integer(int64) :: expected_intersections( &
        document_count, &
        candidate_count &
    )

    integer(int64) :: expected_unions( &
        document_count, &
        candidate_count &
    )

    real(real64) :: expected_scores( &
        document_count, &
        candidate_count &
    )

    integer :: candidate_index
    integer(int64) :: candidate_start
    integer(int64) :: candidate_end


    ! Candidates:
    !
    ! candidate 0 = [1, 2, 3]
    ! candidate 1 = [2, 4]

    candidate_shingles = [ &
        1_int64, &
        2_int64, &
        3_int64, &
        2_int64, &
        4_int64 &
    ]

    candidate_offsets = [ &
        0_int64, &
        3_int64, &
        5_int64 &
    ]


    ! Corpus:
    !
    ! document 0 = [1, 2, 3]
    ! document 1 = [2, 3, 4]
    ! document 2 = [5]

    corpus_shingles = [ &
        1_int64, &
        2_int64, &
        3_int64, &
        2_int64, &
        3_int64, &
        4_int64, &
        5_int64 &
    ]

    document_offsets = [ &
        0_int64, &
        3_int64, &
        6_int64, &
        7_int64 &
    ]


    ! Rows are documents; columns are candidates.

    expected_intersections(:, 1) = [ &
        3_int64, &
        2_int64, &
        0_int64 &
    ]

    expected_intersections(:, 2) = [ &
        1_int64, &
        2_int64, &
        0_int64 &
    ]

    expected_unions(:, 1) = [ &
        3_int64, &
        4_int64, &
        4_int64 &
    ]

    expected_unions(:, 2) = [ &
        4_int64, &
        3_int64, &
        3_int64 &
    ]

    expected_scores(:, 1) = [ &
        1.0_real64, &
        0.5_real64, &
        0.0_real64 &
    ]

    expected_scores(:, 2) = [ &
        0.25_real64, &
        2.0_real64 / 3.0_real64, &
        0.0_real64 &
    ]


    call jaccard_batch_corpus( &
        candidate_shingles, &
        candidate_offsets, &
        candidate_count, &
        corpus_shingles, &
        document_offsets, &
        document_count, &
        batch_intersections, &
        batch_unions, &
        batch_scores &
    )


    if (any( &
        batch_intersections /= expected_intersections &
    )) then
        error stop "Batch intersection counts are incorrect."
    end if

    if (any(batch_unions /= expected_unions)) then
        error stop "Batch union counts are incorrect."
    end if

    if (any( &
        abs(batch_scores - expected_scores) > tolerance &
    )) then
        error stop "Batch Jaccard scores are incorrect."
    end if


    ! Compare each batch column against the validated
    ! single-candidate serial corpus routine.

    do candidate_index = 1, candidate_count
        candidate_start = &
            candidate_offsets(candidate_index) + 1_int64

        candidate_end = &
            candidate_offsets(candidate_index + 1)

        call jaccard_corpus_serial( &
            candidate_shingles( &
                candidate_start:candidate_end &
            ), &
            corpus_shingles, &
            document_offsets, &
            document_count, &
            repeated_intersections, &
            repeated_unions, &
            repeated_scores &
        )

        if (any( &
            repeated_intersections /= &
            batch_intersections(:, candidate_index) &
        )) then
            error stop &
                "Batch and repeated intersection counts differ."
        end if

        if (any( &
            repeated_unions /= &
            batch_unions(:, candidate_index) &
        )) then
            error stop &
                "Batch and repeated union counts differ."
        end if

        if (any( &
            abs( &
                repeated_scores - &
                batch_scores(:, candidate_index) &
            ) > tolerance &
        )) then
            error stop &
                "Batch and repeated Jaccard scores differ."
        end if
    end do


    print *, "Batch intersections:"
    print *, batch_intersections

    print *
    print *, "Batch unions:"
    print *, batch_unions

    print *
    print *, "Batch scores:"
    print *, batch_scores

    print *
    print *, "Batched Jaccard correctness: PASS"
    print *, "Batch/repeated-search equivalence: PASS"

end program test_lexical_batch_core
