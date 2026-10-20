program test_lexical_core
    use, intrinsic :: iso_fortran_env, only: int64, real64
    use fnlp_lexical_core, only: &
        jaccard_corpus_serial, &
        jaccard_corpus_openmp

    implicit none

    integer(int64) :: candidate(4)
    integer(int64) :: corpus(9)
    integer(int64) :: offsets(4)

    integer(int64) :: serial_intersections(3)
    integer(int64) :: serial_unions(3)
    real(real64) :: serial_scores(3)

    integer(int64) :: openmp_intersections(3)
    integer(int64) :: openmp_unions(3)
    real(real64) :: openmp_scores(3)

    candidate = [10_int64, 20_int64, 40_int64, 70_int64]

    corpus = [ &
        10_int64, 20_int64, 40_int64, &
        15_int64, 20_int64, &
        10_int64, 30_int64, 60_int64, 70_int64 &
    ]

    offsets = [0_int64, 3_int64, 5_int64, 9_int64]

    call jaccard_corpus_serial( &
        candidate, corpus, offsets, 3, &
        serial_intersections, serial_unions, serial_scores &
    )

    call jaccard_corpus_openmp( &
        candidate, corpus, offsets, 3, &
        openmp_intersections, openmp_unions, openmp_scores &
    )

    if (any(serial_intersections /= openmp_intersections)) error stop
    if (any(serial_unions /= openmp_unions)) error stop

    if (any(abs(serial_scores - openmp_scores) > 1.0e-15_real64)) then
        error stop
    end if

    print *, "Intersections:", serial_intersections
    print *, "Unions:", serial_unions
    print *, "Scores:", serial_scores
    print *, "Serial/OpenMP equivalence: PASS"

end program test_lexical_core