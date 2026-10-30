import re

import fortran_nlp


def test_package_exposes_version():
    assert isinstance(
        fortran_nlp.__version__,
        str,
    )

    assert re.fullmatch(
        r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.dev\d+)?",
        fortran_nlp.__version__,
    )
