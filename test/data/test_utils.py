import pytest

from nowcastml.data.utils import get_max_days_in_n_months, get_max_quarters_in_n_months


@pytest.mark.parametrize(
    "n_months, expected",
    [
        (1, 31),
        (2, 62),
        (3, 92),
        (4, 123),
        (5, 153),
        (6, 184),
        (7, 215),
        (8, 245),
        (9, 276),
        (10, 306),
        (11, 337),
        (12, 366),
        (24, 2 * 366),
    ],
)
def test_get_max_days_in_n_months(n_months, expected):
    """Tests that the correct number of days is retrieved for a given number of months"""
    assert get_max_days_in_n_months(n_months) == expected


@pytest.mark.parametrize(
    "n_months, expected",
    [
        (1, 1),
        (2, 2),
        (3, 2),
        (4, 2),
        (5, 3),
        (6, 3),
        (7, 3),
        (8, 4),
        (9, 4),
        (10, 4),
        (11, 5),
    ],
)
def test_get_max_quarters_in_n_months(n_months, expected):
    """Tests that the correct number of quarters is retrieved for a given number of months"""
    assert get_max_quarters_in_n_months(n_months) == expected
