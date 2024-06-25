import numpy as np


def get_max_days_in_n_months(n_months: int) -> int:
    """Calculate the maximum possible number of days in a window of size 'n_months'.

    Will slightly overestimate for windows larger than a year, since every year is assumed to
    be a leap year for simplicity.

    Args:
        n_months: Number of months to consider

    Returns:
        Maximum possible number of days in any window of size 'n_months' months
    """
    # If more than one year, add 366 days for every year and solve the problem for the remaining
    # partial year
    years_offset = 0
    while n_months > 12:
        n_months = n_months - 12
        years_offset += 366

    max_sum = years_offset
    # For simplicity, assume the worst case of a leap year every year
    days_in_months = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for start_month_idx in range(12):
        # np.take allows to select from an array with wraparound
        month_window = np.take(
            days_in_months,
            np.arange(start_month_idx, start_month_idx + n_months, 1),
            mode="wrap",
        )
        num_days = sum(month_window) + years_offset
        max_sum = max(num_days, max_sum)
    return max_sum


def get_max_quarters_in_n_months(n_months: int):
    months_per_quarter = 3
    full_quarters, remainder = divmod(n_months, months_per_quarter)
    offset = 2 if remainder > 1 else 1
    return full_quarters + offset
