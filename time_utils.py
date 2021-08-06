from datetime import datetime


def get_target_date():
    return "2018-07-01"


def get_date_format():
    return "%Y-%m-%d"


def get_day_range(target_date=None, date_format=None, verbose=True):
    if target_date is None:
        target_date = get_target_date()

    if date_format is None:
        date_format = get_date_format()

    current_date = datetime.now()
    date_threshold = datetime.strptime(target_date, date_format)

    dt = current_date - date_threshold

    if verbose:
        print(dt)

    day_range = dt.days

    return day_range


def main():
    day_range = get_day_range()

    return True


if __name__ == "__main__":
    main()
