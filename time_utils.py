from datetime import datetime


def get_target_date():
    return "2018-07-01"


def get_date_format():
    return "%Y-%m-%d"


def get_today_as_datetime():
    return datetime.now()


def get_target_date_as_datetime(target_date=None, date_format=None):
    if target_date is None:
        target_date = get_target_date()

    if date_format is None:
        date_format = get_date_format()

    target_datetime = datetime.strptime(target_date, date_format)

    return target_datetime


def get_day_range(target_date=None, date_format=None, verbose=True):
    current_date = get_today_as_datetime()
    date_threshold = get_target_date_as_datetime(target_date, date_format)

    dt = current_date - date_threshold

    if verbose:
        print(dt)

    day_range = dt.days

    return day_range


def convert_from_datetime_to_timestamp(date_as_datetime, verbose=True):
    date_as_timestamp = int(datetime.timestamp(date_as_datetime))

    if verbose:
        print(f"Unix timestamp: {date_as_timestamp}")

    return date_as_timestamp


def get_target_date_as_timestamp(target_date=None, date_format=None, verbose=True):
    target_datetime = get_target_date_as_datetime(target_date, date_format)
    target_timestamp = int(datetime.timestamp(target_datetime))

    if verbose:
        print(f"Unix timestamp: {target_timestamp}")

    return target_timestamp


def main():
    target_date = get_target_date()
    date_format = get_date_format()
    day_range = get_day_range(target_date, date_format)
    target_timestamp = get_target_date_as_timestamp(target_date, date_format)

    return True


if __name__ == "__main__":
    main()
