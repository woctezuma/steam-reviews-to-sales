import requests

from utils.time_utils import get_target_date_as_timestamp


def get_steam_api_url(app_id):
    return f"https://store.steampowered.com/appreviews/{app_id}"


def get_request_params(target_timestamp=None, verbose=True):
    # References:
    # - https://partner.steamgames.com/doc/store/getreviews
    # - browser dev tools on store pages, e.g. https://store.steampowered.com/app/570/#app_reviews_hash

    if target_timestamp is None:
        target_timestamp = get_target_date_as_timestamp(verbose=verbose)

    params = {
        "json": "1",
        "num_per_page": "0",  # text content of reviews is not needed
        "language": "all",  # caveat: default seems to be "english", so reviews would be missing if unchanged!
        "purchase_type": "all",  # caveat: default is "steam", so reviews would be missing if unchanged!
        "filter_offtopic_activity": "0",  # to un-filter review-bombs, e.g. https://store.steampowered.com/app/481510/
        "start_date": "1",  # this is the minimal value which allows to filter by date
        "end_date": str(target_timestamp),
        "date_range_type": "include",  # this parameter is needed to filter by date
    }

    return params


def download_review_stats(app_id, target_timestamp=None, verbose=True):
    url = get_steam_api_url(app_id)
    params = get_request_params(target_timestamp, verbose=verbose)

    response = requests.get(url, params=params)

    if response.ok:
        result = response.json()
    else:
        result = None

    if verbose:
        print(result)

    return result


def main():
    app_ids = [329070, 573170]
    target_timestamp = get_target_date_as_timestamp()

    for app_id in app_ids:
        result = download_review_stats(app_id, target_timestamp, verbose=True)

    return True


if __name__ == "__main__":
    main()
