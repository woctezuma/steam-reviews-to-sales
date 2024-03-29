import json
import time

from utils.data_utils import load_app_ids
from utils.reviews_utils import download_review_stats


def get_ouput_fname():
    return "data/review_stats.json"


def load_output_dict():
    try:
        with open(get_ouput_fname()) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    return data


def write_output_dict(data):
    with open(get_ouput_fname(), "w") as f:
        json.dump(data, f)

    return


def get_rate_limits():
    # Reference: https://github.com/woctezuma/download-steam-reviews/blob/master/steamreviews/download_reviews.py

    rate_limits = {
        "max_num_queries": 600,
        "cooldown": (4 * 60) + 10,  # 4 minutes plus a cushion,
    }

    return rate_limits


def get_review_fields():
    return [
        "review_score",
        "review_score_desc",
        "total_positive",
        "total_negative",
        "total_reviews",
    ]


def download_data(app_ids, verbose=True):
    data = load_output_dict()
    processed_app_ids = [int(app_id) for app_id in data]

    unprocessed_app_ids = set(app_ids).difference(processed_app_ids)

    rate_limits = get_rate_limits()
    save_every = 15

    for query_count, app_id in enumerate(unprocessed_app_ids, start=1):
        print(f"[{query_count}/{len(unprocessed_app_ids)}] Query for appID = {app_id}")

        result = download_review_stats(app_id, verbose=verbose)

        if result is not None:
            query_summary = result["query_summary"]

            app_info = {}
            for key in get_review_fields():
                app_info[key] = query_summary[key]

            data[app_id] = app_info
        else:
            print(f"[X] Query failed for appID = {app_id}")

        if (query_count % save_every == 0) or (query_count == len(unprocessed_app_ids)):
            write_output_dict(data)

        if query_count % rate_limits["max_num_queries"] == 0:
            write_output_dict(data)
            cooldown_duration = rate_limits["cooldown"]
            print(f"#queries {query_count} reached. Cooldown: {cooldown_duration} s")
            time.sleep(cooldown_duration)

    return data


def main():
    app_ids = load_app_ids()
    data = download_data(app_ids, verbose=False)

    return True


if __name__ == "__main__":
    main()
