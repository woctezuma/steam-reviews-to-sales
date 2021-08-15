import numpy as np

from utils.reviews_utils import get_steam_api_url
from utils.time_utils import get_target_date_as_timestamp


def get_test_apps():
    all_features = ["review_score", "total_positive", "total_negative", "total_reviews"]

    test_apps = {
        "Valheim (5.0 M)": {
            "appID": 892970,
            "data": [9, 127532, 4657, 132189],
            "date": "2021-03-03",
            "news_url": "https://store.steampowered.com/news/app/892970/view/3055101388621224471",
            "sales": 5e6,
        },
        "Valheim (6.0 M)": {
            "appID": 892970,
            "data": [9, 174523, 6398, 180921],
            "date": "2021-03-19",
            "news_url": "https://store.steampowered.com/news/app/892970/view/3047221359533847024",
            "sales": 6e6,
        },
        "Valheim (6.8 M)": {
            "appID": 892970,
            "data": [9, 196306, 7261, 203567],
            "date": "2021-04-01",
            "news_url": "https://embracer.com/report/embracer-group-publishes-full-year-report-and-q4-january-march-2021-operational-ebit-increased-216-to-sek-903-million/",
            "sales": 6.8e6,
        },
        "Iratus (0.3 M)": {
            "appID": 807120,
            "data": [8, 4587, 696, 5283],
            "date": "2021-08-12",
            "news_url": "https://store.steampowered.com/news/app/807120/view/2968421119174680934",
            "sales": 3e5,
        },
        "Frozenheim (0.1 M)": {
            "appID": 1134100,
            "data": [6, 900, 374, 1274],
            "date": "2021-08-15",
            "news_url": "https://store.steampowered.com/news/app/1134100/view/4269962068888223685",
            "sales": 1e5,
        },
    }

    return test_apps, all_features


def check_test_apps(model, features):
    test_apps, all_features = get_test_apps()

    for app_name in test_apps.keys():
        v_test = test_apps[app_name]["data"]

        v = [v_test[i] for (i, f) in enumerate(all_features) if f in features]
        v = np.array(v).reshape(1, -1)

        p = model.predict(v)

        sales_estimate = float(p[0])
        print(f"Input: {app_name} ---> predicted sales: {sales_estimate / 1e6:.3f} M")

    return


def get_api_url(app_id, end_date, verbose=True):
    target_timestamp = get_target_date_as_timestamp(end_date, verbose=verbose)

    url = get_steam_api_url(app_id)
    url += "?json=1&num_per_page=0&language=all&purchase_type=all&filter_offtopic_activity=0"
    url += f"&start_date=1&end_date={target_timestamp}&date_range_type=include"

    return url


def main():
    test_apps, all_features = get_test_apps()

    for app_name, app in test_apps.items():
        url = get_api_url(app_id=app["appID"], end_date=app["date"], verbose=False)
        print(f"{app_name} -> {url}")

    return True


if __name__ == "__main__":
    main()
