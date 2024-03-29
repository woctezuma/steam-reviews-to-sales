import numpy as np

from utils.mapie_utils import (
    get_prediction_bounds,
    invert_prediction_bounds,
    predict_with_mapie,
)
from utils.reviews_utils import get_steam_api_url
from utils.time_utils import get_target_date_as_timestamp
from utils.transform_utils import (
    get_review_chance,
    get_review_multiplier,
    invert_review_chance,
)


def expand_targets_in_test_apps(test_apps, all_features):
    for app_name in test_apps:
        v_test = test_apps[app_name]["data"]

        X = v_test[all_features.index("total_reviews")]
        y = test_apps[app_name]["sales"]

        test_apps[app_name]["review_chance"] = f"{get_review_chance(X, y):.3f}"
        test_apps[app_name]["review_multiplier"] = f"{get_review_multiplier(X, y):.0f}"

    return test_apps


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
        "Valheim (7.9 M)": {
            "appID": 892970,
            "data": [9, 238004, 9824, 247828],
            "date": "2021-07-01",
            "news_url": "https://embracer.com/wp-content/uploads/2021/08/1455434.pdf",
            "sales": 7.9e6,
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

    test_apps = expand_targets_in_test_apps(test_apps, all_features)

    return test_apps, all_features


def check_test_apps(
    model,
    features,
    transform_output=False,
    inverse_func=None,
    arbitrary_display_threshold=3,
):
    if inverse_func is None:
        inverse_func = invert_review_chance

    test_apps, all_features = get_test_apps()

    for app_name in test_apps:
        v_test = test_apps[app_name]["data"]

        v = [v_test[i] for (i, f) in enumerate(all_features) if f in features]
        v = np.array(v).reshape(1, -1)

        p, sigma = predict_with_mapie(model, v)
        p_lower, p_upper = get_prediction_bounds(p, sigma, required_to_be_positive=True)

        if transform_output:
            p = inverse_func(v, p)
            p_lower, p_upper = invert_prediction_bounds(
                v,
                p_lower,
                p_upper,
                inverse_func=inverse_func,
            )

        p_lower = float(p_lower)
        p_upper = float(p_upper)

        if sigma != 0:
            # Arbitrary check to avoid displaying an extremely high (and thus uninformative) upper-bound
            if p_upper > arbitrary_display_threshold * p:
                suffixe = f"(at least {p_lower / 1e6:.3f} M)"
            else:
                suffixe = f"in [ {p_lower / 1e6:.3f} M ; {p_upper / 1e6:.3f} M ]"
        else:
            suffixe = ""

        sales_estimate = float(p[0])
        print(
            f"Input: {app_name} ---> predicted sales: {sales_estimate / 1e6:.3f} M {suffixe}",
        )

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
