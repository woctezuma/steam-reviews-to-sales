import numpy as np


def get_test_apps():
    all_features = ["review_score", "total_positive", "total_negative", "total_reviews"]

    # Reference: https://store.steampowered.com/appreviews/892970?json=1&num_per_page=0&language=all&purchase_type=all&filter_offtopic_activity=0
    test_apps = {
        "Valheim": {"appID": 892970, "data": [9, 246944, 10706, 257650]},
        "Iratus": {"appID": 807120, "data": [8, 4606, 698, 5304]},
    }

    return test_apps, all_features


def check_test_apps(model, features):
    test_apps, all_features = get_test_apps()

    for app_name in test_apps.keys():
        v_test = test_apps[app_name]["data"]

        v = [v_test[i] for (i, f) in enumerate(all_features) if f in features]
        v = np.array(v).reshape(1, -1)

        p = model.predict(v)

        print(f"Input: {app_name} ---> predicted sales: {p[0] / 1e6:.3f} M")

    return
