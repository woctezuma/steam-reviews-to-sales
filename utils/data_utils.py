import csv


def get_input_fname():
    return "data/games_achievements_players_2018-07-01.csv"


def load_input_data(fname=None, skip_header=True):
    if fname is None:
        fname = get_input_fname()

    data = {}

    with open(fname, encoding="utf8", errors="ignore") as f:
        f_reader = csv.reader(f)

        if skip_header:
            next(f_reader)

        for row in f_reader:
            app_name = row[0]
            num_players = row[1]
            app_id = str(row[2])

            data[app_id] = {}
            data[app_id]["name"] = app_name.strip()
            data[app_id]["sales"] = int(num_players.replace(",", ""))

    return data


def load_app_ids(data=None, fname=None, verbose=True):
    if fname is None:
        fname = get_input_fname()

    if data is None:
        data = load_input_data(fname, skip_header=True)

    app_ids = [int(app_id) for app_id in data]

    if verbose:
        print(f"#apps = {len(app_ids)}")

    return app_ids


def main():
    parent_folder = "../"
    fname = parent_folder + get_input_fname()
    data = load_input_data(fname, skip_header=True)
    app_ids = load_app_ids(data)

    return True


if __name__ == "__main__":
    main()
