# Steam reviews to sales

This repository contains Python code to study the relationship between review numbers and sales for Steam games.

## Requirements

- Install the latest version of [Python 3.X][python-download].
- Install the required packages:

```bash
pip install -r requirements.txt
pip install git+https://github.com/google/pwlfit.git
```

[python-download]: <https://www.python.org/downloads/>

## Usage

- To download review numbers as they were on July 1, 2018 for 13,275 games of interest, run:

```bash
python download_data.py
```

Data is downloaded to `data/review_stats.json`.

- To filter extreme values and display a scatter plot of the data, run:

```bash
python analyze_data.py
```

- To fit a model to the data, run:

```bash
python fit_model.py
```

## References

### Data leak for sales

- Ars Technica's article: [Valve leaks Steam game player counts; we have the numbers][arstechnica18-article], July 2018
- Ars Technica's supplemental data: [snapshot of "players estimate"][arstechnica18-data] for 13,281 games

### Data download for reviews

- Steam API's [`appreviews`][steamapi-getreviews] endpoint, to get a dump of reviews on an application in Steam
- [`steamreviews`][pypi-steamreviews]: a PyPI package to download Steam reviews

### Regression tools and documentation

- [`mapie`][pypi-mapie]: Model-Agnostic Prediction Interval Estimator (MAPIE),
- [`pwlfit`][pypi-pwlfit]: Google's library to fit piece-wise linear functions,
- [`statsmodels`][pypi-statsmodels]: econometric and statistical modeling,
- [`tidfit`][pypi-tidfit]: a wrapper around SciPy's `curve_fit` for 1D signals.
- Some documentation by [`scipy`][pypi-scipy] and [`sklearn`][pypi-sklearn]:
  - [supervised learning][sklearn-supervised-learning-doc]
  - [common pitfalls with linear regression][sklearn-common-pitfalls-doc]
  - [losses for robust fitting][scipy-robust-fitting-doc]

### Blog posts

- Jake Birkett, [How to estimate how many sales a Steam game has made][birkett15], March 2015
- Jake Birkett, [Using Steam reviews to estimate sales][birkett18], May 2018
- Simon Carless, [How that game sold on Steam, using the 'NB number'][carless20], August 2020
- VG Insights, [How to Estimate Steam Video Game Sales?][vginsights21], August 2021
- Simon Carless, [Epic advances, Steam reviews, oh my!][carless21], August 2021

<!-- Definitions -->

[arstechnica18-article]: <https://arstechnica.com/gaming/2018/07/steam-data-leak-reveals-precise-player-count-for-thousands-of-games/>
[arstechnica18-data]: <http://www.arstechnica.com/wp-content/uploads/2018/07/games_achievements_players_2018-07-01.csv>

[steamapi-getreviews]: <https://partner.steamgames.com/doc/store/getreviews>
[pypi-steamreviews]: <https://github.com/woctezuma/download-steam-reviews>

[pypi-mapie]: <https://github.com/scikit-learn-contrib/MAPIE>
[pypi-pwlfit]: <https://github.com/google/pwlfit>
[pypi-statsmodels]: <https://github.com/statsmodels/statsmodels>
[pypi-tidfit]: <https://github.com/aminnj/tidfit>
[pypi-scipy]: <https://github.com/scipy/scipy>
[pypi-sklearn]: <https://github.com/scikit-learn/scikit-learn>
[sklearn-supervised-learning-doc]: <https://scikit-learn.org/stable/supervised_learning.html>
[sklearn-common-pitfalls-doc]: <https://scikit-learn.org/stable/auto_examples/inspection/plot_linear_model_coefficient_interpretation.html>
[scipy-robust-fitting-doc]: <https://scipy-cookbook.readthedocs.io/items/robust_regression.html>

[birkett15]: <https://greyaliengames.com/blog/how-to-estimate-how-many-sales-a-steam-game-has-made/>
[birkett18]: <https://www.gamasutra.com/blogs/JakeBirkett/20180504/317366/Using_Steam_reviews_to_estimate_sales.php>
[carless20]: <https://newsletter.gamediscover.co/p/how-that-game-sold-on-steam-using>
[vginsights21]: <https://vginsights.com/insights/article/how-to-estimate-steam-video-game-sales/>
[carless21]: <https://newsletter.gamediscover.co/p/epic-advances-steam-reviews-oh-my>

