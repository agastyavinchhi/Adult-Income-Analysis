# Adult Income

Predicting whether someone earns more than $50K a year using the [Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) dataset.

The point of this project was to take a messy real-world dataset from start to finish: clean it up, use some plots to figure out which features actually carry signal, train a model on them, and then check whether the features the model leans on line up with what the plots suggested.

Why is refactoring good?

Refactor today, otherwise you will end up having to refactor tomorrow!

## Files

```
main.py                     # the pipeline: load -> preprocess -> train -> evaluate
main.ipynb                  # cleaning, exploration, plots, the model, Polars benchmark
tests/                      # pytest unit and system tests
adult.csv                   # the dataset
rust_vs_python_intro.ipynb  # the Rust notebook from class
images/                     # exported plots
pyproject.toml              # dependencies and pytest config
uv.lock                     # pinned versions for reproducible installs
```

## Dataset

About 32.5K rows of 1994 US census data. Each row is a person with 14 features and a target column, `income`, which is either `<=50K` or `>50K`.

Six of the features are numeric — `age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`, and `hours.per.week` — and the other eight are text categories: `workclass`, `education`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, and `native.country`.

## What I did

The exploration lives in `main.ipynb`; the final pipeline (load, clean, train, evaluate) is in `main.py`.

Started by loading the CSV and running `head()`, `describe()`, and `info()` to understand the columns and types. Missing values are stored as `"?"` rather than as actual nulls, so Pandas doesn't flag them — I replaced them first, which showed the gaps were concentrated in `workclass` (1,836), `occupation` (1,843), and `native.country` (583). Dropping those rows along with 24 duplicates left 30,139 clean rows.

Then I did some quick exploration, filtering people working over 40 hours a week and grouping by education, and made three plots to see which features actually separate the two income groups:

- Share earning >50K by education: Prof-school and Doctorate around 75%, Bachelors 42%, HS-grad around 16%
- Age distribution by income: almost nobody under 25 earns >50K, and high earners cluster between about 35 and 55
- Hours per week by income: high earners work more hours

[![Income by education](images/income_by_education.png)](images/income_by_education.png)

## The model

The plots made it clear the features had signal, so I used all 14 of them. The numeric columns go in as they are, and the categorical ones get one-hot encoded with `pd.get_dummies()` so each category becomes its own binary column. I converted `income` into a binary `high_income` target, split the data 80/20, and trained a `GradientBoostingClassifier` on defaults with `random_state=42`.

It gets **86.5% accuracy** on the test set. Worth noting that about 75% of the dataset earns `<=50K`, so always guessing "low income" would already score 75% — the model is a real improvement on that, but not as big a jump as 86.5% sounds on its own.

Here's what it ended up relying on:

| Feature | Importance |
|---|---|
| `marital.status_Married-civ-spouse` | 0.39 |
| `education.num` | 0.20 |
| `capital.gain` | 0.19 |
| `capital.loss` | 0.06 |
| `age` | 0.06 |
| `hours.per.week` | 0.04 |

Those six account for almost all of it. Occupation shows up further down, but split across individual job categories, so no single one contributes much — `Exec-managerial` is the largest at 0.016. Education, age, and hours all matching what I saw in the plots was a good sanity check.

## Other Deliverables

At the end of `main.ipynb`, I reran the same cleaning and groupby in Polars to compare against Pandas to do some performance analysis. Polars was about 5x faster according to my analysis.

Finally, `rust_vs_python_intro.ipynb` is the Rust notebook we went over in class. I edited the cells to experiment with Rust and made changes to the notebook accordingly.

## Running it

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `pyproject.toml` lists what the project needs and `uv.lock` pins the exact versions, so every install is identical.

```bash
make install    # create .venv and install everything from uv.lock
make run        # run the pipeline (python main.py) and print test accuracy
make test       # run the unit tests with pytest
make notebook   # open main.ipynb in Jupyter
make execute    # run the whole notebook headless and save outputs
make clean      # remove checkpoints and caches
```
