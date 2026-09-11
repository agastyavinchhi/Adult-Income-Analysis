# Adult Income

Predicting whether someone earns more than $50K a year using the [Adult Census Income](https://archive.ics.uci.edu/dataset/2/adult) dataset.

## Dataset

About 32.5K rows of 1994 US census data. Each row is a person with 14 features (age, education, occupation, hours per week, etc.) and a target column, `income`, which is either `<=50K` or `>50K`.

## What I did (`main.ipynb`)

Started by loading the CSV and running `head()`, `describe()` and `info()` to understand the columns and types. Missing values are stored as `"?"`, so I replaced those, dropped them along with 24 duplicate rows, and ended up with 30,139 clean rows.

Then I did some quick exploration, filtering people working over 40 hours and grouping by education, and made three plots to see which features actually separate the two income groups:

- Share earning >50K by education: Prof-school and Doctorate around 75%, HS-grad around 16%
- Age distribution by income: almost nobody under 25 earns >50K
- Hours per week by income: high earners work more hours

![Income by education](images/income_by_education.png)

That was enough to convince me the features had signal, so I one-hot encoded the categorical columns and trained a `GradientBoostingClassifier`. It gets 86.5% accuracy on the test set, with marital status, education and capital gain as the most important features.

At the end I reran the same cleaning and groupby in Polars to compare against Pandas to do some performance analysis. Polars was about 5x faster according to my analysis.

Finally, `rust_vs_python_intro.ipynb` is the Rust notebook we went over class. I edited the cells to experiment with Rust and made changes to the notebook accordingly.

## Running it

```bash
make install   # install dependencies from requirements.txt
make run       # open the notebook in Jupyter
make execute   # run the whole notebook headless and save outputs
```

Use a different interpreter with `make install PYTHON=/path/to/python`.
