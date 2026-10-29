# EPFL Machine Learning - Project 1

A NumPy implementation of the six required methods and a ridge regression
baseline for the supplied heart-disease dataset.

## Running the model

The experiment was run with Python 3.14.6 and NumPy 2.5.3. To install the same
NumPy version, use Python 3.12 or newer and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The code itself also works with the course's NumPy 1.23 environment. No pandas,
scikit-learn, or other ML packages are needed.

Download the course data from the
[Project 1 competition](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1).
The data is not included in this repository. Put the three files here:

```text
data/dataset/x_train.csv
data/dataset/y_train.csv
data/dataset/x_test.csv
```

Alternatively, put the original `dataset_to_release.zip` in `data/`. The script
can extract the three CSV files from that archive. Keep the original file names
and row order.

Then run:

```bash
python run.py
```

For data stored somewhere else:

```bash
python run.py --data-dir /path/to/dataset
```

The script writes `submission.csv` and `submission.metrics.json`. The CSV contains
the test IDs and predictions in {-1, +1}. It can be uploaded to AIcrowd. The JSON
contains the local validation results. To write to a different location, use
`--output /path/to/submission.csv`. The full dataset needs several GB of RAM.

## Approach

- Use an 80/20 random split, with `np.random.default_rng(42)`.
- Drop entirely missing columns and fill missing values with column medians.
- Remove constant columns, standardize the remaining columns, and add an intercept.
- Fit ridge regression on targets 0/1 with lambda = 0.001.
- Try thresholds from 0.05 to 0.50 in steps of 0.01. Keep the one with the best
  validation F1; ties go to the smaller threshold.
- Refit preprocessing and ridge on all labeled rows, then predict the test set.

Preprocessing statistics come only from the fitting rows. Validation and test
rows use those same statistics. The ridge penalty includes the intercept weight.
Only empty and non-finite values are treated as missing; survey-specific codes
and categorical variables are not given special handling.

## Validation results

There are 328,135 labeled rows and 109,379 test rows, with 321 input features.
The split has 262,508 fitting rows and 65,627 validation rows. After preprocessing,
315 features remain, plus the intercept. About 8.83% of the labels are positive.

| Prediction rule | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Always negative | 0.9112 | 0.0000 | 0.0000 | 0.0000 |
| Ridge, threshold 0.50 | 0.9118 | 0.8197 | 0.0086 | 0.0170 |
| Ridge, threshold 0.22 | 0.8702 | 0.3455 | 0.5166 | 0.4140 |

Lowering the threshold catches more positives, but increases false positives.
The threshold was selected on this same validation split, so its F1 can be
optimistic. There is no separate final holdout or cross-validation, and no local
test score because test labels are unavailable. The saved JSON gives the full
values rather than the rounded numbers above.

## Files

- `implementations.py`: GD, SGD (one example per step), least squares, ridge,
  logistic regression, and regularized logistic regression.
- `run.py`: data loading, preprocessing, validation, fitting, and CSV output.
- `helpers.py`: the loading and submission helpers supplied by the course.
- `submission.metrics.json`: the saved full-data validation results.
- `report.pdf`: the two-page report. Its LaTeX source is in `report/`.

The six required functions return `(w, loss)`, with one-dimensional weight
vectors. MSE includes the factor 1/2. Returned losses exclude regularization
penalties. Logistic functions expect targets 0/1; their loss is mean binary
cross-entropy. SGD uses NumPy's global random generator, so call
`np.random.seed(...)` first when checking it. The prediction pipeline uses the
fixed split seed above and does not use SGD.

The public tests are available in the
[course repository](https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests).
They can be run against this repository's permanent commit URL, as described
in the course's testing instructions.
