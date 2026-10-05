# Data

This project uses the **Credit Card Fraud Detection** dataset: 284,807 card
payments by European cardholders over two days in September 2013, 492 of
them fraud. It was released by Worldline and the Machine Learning Group of
the Université Libre de Bruxelles (ULB).

The file is not stored in this repository. To get it:

1. Download `creditcard.csv` from Kaggle:
   <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
   (or with the Kaggle CLI:
   `kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip`)
2. Put it here, as `data/creditcard.csv`.

Check you have the right file: 284,807 rows, 31 columns, 492 rows with
`Class = 1`.

The dataset is released under the Open Database License (ODbL 1.0).

Citation: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca
Bontempi. *Calibrating Probability with Undersampling for Unbalanced
Classification.* IEEE Symposium on Computational Intelligence and Data
Mining (CIDM), 2015.
