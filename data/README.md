# Dataset Information

## SMS Spam Collection Dataset

### Overview

This directory contains the SMS Spam Collection dataset used for training and evaluating spam classification models.

### Dataset Source

- **Name**: SMS Spam Collection
- **Original Source**: UCI Machine Learning Repository
- **Kaggle Link**: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
- **UCI Link**: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

### How to Download

#### Option 1: Kaggle (Recommended)

1. Visit: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Download `spam.csv`
3. Place in `data/raw/spam.csv`

#### Option 2: Direct Download

```bash
# Using kaggle CLI (requires kaggle credentials)
kaggle datasets download uciml/sms-spam-collection-dataset
unzip sms-spam-collection-dataset.zip -d data/raw/
```

### Dataset Details

| Attribute | Description |
|-----------|-------------|
| Format | CSV |
| Encoding | Latin-1 |
| Total Records | 5,574 |
| Features | 2 (label, message) |

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Ham (Legitimate) | 4,827 | 86.6% |
| Spam | 747 | 13.4% |

### Column Description

| Column | Type | Description |
|--------|------|-------------|
| v1 (label) | String | Message classification ('ham' or 'spam') |
| v2 (message) | String | SMS message text content |

### Sample Records

```
v1,v2
ham,"Go until jurong point, crazy.. Available only in bugis..."
ham,"Ok lar... Joking wif u oni..."
spam,"Free entry in 2 a wkly comp to win FA Cup final tkts..."
ham,"U dun say so early hor... U c already then say..."
spam,"WINNER!! As a valued network customer you have been selected..."
```

### Data Files

After preprocessing, the following files are created:

```
data/
├── raw/
│   └── spam.csv              # Original dataset (you download this)
├── processed/
│   ├── analyzed_data.csv     # After EDA notebook
│   └── preprocessed_spam.csv # After preprocessing notebook
└── README.md                 # This file
```

### Citation

If you use this dataset, please cite:

```
Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A.
Contributions to the Study of SMS Spam Filtering: New Collection and Results.
Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11),
Mountain View, CA, USA, 2011.
```

### License

The dataset is provided for research purposes. Please check the original source for usage terms.

### Notes

- The raw CSV uses Latin-1 encoding (not UTF-8)
- Some columns beyond v1 and v2 may exist but are not needed
- Ham/spam labels are lowercase strings
