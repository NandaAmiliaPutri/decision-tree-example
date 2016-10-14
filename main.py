import sys

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score


def main():
    fname = sys.argv[1]
    df = pd.read_csv(fname, comment='#')

    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=4,
        presort=True,
    )
    train_x = df.drop('in_sf', axis=1).values
    train_y = df['in_sf'].values

    rs = ShuffleSplit(n_splits=10, test_size=0.2)
    score = []
    for tr_idx, te_idx in rs.split(train_x):
        subtrain_x = train_x[tr_idx, :]
        subtrain_y = train_y[tr_idx]
        validation_x = train_x[te_idx, :]
        validation_y = train_y[te_idx]

        clf.fit(subtrain_x, subtrain_y)
        pred_class = clf.predict(validation_x)
        pred_prob = clf.predict_proba(validation_x)
        print(pred_prob)
        print(pred_class)
        print('accuracy: {}'.format(clf.score(validation_x, validation_y)))
        print('ROCAUC: {}'.format(roc_auc_score(validation_y, pred_prob[:, 1])))


if __name__ == '__main__':
    main()
