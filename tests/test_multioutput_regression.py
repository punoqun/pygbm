import os
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest
from sklearn.utils.testing import assert_raises_regex
from sklearn.datasets import make_classification, make_regression

from pygbm import GradientBoostingClassifier
from pygbm import GradientBoostingRegressor
from pygbm.binning import BinMapper
from sklearn.model_selection import train_test_split
from sklearn.random_projection import SparseRandomProjection
from sklearn.utils import shuffle
import pandas as pd
from sklearn.metrics import r2_score


def test_atp1d():
    df = pd.read_csv('/home/Kenny/Documents/atp1d.csv')
    target = df.loc[:, df.columns.str.startswith('LBL')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    target = SparseRandomProjection(n_components=1, random_state=42).fit_transform(X=target)
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42, shuffle=True)
    gb = GradientBoostingRegressor(
        verbose=1,
        random_state=42,
        # learning_rate=0.1,
        max_iter=100,
        # min_samples_leaf=15
    )
    gb.fit(X_train, y_train)
    y_preds = gb.predict(X_test)
    pd.DataFrame(y_preds).to_csv('/home/Kenny/PycharmProjects/DreamPretermBirth/CompetitionPart2/try.csv',index=False)
    r2 = r2_score(y_test, y_preds, multioutput='uniform_average')
    print(r2)

def test_edm():
    df = pd.read_csv('/home/Kenny/Documents/edm.csv')
    target = df.loc[:, ['DFlow', 'DGap']]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42, shuffle=False)
    gb = GradientBoostingRegressor(
        verbose=1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_preds = gb.predict_multi(X_test)
    r2 = r2_score(y_test, y_preds, multioutput='uniform_average')
    print(r2)


def test_scm1d():
    df = pd.read_csv('/home/Kenny/Documents/scm1d.csv')
    target = df.loc[:, df.columns.str.contains('L')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=1658.0/8145.0, random_state=42, shuffle=False)
    gb = GradientBoostingRegressor(
        verbose=1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_preds = gb.predict_multi(X_test, np.shape(y_test)[1])
    r2 = r2_score(y_test, y_preds, multioutput='uniform_average')
    print(r2)


def test_scm20d():
    df = pd.read_csv('/home/Kenny/Documents/scm20d.csv')
    target = df.loc[:, df.columns.str.contains('L')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=1503.0/7463.0, random_state=42, shuffle=False)
    gb = GradientBoostingRegressor(
        verbose=1,
        random_state=42,
        max_iter=100,
        min_samples_leaf=10
    )
    gb.fit(X_train, y_train)
    y_preds = gb.predict_multi(X_test, np.shape(y_test)[1])
    r2 = r2_score(y_test, y_preds, multioutput='variance_weighted')
    print(r2)


def test_wq():
    df = pd.read_csv('/home/Kenny/Documents/water-quality.csv')
    target = df.loc[:, df.columns.str.startswith('x')]
    df.drop(target.columns, axis=1, inplace=True)
    df, target = df.to_numpy(), target.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.5, random_state=42, shuffle=False)
    gb = GradientBoostingRegressor(
        verbose=1,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_preds = gb.predict_multi(X_test, np.shape(y_test)[1])
    r2 = r2_score(y_test, y_preds, multioutput='variance_weighted')
    print(r2)
