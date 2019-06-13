#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `diversity` package."""

import pytest
import diversity as div
from diversity.datasets import *

def test_skew_pmf():
    As = np.linspace(0.01, 10, 25)
    Ns = np.arange(2, 20)
    x = np.array([np.linalg.norm(1-skew_pmf(a, n).sum()) for a in As for n in Ns])
    assert(np.all(np.less(x, 1e-6)))

def test_tsallis_renyi_entropy():
    ntests = 100
    for i in range(ntests):
        q = np.random.uniform(0, 10)
        p = skew_pmf(np.random.uniform(0.01, 2), np.random.randint(100)+1)
        Tq = div.tsallis_entropy(p, q)
        Rq = div.renyi_entropy(p, q)
        RTq = (np.exp((1-q)*Rq)-1)/(1-q)
        assert(np.all(np.less(np.linalg.norm(Tq-RTq), 1e-8)))

def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string
