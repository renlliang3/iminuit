import pytest
import numpy as np
from numpy.testing import assert_allclose
from iminuit import Minuit
from iminuit.cost import (
    UnbinnedNLL,
    BinnedNLL,
    ExtendedUnbinnedNLL,
    ExtendedBinnedNLL,
    LeastSquares,
    NormalConstraint,
)

stats = pytest.importorskip("scipy.stats")
norm = stats.norm


@pytest.fixture
def unbinned():
    np.random.seed(1)
    x = np.random.randn(1000)
    mle = (len(x), np.mean(x), np.std(x, ddof=1))
    return mle, x


@pytest.fixture
def binned(unbinned):
    mle, x = unbinned
    nx, xe = np.histogram(x, bins=50, range=(-3, 3))
    return mle, nx, xe


@pytest.mark.parametrize("verbose", (0, 1))
def test_UnbinnedNLL(unbinned, verbose):
    mle, x = unbinned

    def pdf(x, mu, sigma):
        return norm(mu, sigma).pdf(x)

    cost = UnbinnedNLL(x, pdf, verbose=verbose)
    m = Minuit(cost, mu=0, sigma=1, limit_sigma=(0, None))
    m.migrad()
    assert_allclose(m.args, mle[1:], atol=1e-3)

    # add bad value and mask it out
    cost.data[1] = np.nan
    cost.mask = np.arange(len(cost.data)) != 1
    m.migrad()
    assert_allclose(m.args, mle[1:], rtol=0.02)


@pytest.mark.parametrize("verbose", (0, 1))
def test_ExtendedUnbinnedNLL(unbinned, verbose):
    mle, x = unbinned

    def scaled_pdf(x, n, mu, sigma):
        return n, n * norm(mu, sigma).pdf(x)

    cost = ExtendedUnbinnedNLL(x, scaled_pdf, verbose=verbose)
    m = Minuit(cost, n=len(x), mu=0, sigma=1, limit_n=(0, None), limit_sigma=(0, None),)
    m.migrad()
    assert_allclose(m.args, mle, atol=1e-3)

    # add bad value and mask it out
    cost.data[1] = np.nan
    cost.mask = np.arange(len(cost.data)) != 1
    m.migrad()
    assert_allclose(m.args, mle, rtol=0.02)


@pytest.mark.parametrize("verbose", (0, 1))
def test_BinnedNLL(binned, verbose):
    mle, nx, xe = binned

    def cdf(x, mu, sigma):
        return norm(mu, sigma).cdf(x)

    cost = BinnedNLL(nx, xe, cdf, verbose=verbose)
    m = Minuit(cost, mu=0, sigma=1, limit_sigma=(0, None))
    m.migrad()
    # binning loses information compared to unbinned case
    assert_allclose(m.args, mle[1:], rtol=0.15)

    # add bad value and mask it out
    cost.n[1] = -1000
    cost.mask = np.arange(len(cost.n)) != 1
    m.migrad()
    assert_allclose(m.args, mle[1:], atol=0.04)


@pytest.mark.parametrize("verbose", (0, 1))
def test_ExtendedBinnedNLL(binned, verbose):
    mle, nx, xe = binned

    def scaled_cdf(x, n, mu, sigma):
        return n * norm(mu, sigma).cdf(x)

    cost = ExtendedBinnedNLL(nx, xe, scaled_cdf, verbose=verbose)
    m = Minuit(cost, n=mle[0], mu=0, sigma=1, limit_n=(0, None), limit_sigma=(0, None))
    m.migrad()
    # binning loses information compared to unbinned case
    assert_allclose(m.args, mle, rtol=0.15)

    # add bad value and mask it out
    cost.n[1] = -1000
    cost.mask = np.arange(len(cost.n)) != 1
    m.migrad()
    assert_allclose(m.args, mle, rtol=0.06)


@pytest.mark.parametrize("loss", ["linear", "soft_l1", np.arctan])
@pytest.mark.parametrize("verbose", (0, 1))
def test_LeastSquares(loss, verbose):
    np.random.seed(1)
    x = np.random.rand(20)
    y = 2 * x + 1
    ye = 0.1
    y += ye * np.random.randn(len(y))

    def model(x, a, b):
        return a + b * x

    cost = LeastSquares(x, y, ye, model, loss=loss, verbose=verbose)
    m = Minuit(cost, a=0, b=0)
    m.migrad()
    assert_allclose(m.args, (1, 2), rtol=0.03)
    assert cost.loss == loss
    if loss != "linear":
        cost.loss = "linear"
        assert cost.loss != loss
    m.migrad()
    assert_allclose(m.args, (1, 2), rtol=0.02)

    # add bad value and mask it out
    cost.y[1] = np.nan
    cost.mask = np.arange(len(y)) != 1
    m.migrad()
    assert_allclose(m.args, (1, 2), rtol=0.03)


def test_addable_cost_1():
    def model1(x, a):
        return a + x

    def model2(x, b):
        return b + x

    lsq1 = LeastSquares(1, 2, 3, model1)
    assert lsq1.func_code.co_varnames == ["a"]

    lsq2 = LeastSquares(1, 3, 4, model2)
    assert lsq2.func_code.co_varnames == ["b"]

    lsq3 = lsq1 + lsq2
    assert lsq3.func_code.co_varnames == ["a", "b"]

    assert lsq3(1, 2) == lsq1(1) + lsq2(2)

    m = Minuit(lsq3, a=1, b=2)
    m.migrad()
    assert m.parameters == ["a", "b"]
    assert_allclose(m.values[:], (1, 2))
    assert_allclose(m.errors[:], (3, 4))
    assert_allclose(m.np_covariance(), ((9, 0), (0, 16)), atol=1e-10)


def test_addable_cost_2():
    def model(x, a):
        return a + x

    lsq1 = LeastSquares([1], [2], [1], model)
    lsq2 = lsq1 + NormalConstraint("a", 1, 1)
    assert lsq1.func_code.co_varnames == ["a"]
    assert lsq2.func_code.co_varnames == ["a"]

    m = Minuit(lsq1, pedantic=False)
    m.migrad()
    assert_allclose(m.values[:], (1,))
    assert_allclose(m.errors[:], (1,))

    m = Minuit(lsq2, pedantic=False)
    m.migrad()
    assert_allclose(m.values[:], (1,))
    assert_allclose(m.errors[:], (np.sqrt(0.5),))
    # assert_allclose(m.np_covariance(), ((9, 0), (0, 16)), atol=1e-10)
