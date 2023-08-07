import sys
from random import uniform

import numpy as np

sys.path.insert(0, "../.venv/lib/python3.10/site-packages/flash_rbf/")
sys.path.insert(0, "../.venv/lib/python3.10/site-packages/")
sys.path.insert(0, ".")

import dependencies.interpolators as py

import flash_rbf as fl

dimension = 100


def blackbox(pts):
    return 10.0 * dimension + np.sum(
        np.power(pts, 2) + 10.0 * np.cos(2.0 * np.pi * pts)
    )

bounds = (-5.0, 5.0)

x_train = np.array(
    [[uniform(bounds[0], bounds[1]) for _ in range(dimension)] for _ in range(10)]
)
y_train = np.array([blackbox(x) for x in x_train])

training_data = (x_train, y_train)

x_new = np.array(
    [[uniform(bounds[0], bounds[1]) for _ in range(dimension)] for _ in range(50)]
)


def flash_rbf_model(training_data, new_data):
    # Train the rbf model
    newrbf = fl.Rbf(training_data[0], training_data[1], "gaussian", 1.0)

    # Predicting
    pred1 = newrbf.predict(new_data[:5])

    # Rbf live update
    for data in new_data[:5]:
        y = blackbox(data)
        x_new = np.array([data])
        y_new = np.array([y])

        newrbf.update(x_new, y_new)

    # Predicting updated model
    pred2 = newrbf.predict(new_data[5:10])


def scipy_model(training_data, new_data):
    # Train the rbf model
    newrbf = py.RBFscipy(training_data[0], training_data[1], 1.0)

    # Predicting
    pred1 = newrbf.predict(new_data[:5])

    # Rbf live update
    for data in new_data[:5]:
        y = blackbox(data)
        x_new = np.array([data])
        y_new = np.array([y])

        newrbf.update(x_new, y_new)

    # Predicting updated model
    pred2 = newrbf.predict(new_data[5:10])


def numpy_model(training_data, new_data):
    # Train the rbf model
    newrbf = py.RBFnumpy(training_data[0], training_data[1], 1.0)

    # Predicting
    pred1 = newrbf.predict(new_data[:5])

    # Rbf live update
    for data in new_data[:5]:
        y = blackbox(data)
        x_new = np.array([data])
        y_new = np.array([y])

        newrbf.update(x_new, y_new)

    # Predicting updated model
    pred2 = newrbf.predict(new_data[5:10])


def test_scipy(benchmark):
    benchmark(scipy_model, training_data, x_new)


def test_flash_rbf(benchmark):
    benchmark(flash_rbf_model, training_data, x_new)


def test_numpy(benchmark):
    benchmark(numpy_model, training_data, x_new)
