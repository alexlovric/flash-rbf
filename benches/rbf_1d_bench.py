from random import uniform
import sys
import numpy as np

sys.path.insert(0, "../.venv/lib/python3.10/site-packages/flash_rbf/")
sys.path.insert(0, "../.venv/lib/python3.10/site-packages/")
sys.path.insert(0, ".")

import flash_rbf as fl
from dependencies.interpolators import RBFscipy

bounds = (0.1, 5.0)
random_data = np.array([uniform(bounds[0], bounds[1]) for _ in range(50)])


def flash_rbf_model(data):
    # Generate some test data
    def blackbox(x):
        return x**2 + 2.0 * np.sin(2.0 * np.pi * x)

    bounds = (0.1, 5.0)
    x_train = np.linspace(bounds[0], bounds[1], 10)
    x_train = x_train.reshape(-1, 1)
    y_train = blackbox(x_train)

    # Train the rbf model
    newrbf = fl.Rbf(x_train, y_train, "gaussian", 1.0)

    # Creating random data to predict on
    x_guess = np.empty((0, 1))
    for i in range(5):
        x_guess = np.append(x_guess, np.array([[data[i]]]), axis=0)

    # Predicting
    pred1 = newrbf.predict(x_guess)

    # Rbf live update
    for _ in range(5):
        x = uniform(bounds[0], bounds[1])
        y = blackbox(x)
        x_new = np.array([[x]])
        y_new = np.array([y])

        newrbf.update(x_new, y_new)

    # Predicting updated model
    x_guess = np.empty((0, 1))
    for i in range(5, 10):
        x_guess = np.append(x_guess, np.array([[data[i]]]), axis=0)
    pred2 = newrbf.predict(x_new)


def scipy_model(data):
    # Generate some test data
    def blackbox(x):
        return x**2 + 2.0 * np.sin(2.0 * np.pi * x)

    bounds = (0.1, 5.0)
    x_train = np.linspace(bounds[0], bounds[1], 10)
    x_train = x_train.reshape(-1, 1)
    y_train = blackbox(x_train)

    # Train the rbf model
    newrbf = RBFscipy(x_train, y_train, 1.0)

    # Creating random data to predict on
    x_guess = np.empty((0, 1))
    for i in range(5):
        x_guess = np.append(x_guess, np.array([[data[i]]]), axis=0)

    # Predicting
    pred1 = newrbf.predict(x_guess)

    # Rbf live update
    for _ in range(5):
        x = uniform(bounds[0], bounds[1])
        y = blackbox(x)
        x_new = np.array([[x]])
        y_new = np.array([y])

        newrbf.update(x_new, y_new)

    # Predicting updated model
    x_guess = np.empty((0, 1))
    for i in range(5, 10):
        x_guess = np.append(x_guess, np.array([[data[i]]]), axis=0)
    pred2 = newrbf.predict(x_new)


def test_python_core(benchmark):
    benchmark(scipy_model, random_data)


def test_flash_rbf(benchmark):
    benchmark(flash_rbf_model, random_data)
