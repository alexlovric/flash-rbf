class Rbf:
    """
    Radial basis function interpolator.
    """

    def __init__(
        self,
        x: list[float],
        y: list[float],
        kernel: str | None,
        epsilon: float | None,
    ) -> None:
        """Instantiate a radial basis function interpolator.

        Parameters
        ----------
        x : list[float]
            A n*m matrix containing the training data points.
        y : list[float]
            A n vector containing the corresponding training output values.
        kernel : str | None
            An optional kernel function name.
        epsilon : float | None
            An optional bandwidth parameter for the kernel.
            Defaults to 1. if `None` given.
        """
        self.x = x
        self.y = y
        self.kernel = kernel
        self.epsilon = epsilon
