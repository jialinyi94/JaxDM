def exponent_scheduler(init_value: float, exponent: float = 1/2):
    """Learning rate decays according to a power function on the steps.

    Parameters
    ----------
    init_value : float
        initial learning rate.
    exponent : float, optional
        the order of exponent, by default 1/2

    Yields
    ------
    float
        the learning rate
    """
    step = 1
    while True:
        lr = init_value / (step) ** (exponent)
        yield lr
        step += 1
