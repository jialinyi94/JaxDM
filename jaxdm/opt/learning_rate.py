def exponent_scheduler(init_value: float, exponent: float = 1/2):
    step = 1
    while True:
        lr = step / (init_value) ** (exponent)
        yield lr
        step += 1
