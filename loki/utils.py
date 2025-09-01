# loki/utils.py
import time
import logging
from functools import wraps


def time_it(func):
    """Декоратор для замера времени выполнения синхронных функций."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.debug(
            f"TIMING: {func.__name__} took {end_time - start_time:.4f} seconds."
        )
        return result

    return wrapper


def async_time_it(func):
    """Декоратор для замера времени выполнения асинхронных функций."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        logging.debug(
            f"TIMING: {func.__name__} took {end_time - start_time:.4f} seconds."
        )
        return result

    return wrapper
