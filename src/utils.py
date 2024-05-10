import sklearn
from typing import Callable

def scoring_check(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError:
            if kwargs['scoring'] not in sklearn.metrics.get_scorer_names():
                raise ValueError(
                    f"Invalid scoring metric: {kwargs['scoring']}. Select one of the following: {list(sklearn.metrics.get_scorer_names())}"
                )
    return wrapper
