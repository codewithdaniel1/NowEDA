"""Shared dtype predicates for pandas 1.x, 2.x and 3.x."""
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


def is_textual(series):
    """Include object, extension strings, and categorical values."""
    dtype = series.dtype
    return (is_object_dtype(dtype) or is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype))
