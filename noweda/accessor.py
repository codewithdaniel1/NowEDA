import pandas as pd
from noweda.core.engine import AutoEDAEngine
from noweda.plugins import default_plugins

@pd.api.extensions.register_dataframe_accessor("noweda")
class NowEDAAccessor:

    def __init__(self, pandas_obj):
        self._df = pandas_obj
        self._report = None

    def _ensure_analyzed(self):
        if self._report is None:
            engine = AutoEDAEngine(default_plugins())
            self._report = engine.run_df(self._df)

    def summary(self):
        self._ensure_analyzed()
        return self._report["results"]

    def insights(self):
        self._ensure_analyzed()
        return self._report["insights"]

    def score(self):
        self._ensure_analyzed()
        return self._report["scores"]

    def report(self):
        self._ensure_analyzed()
        return self._report
