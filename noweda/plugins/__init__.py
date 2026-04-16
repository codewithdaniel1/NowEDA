from .missing import MissingDataPlugin
from .stats import StatsPlugin
from .schema import SchemaPlugin
from .duplicates import DuplicatesPlugin
from .correlation import CorrelationPlugin
from .outliers import OutlierPlugin
from .pii import PIIDetectorPlugin
from .encoding import EncodingDetectionPlugin


def default_plugins():
    return [
        SchemaPlugin(),
        StatsPlugin(),
        MissingDataPlugin(),
        DuplicatesPlugin(),
        CorrelationPlugin(),
        OutlierPlugin(),
        PIIDetectorPlugin(),
        EncodingDetectionPlugin(),
    ]
