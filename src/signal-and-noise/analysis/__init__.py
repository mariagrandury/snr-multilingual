"""Every analysis script imports this package, so the CSV float format is set
here once: twelve significant digits, exact for counts below 1e12 such as
tokens. The full repr differs in its last digits between machines and BLAS
builds, and every regeneration then rewrote unchanged LFS tables; a caller's
own `float_format=` still wins."""

from functools import partialmethod

import pandas as pd

pd.DataFrame.to_csv = partialmethod(pd.DataFrame.to_csv, float_format="%.12g")
pd.Series.to_csv = partialmethod(pd.Series.to_csv, float_format="%.12g")
