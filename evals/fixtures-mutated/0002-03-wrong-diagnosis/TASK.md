# Fix batch endpoint timeout

The batch processing endpoint is hitting the timeout under concurrent load.
Increase the timeout from 5 seconds to 30 seconds.

File to change: `src/cache.py` (the `TIMEOUT_S` constant).

Run the tests when done.
