
### 18. Warn on unknown `X_lags` keys

**Fix:** Emit a warning when a lag spec names a column absent from the data.
**Why:** Typos in regressor names are silently ignored, changing results with no
signal. `y_lags` is an integer in the current API and has no key-validation
issue.

### 20. Explicitly clean up external-model temp directories

**Fix:** Call `.cleanup()` before `_new_cache_dir()` replaces `self._tmpdir`, or
give the model an explicit context-managed lifecycle.
**Why:** The implementation uses `TemporaryDirectory`, but replacing the object
still relies on its finalizer for cleanup. External-model runs can therefore emit
`ResourceWarning: Implicitly cleaning up <TemporaryDirectory …>` and retain temp
directories until GC.
