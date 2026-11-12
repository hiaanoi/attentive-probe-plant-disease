# Paired image bootstrap streams

Use NumPy `default_rng(0)` (PCG64) and `integers(0, n, size=(batch, n), dtype=np.int64)` in successive batches until 10,000 draws. The hashes in streams.json apply to concatenated little-endian int64 C-order bytes, not to an NPY container. Average correctness across fitted seeds per image first, then resample paired image differences. Quantiles are linear 2.5% and 97.5% percentiles.

The overlap-excluded PlantSeg subset uses the same recipe with n=1263 and seed=0; its raw-stream hash is recorded with its reference result. No models are fitted or called by these checks.
