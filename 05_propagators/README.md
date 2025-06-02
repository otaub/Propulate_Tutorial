# Try a different propagator

- [ ] the CMA propagator requires a CMA adapter

```python
    adapter = ActiveCMA()
    propagator = CMAPropagator(adapter, limits, rng=rng)
```
