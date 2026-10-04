# Original synthetic fixtures

Purpose: arithmetic, gradient, protocol and failure-mode diagnostics. Fixtures in `tests/` are authored for this project, contain no external personal data and have exact constructed outcomes. The Bernoulli correctness case has probability .7 and a confidence optimum at .70 under expected squared score. Gradient tests compare finite differences and many independent eight-sample leave-one-out groups against exact derivatives.

These fixtures are tests, not training examples or natural-data evaluation evidence. They support implementation correctness within tested numerical tolerances, not generalization, calibration on deployed requests, or low operational error rates. The HTTP fake is only a server contract fixture; real-artifact equivalence is recorded separately.
