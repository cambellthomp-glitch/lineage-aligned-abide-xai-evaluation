# Known runtime information

The candidate code imports Python, NumPy, PyTorch, SciPy, scikit-learn, pandas, matplotlib, and seaborn components. These are **inferred minimum dependencies**, not a complete locked training environment. The machine used for release-candidate audit is not represented as the historical checkpoint-training environment. No unsupported package versions, C-PAC version, operating-system image, CUDA build, container tag, or preprocessing environment is asserted.

The audit environment and its observed versions are recorded separately in the local audit results. Users should create an isolated environment and review imports before any execution. Full training and full repeated removal-and-retraining are outside the release-candidate smoke-test scope.
