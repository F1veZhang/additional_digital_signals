# Recorded computational environment

These versions are taken from preserved execution metadata, not inferred from the machine used to prepare this release.

| Component | Recorded version |
|---|---|
| Python | 3.11.13, Anaconda build |
| Operating system | Windows 10, AMD64 |
| NumPy | 2.0.1 |
| pandas | 2.3.2 |
| SciPy | 1.16.0 |
| scikit-learn | 1.7.1 |
| CatBoost | 1.2.8 |
| PyTorch | 2.8.0+cu129 |
| CUDA build / cuDNN | 12.9 / 91002 |
| GPU | NVIDIA GeForce RTX 4070 SUPER |

`requirements.txt` pins the recorded package versions where known. Its `torch==2.8.0` entry does not by itself reproduce the platform-specific `+cu129` build. The inspected metadata did not record the `threadpoolctl` version. This is therefore not a complete environment lock, and the environment was not reinstalled or computationally revalidated while assembling the release.

Metadata sources: `response_online_offline_20261013/runtime_versions.json` and `response_multiplatform_recovery_20261015/audit/hardware_and_runtime.json`. Local executable paths and hardware audit prose are not copied into the public repository.
