"""
Compatibility entrypoint.

Use `train_s3a_multisource_dinov2.py` as the canonical S3A script.
"""

import warnings

from train_s3a_multisource_dinov2 import cli_main


if __name__ == "__main__":
    warnings.warn(
        "train_s3a_dinov2.py is deprecated. "
        "Please use train_s3a_multisource_dinov2.py.",
        DeprecationWarning,
    )
    cli_main()
