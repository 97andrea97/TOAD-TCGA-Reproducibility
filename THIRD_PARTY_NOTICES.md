This repository includes code derived from third-party open-source projects.

## TOAD (AGPL-3.0)
- Upstream: https://github.com/mahmoodlab/TOAD
- License: AGPL-3.0 (see LICENSE and licenses/TOAD_AGPL-3.0.txt)
- Notes: This repository contains a modified version of TOAD for tumor site classification.
- Included components: all scripts.
- Modifications in this repository:
    - primary-site classification only (architecture/loss adapted)
    - learning-curve runner for label fractions (x% training data)
    - patient-level stratification enabled by default
    - additional logging and reproducibility utilities
    - containerized execution support
    - release of pretrained weights


## CLAM (GPL-3.0)
- Upstream: https://github.com/mahmoodlab/CLAM
- License: GPL-3.0 (see licenses/CLAM_GPL-3.0.txt)
- Notes: This repository includes a subset of CLAM code used for preprocessing.

## Notes
- This repository is distributed under the GNU Affero General Public License v3.0 (AGPL-3.0). See LICENSE.
- Copyright and license headers from upstream-derived files should be preserved.
EOF

