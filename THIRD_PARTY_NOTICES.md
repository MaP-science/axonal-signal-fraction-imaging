# Third-party notices

Project-original source code is licensed under the GNU Affero General Public
License, version 3 (AGPL-3.0-only)
in [LICENSE](LICENSE). Third-party portions retain their original licenses and
copyright notices. The project license does not replace dependency licenses.
Data arrays and pretrained models have the separate scope described in
[DATA_AND_MODEL_PROVENANCE.md](DATA_AND_MODEL_PROVENANCE.md).

## DIPY-derived functions

The following functions in `asfimaging/shtools/sphericalharmonics.py` are
adapted from DIPY and retain the BSD-3-Clause terms:

| Local function | Upstream module and function |
| --- | --- |
| `cart2sphere` | `dipy.core.geometry.cart2sphere` |
| `real_sph_harm` | `dipy.reconst.shm.real_sph_harm` |
| `real_sh_descoteaux_from_index` | `dipy.reconst.shm.real_sh_descoteaux_from_index` |
| `spherical_harmonics` | `dipy.reconst.shm.spherical_harmonics` |

Copyright (c) 2008-2025, dipy developers. All rights reserved.

The complete upstream notice, conditions, and disclaimer are reproduced
verbatim in [LICENSES/DIPY-BSD-3-Clause.txt](LICENSES/DIPY-BSD-3-Clause.txt).
Keep that file with source redistributions and reproduce the notice in the
documentation or other materials accompanying binary redistributions.

Provenance was checked against DIPY 1.11.0 on 2026-11-24:

- [Geometry source](https://github.com/dipy/dipy/blob/1.11.0/dipy/core/geometry.py)
- [Spherical harmonics source](https://github.com/dipy/dipy/blob/1.11.0/dipy/reconst/shm.py)
- [Upstream license](https://github.com/dipy/dipy/blob/1.11.0/LICENSE)

The original copied release or commit and adaptation date are unknown.
The reference version and copyright notice above do not establish that
1.11.0 was the version originally copied. The local functions differ from
that reference in documentation, parameter names/signatures, warning handling,
zero-radius/angle handling, and the SciPy spherical-harmonic call. In particular,
the local implementation calls `scipy.special.sph_harm_y` directly.
These are observations from comparison, not a complete historical change log.

## Tools used to generate bundled data

`asfimaging/shtools/data/directions_325_l_24.npy` was generated with
[qspace](https://github.com/ecaruyer/qspace) (Copyright (c) 2013-2015, Emmanuel
Caruyer; Revised BSD license). No qspace code is included in this repository.
See [DATA_AND_MODEL_PROVENANCE.md](DATA_AND_MODEL_PROVENANCE.md) for details and
the requested citation.

## Installed dependencies

Dependencies are separately licensed; installing or redistributing them does
not place them under the project's AGPL-3.0 license. Preserve their upstream
license and notice files if distributing an environment, wheel bundle,
container, or executable. Some installed distributions include additional
third-party code, fonts, and native libraries with their own terms.
This file documents the third-party material copied into this repository;
it is not a complete notice inventory for a packaged runtime environment.
