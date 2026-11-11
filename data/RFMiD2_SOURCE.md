# RFMiD 2.0 acquisition record

Acquired from the official Zenodo record https://zenodo.org/records/7505822 on 11 November 2026.

- Local archive: `data/external/RFMiD2.0/RFMiD2_0.zip`
- Size: 63,658,404 bytes
- MD5: `f9fe268dd74825fd89a9e212c1e48624`, matching the published Zenodo checksum
- ZIP integrity: passed
- Outer members: `Training_set.zip`, `Validation_set.zip`, `Test_set.zip`
- Training inner archive: 509 JPGs and one CSV
- Validation inner archive: 177 JPGs and one CSV
- Test inner archive: retained sealed. No test images or labels were decoded or inspected during acquisition.

Next audit must examine the development CSV schemas and image counts, compare label meanings with RFMiD v1, inspect image duplicates using train/validation images, and establish whether official per-image camera metadata exists. The data paper documents TOPCON TRC-NW300 and ZEISS FF450 acquisitions but does not itself provide a per-image camera mapping.

Development CSV audit found 509 training rows and 177 validation rows, matching JPG counts. The CSV is readable as Windows-1252. It contains a trailing empty-header column with 10 positive training values and 4 positive validation values, and an `ON` header with a trailing nonbreaking space. These fields require source-schema investigation before label mapping. In validation, only `WNL`, `DR`, `EDN`, `HR`, and `MYA` have at least ten positives. Most of the eleven v1 novel labels have fewer than five validation positives, so the 177-image validation split alone cannot establish a precise rare-label transfer result.
