# TRACE knowledge-source inventory

Snapshot date: **2026-11-10**

All files under `raw/` are immutable source snapshots. Curated TRACE records
must be generated into a different directory and must retain source IDs,
versions, licences and assertion provenance.

| File | Purpose | Version/release | Licence/source rule |
|---|---|---|---|
| `mondo-rare.json` | canonical rare-disease identifiers and hierarchy | 2026-11-03 | CC BY 4.0 |
| `hp.json` | human phenotype/finding identifiers | 2026-09-01 | preserve HPO attribution and current project terms |
| `uberon-basic.obo` | retinal/anatomical identifiers | 2026-11-05 | CC BY 3.0 |
| `ro.json` | standard biomedical relations | 2026-09-04 | CC0 1.0 |
| `eco.obo` | evidence types | 2026-07-10 | CC0 1.0 |
| `orphanet_nomenclature_en_product1.xml` | ORPHA names and alignments | 2026-06-23 | CC BY 4.0 |
| `orphanet_phenotypes_en_product4.xml` | ORPHA–HPO phenotype associations | 2026-06-23 | CC BY 4.0 |
| `orphanet_genes_en_product6.xml` | Orphanet gene associations | 2026-06-23 | CC BY 4.0; association is not evidence strength |
| `hgnc_complete_set.json` | approved human gene identifiers and symbols | downloaded 2026-11-10 | CC0 1.0 |
| `clingen_gene_disease_validity.csv` | expert gene–disease validity classifications | generated 2026-11-10 | cite ClinGen; research decision support only |
| `biolink-model.yaml` | biomedical node/edge schema vocabulary | downloaded 2026-11-10 | upstream Biolink terms and attribution |

## Important exclusions

- OMIM bulk data is not included because its redistribution and download terms
  are restricted.
- UMLS and SNOMED CT bulk releases are not included because they require their
  own licences/accounts and are unnecessary for the pilot.
- AAO guideline text is not copied. TRACE assertions may cite a guideline and
  store a short curator-written paraphrase with a precise locator.
- An ontology association is not automatically a clinically confirmed imaging
  relationship. Imaging assertions still require literature or guideline
  evidence and a review status.
