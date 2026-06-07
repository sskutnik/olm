# TODO

## SCALE Artifact Standards Needed by OLM

OLM's low-order consistency workflow needs TRITON and Polaris artifacts to carry
the same basis-aware depletion metadata in their machine-readable files. Current
workarounds parse product-specific output tables and F71 cases because F33 and
F71 files do not yet expose the information consistently enough for a single
artifact path.

Needed standards:

- F33 and F71 files should write burnup on the same basis used to generate the
  library. If the high-order model uses a fuel basis, the library burnup stored
  in the F33 and the corresponding F71 case should be fuel-basis burnup.
- F33 and F71 files should expose the selected depletion basis explicitly:
  product, basis name, material or case ids, initial heavy metal, power history,
  burn length, cumulative burnup, and library position.
- TRITON and Polaris should use consistent semantics for time-dependent power:
  whether a power value is an interval average or a value at a time point must be
  explicit and machine-readable.
- F71 inventory cases and F33 time-dependent libraries should have a stable,
  documented relationship. OLM should not need product-specific case guessing
  such as TRITON source/fuel case `-2` or Polaris material-class output parsing.
- Burnup grids written to text output, F33 tags, and F71 info tables should agree
  to normal floating-point precision for the same basis. OLM should not need to
  choose between a text-output burnup grid and an F71-derived `energy/initialhm`
  grid.
- Product outputs should identify single-zone versus multi-zone fuel models so a
  low-order consistency check can distinguish true one-zone comparisons from
  assembly-average homogenization checks.

OLM implementation direction:

- Prefer explicit fuel-basis high-order models for low-order consistency checks.
- Treat system-power-to-fuel-power backout as a legacy compatibility approach,
  not the default validation strategy.
- For TRITON, use the basis-aware output library table for arpdata burnup tags
  and ORIGAMI cycle history until F33/F71 files carry this metadata
  consistently.
- For Polaris, use the explicit `FUEL` basis case and fuel F33 artifact.
