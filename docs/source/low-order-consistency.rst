Low-Order Consistency Checks
============================

OLM compares each high-order depletion calculation against an ORIGAMI
calculation that uses the assembled ORIGEN library. The check is intended to
answer one question: if the library and the ORIGAMI input represent the same
fuel basis, do the inventories agree closely enough?

Inventory Metric
----------------

The primary comparison metric is :code:`grams_per_initial_hm`, reported as
:code:`g/gIHM`. For nuclide :math:`i`, OLM converts moles to grams and divides
by the initial heavy-metal mass for that state point. This avoids a failure mode
of atom-fraction comparisons: a fuel-only ORIGAMI calculation and a high-order
calculation that includes different non-fuel atoms can have different atom
fraction denominators even when the fuel nuclide masses are consistent.

OLM still computes atom-fraction summaries internally because they are useful
for diagnosis, but quick-validation dashboards and example checks should use
and report :code:`g/gIHM`.

Time-Zero Expectation
---------------------

At time zero, a consistent high-order and low-order setup should agree exactly
within serialization precision. A nonzero time-zero error usually means one of
these is wrong:

* the high-order fuel basis does not match the ORIGAMI fuel basis;
* the ORIGAMI template omitted an assembly-average constituent such as Gd2O3 or
  Cr2O3;
* the wrong high-order material/case was extracted;
* the inventory was compared with an atom-fraction denominator that includes
  different atoms.

Convergence Controls
--------------------

The ORIGAMI check templates require:

* :code:`convergence_control.nlib`
* :code:`convergence_control.nburn`

These names are injected by :code:`LowOrderConsistency`; they should not be
invented independently in templates. The public check parameters are:

* :code:`nlib_start`, :code:`nlib_max`
* :code:`nburn_start`, :code:`nburn_max`
* :code:`q1_stop_criteria`, :code:`q2_stop_criteria`

For each :code:`nburn`, OLM doubles :code:`nlib` until both :code:`q1` and
:code:`q2` change by no more than the configured stop criteria, or until
:code:`nlib_max` is reached. After an :code:`nlib` pass converges, OLM doubles
:code:`nburn` and repeats the :code:`nlib` convergence pass. The final
:code:`nburn` pass is accepted only when its final :code:`q1` and :code:`q2`
are stable relative to the previous :code:`nburn` pass, or when the configured
range has no variation.

If either convergence loop reaches its maximum before stabilizing, the check
fails even if the absolute :code:`q1` and :code:`q2` targets pass.

These controls only test convergence of the low-order ORIGAMI reconstruction
against the high-order data that already exist. They do not prove that the
high-order TRITON or Polaris depletion calculation is itself converged in
depletion space.

High-order depletion-space convergence is a prerequisite for a meaningful
high-order/low-order consistency check. The transport approximation can be
coarse, and even physically wrong, as long as it is deterministic and produces
a stable depletion-space reference. If the high-order calculation is not
sufficiently converged in depletion space, there is no defensible target for
ORIGAMI to reconstruct and no reason to expect :code:`q1` or :code:`q2` to
stabilize.

The high-order reference must converge both the number densities and the
one-group cross sections represented in the ORIGEN library. In practice, cross
sections usually vary more slowly than number densities, so convergence of the
number densities is a strong practical indicator that the cross-section
representation is also converged enough for the consistency check. The
high-order inventory and cross sections can move because of solver differences,
depletion time-step choices, burnup-grid coarseness, predictor/corrector
settings, or other product-specific numerical options.

If a case requires very large :code:`nlib` or :code:`nburn` values, or if the
:code:`q1`/:code:`q2` history moves nonmonotonically as ORIGAMI is refined,
treat that as evidence that the high-order depletion-space reference may be
insufficiently refined before loosening the ORIGAMI convergence criteria.

Single-Zone Equality Limit
--------------------------

After the high-order calculation has converged in depletion space, the
low-order problem is defined by two inputs:

* the initial condition;
* the one-group cross sections represented in the ORIGEN library.

The initial condition should be exactly equal between the high-order reference
and the ORIGAMI reconstruction, within serialization precision. Inventory
through time should then converge toward equality when all of the following are
true:

* The high-order depletion-space solution is converged. The limit to which the
  high-order calculation is converged defines the best consistency target. For
  example, if high-order :sup:`235`U is converged to within 0.01% of the
  infinitely refined depletion-time-grid limit, then the best expected
  low-order agreement is also limited by that 0.01%.
* The representation of the cross sections used for interpolation is
  converged, smooth over the interpolation domain, and free of avoidable
  representation error.
* The low-order ORIGAMI time grid, including interpolation through the reactor
  library, is refined to convergence.
* Power and burnup normalization are consistently defined between the
  high-order and low-order problems.
* The high-order and low-order models represent a single depletion zone. If the
  high-order case evolves multiple fuel zones and ORIGAMI evolves one averaged
  zone, the remaining difference includes a real lumping approximation.
* The same depletion data are used: nuclear data, decay chain, fission yields,
  branching data, recoverable energy assumptions, and nuclide set.
* The high-order inventory and cross sections are extracted from the exact
  material or case intended for ORIGAMI, such as the :code:`FUEL` basis rather
  than a total-system or unrelated material basis.
* The comparison uses the same basis and units, including :code:`g/gIHM`, the
  same initial-heavy-metal normalization, and the same time points or a
  well-defined endpoint matching rule.

Polaris and TRITON Artifacts
----------------------------

OLM uses an explicit high-order artifact contract:

* :code:`TRITON` for :code:`=t-depl*`, :code:`=t5-depl*`, and
  :code:`=t6-depl*` inputs;
* :code:`Polaris` for :code:`=polaris*` inputs.

TRITON and Polaris do not expose identical output artifacts. OLM therefore
uses product-specific extraction rules rather than guessing from file suffixes.
For TRITON, OLM uses the fuel/source inventory case. For Polaris, OLM locates
the :code:`FUEL` material-class case and uses the :code:`FUEL` F33 archive.

Fuel Basis
----------

Low-order consistency is meaningful only when the high-order power basis and
the ORIGAMI fuel basis describe the same fuel inventory. The quick Polaris
examples use:

.. code-block:: text

   basis ALL=no FUEL=YES

This keeps the Polaris power normalization on fuel materials, which is the
same inventory class ORIGAMI receives. TRITON examples deplete and extract the
fuel mixture used by ORIGAMI. For TRITON, the :code:`read depletion` block
defines the burnup basis. The quick examples deplete only the fuel mixture used
for ORIGAMI, for example:

.. code-block:: text

   read depletion
     10
   end depletion

OLM tags TRITON arpdata burnups and builds the ORIGAMI cycle history from the
TRITON output library table because that table follows the basis mixtures
selected in the :code:`DEPLETION` block. OLM still uses the TRITON F71
fuel/source case for inventory extraction and initial heavy metal.

There are two ways to handle power consistency:

* Standard: make the high-order basis equal to fuel so the reported burnup is
  already fuel burnup. For Polaris, the quick examples use
  :code:`basis ALL=no FUEL=YES`; for TRITON, the quick examples deplete only
  the ORIGEN fuel mixture. The resulting basis matches the fuel-only ORIGAMI
  reconstruction.
* Legacy compatibility: provide system power in the high-order calculation,
  then back out the equivalent fuel power and fuel-basis burnup for the ORIGAMI
  fuel-only reconstruction. This path is more fragile and should not be used
  for new validation examples. For consistency, burnup reported on a system
  basis is not automatically the burnup that should be tagged on the fuel
  library; the fuel library burnup must be on the same fuel basis as the ORIGAMI
  calculation.

Assembly Averages
-----------------

Some high-order examples contain multiple fuel materials or poison-bearing
fuel pins. ORIGAMI quick checks use an assembly-average low-order model for
those cases. Assembly-average inputs are passed to templates under
:code:`assembly_average`.

For a single fuel mixture, the high-order and ORIGAMI results should converge
tightly as :code:`nlib` and :code:`nburn` are refined. For an assembly with
multiple fuel zones, remaining differences can reflect real homogenization and
spatial-spectrum effects, not only ORIGAMI time-step convergence.

Reference Quality
-----------------

For a sufficiently consistent and numerically refined one-zone pin-cell case,
OLM should be able to achieve approximately:

* :code:`q1 >= 0.94`
* :code:`q2 >= 0.99`

This target comes from a local UOX Polaris single-pin check using a
:code:`0 -> 1 GWd/MTIHM` step, a Polaris :code:`FUEL` material library generated
with 128 high-order burnup substeps, and an ORIGAMI reconstruction using
:code:`nlib=128`. Assembly cases, coarser burnup grids, or inconsistent
material bases are not expected to meet this target without additional
refinement or model changes.

Quick Examples
--------------

The quick examples intentionally use simplified transport/runtime settings so
they can be used for PR validation. Model templates label these settings with
:code:`QUICK VALIDATION SETTINGS` or :code:`quick-validation simplifications`.
Those settings should be removed or tightened for production reactor-library
calculations.

Fast, approximate transport settings are acceptable for these examples as long
as they are clearly labeled and deterministic. The depletion time grid is
different: because low-order consistency treats the high-order depletion-space
solution as the reference, the high-order burnup grid must be fine enough to
produce stable inventories and one-group cross sections. For Polaris examples,
this is the :code:`time.gwd_burnups` list in the OLM config, rendered as the
Polaris :code:`bu` block.

Refine the Polaris burnup grid when successive high-order calculations produce
material inventory changes larger than the intended comparison tolerance. Do
the same kind of check after changing depletion solver options or other
time-integration settings. Do not compensate for an unstable high-order
reference by simply increasing ORIGAMI :code:`nlib` or :code:`nburn`; that
makes ORIGAMI better at reconstructing an unstable reference, not a better
validation case.

The PR-validation example set covers:

* :code:`examples/polaris_uoxgd_quick`: BWR UOX+Gd2O3 assembly behavior;
* :code:`examples/polaris_uox_pin_quick`: Polaris UOX PWR single-pin behavior;
* :code:`examples/polaris_uoxgd_pin_quick`: Polaris BWR UOX+Gd2O3
  single-pin behavior;
* :code:`examples/polaris_uoxgdcr_pin_quick`: Polaris BWR UOX+Gd2O3+Cr2O3
  single-pin behavior;
* :code:`examples/polaris_mox_pin_quick`: Polaris MOX PWR single-pin behavior;
* :code:`examples/triton_uox_pin_quick`: TRITON UOX PWR single-pin behavior;
* :code:`examples/triton_mox_pin_quick`: TRITON MOX PWR single-pin behavior.

Validation Reporting
--------------------

The GitHub validation comment should show only the current :code:`g/gIHM`
summary table. Detailed artifacts, such as histograms and nuclide plots, may be
uploaded for review, but the PR table should remain a short summary by SCALE
version, product, case, example path, :code:`q1`, :code:`q2`, and pass/fail
status.
