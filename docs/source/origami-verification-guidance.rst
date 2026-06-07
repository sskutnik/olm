ORIGAMI Verification Guidance
=============================

Source: ``sgtech-v4.pptx``, "Status Update for SGTech Project:
User-friendly ORIGEN Reactor Library Generation", SGTech IAEA visit,
September 21, 2023.

This guidance captures the technical strategy for how OLM should explain,
check, and report ORIGAMI low-order results against high-order
transport/depletion results.

Core Model
----------

Treat ORIGAMI as a fast low-order reconstruction of a high-order depletion
solution.

* High-order reference: TRITON or Polaris coupled transport/depletion
  calculations.
* Low-order result: ORIGAMI/ORIGEN inventory calculations using interpolated
  one-group cross section data from an ORIGEN reactor library.
* OLM role: generate the high-order state grid, assemble the interpolatable
  reactor library, run checks that compare low-order reconstructions to
  high-order reference cases, and report the comparison clearly.

Do not present ORIGAMI low-order results as independently equivalent to TRITON
or Polaris. The defensible claim is that a generated ORIGEN reactor library
should allow ORIGAMI to reproduce the relevant high-order inventory solution
within understood error sources.

Why The Low-Order Strategy Exists
---------------------------------

OLM moves expensive coupled transport/depletion work up front. The user should
be able to run many later inventory calculations quickly by interpolating
reactor-library data instead of rerunning transport. This is appropriate when
the generated library spans the relevant operating and design envelope, such as
burnup, moderator density, enrichment, Pu content, Pu vector, or other state
variables that materially affect the spectrum.

Reports and user-facing messages should:

* explain the high-order calculations used to build the library;
* explain the interpolation axes and their intended applicability;
* show that ORIGAMI can reproduce withheld or representative high-order
  solutions;
* warn when user input appears outside the library's intended applicability.

Verification Scope
------------------

Verification should answer this question: can ORIGAMI, using the generated
library, reproduce the high-order solution that the library is intended to
approximate?

Verification belongs in the :code:`check` stage of :code:`olm create`. It
should produce machine-readable summaries and human-readable report sections
with figures where possible.

Verification should compare nuclides important for the use case, not only
aggregate norms. At minimum, report actinides and fission products separately
when they behave differently.

Low-Order Error Sources
-----------------------

Keep these error sources separate in design, tests, reports, and user-facing
explanations.

Numerical Solver Difference
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The high-order calculation and ORIGAMI can differ because their depletion
solvers, burnup step sizes, and internal substep choices differ.

Guidance:

* Ensure the high-order reference calculation is numerically converged enough
  in depletion space for the intended comparison. If the high-order calculation
  is not sufficiently converged in depletion space, there is no hope for a
  meaningful high-order/low-order consistency result. The transport
  approximation can be coarse or physically wrong as long as it is deterministic
  and produces a stable depletion-space reference.
* Treat high-order depletion-space convergence as convergence of both the
  number densities and the one-group cross sections represented in the ORIGEN
  library. Because cross sections usually vary more slowly than number
  densities, converged number densities are a strong practical indicator that
  the cross-section representation is also converged enough for the consistency
  check.
* High-order inventory or cross-section instability can come from solver
  differences, time-stepping differences, burnup-grid coarseness,
  predictor/corrector settings, or other product-specific numerical options.
* Ensure the ORIGAMI calculation is numerically converged enough for the
  intended comparison.
* Treat transport simplification and depletion time-grid refinement as separate
  choices. Quick validation cases may use fast, approximate transport settings,
  but the high-order depletion time grid must still be refined enough that the
  high-order depletion-space solution is a stable reference.
* Do not require bitwise or step-for-step equivalence between the high-order
  code and ORIGAMI.
* If a discrepancy is near 1% for sensitive nuclides, investigate solver
  options and time-step defaults before blaming library interpolation.

The SGTech presentation observed solver differences around 0.1% to 1%,
depending on nuclide, for default ORIGAMI-style verification problems.

Single-Zone Consistency Limit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a single depletion zone, a consistent low-order calculation is possible
only after the high-order depletion-space reference has converged. At that
point, the ORIGAMI problem is defined by the initial condition and the
one-group cross sections represented in the ORIGEN library.

The initial condition should match exactly, within serialization precision.
The inventory through time should converge toward the high-order result when
these conditions hold:

* The high-order depletion-space solution is converged. The high-order
  convergence limit sets the best possible low-order consistency target. If the
  high-order reference is 0.01% from its infinitely refined depletion-time-grid
  limit for a nuclide, the low-order result cannot be expected to agree better
  than that high-order reference quality.
* The cross-section representation used for interpolation is converged, smooth
  over the interpolation domain, and free of avoidable representation error.
* The low-order ORIGAMI time grid and reactor-library interpolation path are
  refined to convergence.
* Power and burnup normalization are consistently defined. The standard method
  is to make the high-order basis equal to fuel so the reported burnup is
  already fuel burnup and matches the fuel-only ORIGAMI reconstruction. The
  legacy compatibility method is to provide system power in the high-order
  calculation and explicitly back out the equivalent fuel power and fuel-basis
  burnup for ORIGAMI. That path is more fragile and should not be used for new
  validation examples. TRITON-reported system burnup is not automatically the
  burnup that should be tagged on a fuel-only ORIGEN library.
* The high-order and low-order cases represent one depletion zone. If the
  high-order model contains multiple zones and ORIGAMI uses one averaged zone,
  the comparison includes nonlinear lumping error.
* The high-order and low-order calculations use the same depletion data:
  nuclear data, decay chain, fission yields, branching data, recoverable energy
  assumptions, and nuclide set.
* The high-order inventory and cross sections are extracted from the intended
  material or case, such as the :code:`FUEL` basis, not a total-system or
  unrelated material basis.
* The comparison uses a consistent basis and units, including :code:`g/gIHM`,
  initial-heavy-metal normalization, and identical time points or an explicit
  endpoint matching rule.

Interpolation Error
~~~~~~~~~~~~~~~~~~~

ORIGAMI uses interpolation in one-group cross section space as a substitute for
additional high-order transport calculations.

Guidance:

* Interpolation error should decrease with grid refinement when the cross
  sections vary smoothly over the chosen axes.
* The high-order grid must be sufficiently refined for the interpolation
  method.
* For Polaris examples, refine :code:`time.gwd_burnups` until successive
  high-order inventory results and cross-section representations are stable
  enough for the intended high-order/low-order comparison, then revisit solver
  and time-integration options if the reference still moves too much.
* The ORIGAMI burnup/time grid must be sufficiently refined for the intended
  inventory comparison.
* Reports should identify the interpolation axes and distinguish interpolation
  error from solver and lumping effects.

The SGTech presentation observed interpolation error below 1% in the discussed
verification studies.

Nonlinear Lumping Error
~~~~~~~~~~~~~~~~~~~~~~~

Depletion and decay are nonlinear operations. Evolving separate fuel regions in
separate spectra and then averaging is not generally equal to evolving one
averaged composition in one averaged spectrum.

High-order models can evolve pins or regions separately and then sum to an
assembly average. Low-order ORIGAMI may use assembly-average cross section data
to regenerate an average result.

Guidance:

* Identify when a verification problem includes lumping effects.
* Isolate lumping error from interpolation and solver error where possible.
* Explain that lumping is a model-form approximation, not a numerical accident.
* Use one-zone pin-cell tests when the goal is to remove lumping and isolate
  other error sources.

The SGTech presentation observed lumping error below 0.5% in the discussed
default verification studies.

Isolating Error Sources
-----------------------

Design checks so each error source can be isolated where possible:

* To isolate solver difference, compare at state points where interpolation and
  lumping are removed or minimized.
* To remove interpolation error, use a state point that exists exactly in the
  high-order grid.
* To remove lumping error, use a one-zone pin-cell problem or another case with
  no spatial lumping.
* To study combined effects, label the combination explicitly, such as solver
  plus lumping or solver plus interpolation.

Reports should avoid a single unexplained "OLM error" number. Prefer a
decomposition that tells the user what kind of approximation is being tested.

Reference Achievable Quality Target
-----------------------------------

For a sufficiently consistent and numerically refined one-zone pin-cell case,
OLM should be able to achieve approximately:

* :code:`q1 >= 0.94`
* :code:`q2 >= 0.99`

Reference local result from June 7, 2026:

* Case: UOX single-pin Polaris model, SCALE 6.3.3,
  :code:`0 -> 1 GWd/MTIHM`
* High-order reference: Polaris :code:`n=128` burnup substeps using the
  :code:`FUEL` material library
* Low-order reconstruction: ORIGAMI using the same Polaris :code:`FUEL`
  library, :code:`nlib=128`
* Metric: :code:`grams_per_initial_hm` (:code:`g/gIHM`)
* Thresholds: :code:`epsr=1e-3`, :code:`epsa=1e-6`
* Compared points: :code:`129`
* Achieved: :code:`q1=0.94474604`, :code:`q2=0.99447460`

Use this as a practical target for cases where interpolation, spatial lumping,
and time-grid inconsistencies have been minimized. Assembly cases with lumping,
coarser burnup grids, or inconsistent material definitions are not expected to
meet this target without additional refinement or model changes.

Applicability And User Input Consistency
----------------------------------------

OLM should help users avoid applying a library outside the conditions it
represents. The SGTech presentation identified a missing or incomplete check:
ORIGAMI did not yet fully check consistency between the user's initial
composition and the library's applicability.

Guidance:

* Add checks or warnings when user input is outside the generated library
  domain.
* Distinguish spectrum-relevant composition changes from changes that mostly
  affect inventory but not the spectrum.
* Do not reject harmless isotopic differences simply because exact
  compositions differ.
* Prefer clear warnings with the relevant axis, expected range, and provided
  value.

Example principle: changing Am241 content may not invalidate cross sections if
that isotope does not materially affect the spectrum. The check should be
physics-aware, not just exact-string or exact-vector matching.

Validation Scope
----------------

Validation answers a different question from verification: does the overall
workflow produce credible results against measurement data for real cases?

Validation can include SFCOMPO spent fuel assay cases such as Beznau-1 M308 K7
BM5 and MALIBU-style MOX cases. Treat measured inventories, burnup monitors,
power history, cooling time, and operating history as part of the validation
problem.

Guidance:

* Separate code-to-code verification from measurement validation.
* Preserve measured uncertainty bands where available.
* Report calculated-to-experimental differences by nuclide.
* Interpret burnup monitors carefully because a burnup mismatch affects many
  nuclides.
* Clearly identify whether power or burnup calibration was applied.

Power History And Burnup Calibration
------------------------------------

When validation data lacks detailed uncertainty for sample-specific power
history, power history uncertainty can dominate the interpretation.

Guidance:

* Compare results before and after burnup or FIMA calibration when a measured
  burnup monitor is available.
* State the calibration factor if one is used.
* Do not hide the uncalibrated result; it is diagnostic.
* For uncertainty studies, sample cycle power multipliers and show the impact
  on key nuclides.
* Explain skewed burnup distributions if the perturbation strategy changes the
  effective cycle burnup.

Reporting Expectations
----------------------

OLM reports should make the verification story clear to both developers and
analysts:

* State the high-order reference case.
* State the low-order ORIGAMI reconstruction case.
* List the reactor library and interpolation axes.
* Show nuclide-by-nuclide comparisons for important actinides and fission
  products.
* Separate verification checks from validation comparisons.
* Identify which error sources are present in each figure or table.
* Include warnings when input applicability is questionable.
* Preserve enough metadata to reproduce the comparison.

Development Guidance
--------------------

When changing OLM checking, reporting, or ORIGAMI integration:

* Keep the high-order/low-order distinction explicit in names, docs, and plots.
* Prefer small focused checks that isolate solver, interpolation, and lumping
  behavior.
* Avoid silently combining unrelated discrepancies into one score.
* Add tests using existing realistic data when available.
* Use property-based tests for mathematical interpolation behavior when
  practical.
* Keep report output deterministic so changes can be reviewed.
* Treat warnings as part of the user contract: they should be concrete,
  actionable, and reproducible.

Terminology
-----------

High-order or HI
    TRITON or Polaris coupled transport/depletion reference calculation.

Low-order or LO
    ORIGAMI/ORIGEN reconstruction using an interpolated reactor library.

Reactor library
    ORIGEN library containing one-group cross section data over a generated
    state space.

Verification
    Low-order reconstruction compared to high-order reference.

Validation
    Calculated results compared to measurement data.

Lumping
    Error from replacing separate nonlinear depletion histories with an
    averaged representation.
