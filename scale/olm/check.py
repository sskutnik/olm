"""
Module for checking classes.
"""
__all__ = ["sequencer", "GridGradient", "LowOrderConsistency"]

import numpy as np
from tqdm import tqdm, tqdm_notebook
import scale.olm.core as core
import scale.olm.run as run
import json
from pathlib import Path
import copy
import os
import scale.olm.internal as internal
from typing import List, Union, Dict, Literal


class CheckInfo:
    def __init__(self):
        self.test_pass = False


Model = Dict[str, any]
Env = Dict[str, any]

# -----------------------------------------------------------------------------------------

_TYPE_SEQUENCER = "scale.olm.check:sequencer"


def _schema_sequencer(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_SEQUENCER, with_state=with_state)
    return _schema


def _test_args_sequencer(with_state: bool = False):
    return {
        "_type": _TYPE_SEQUENCER,
        "sequence": [
            {"eps0": 0.0001, "_type": "scale.olm.check:GridGradient"},
            {
                "_type": "scale.olm.check:LowOrderConsistency",
                "name": "loc",
                "template": "model/origami/system-uox.jt.inp",
                "target_q1": 0.70,
                "target_q2": 0.95,
                "eps0": 1e-12,
                "epsa": 1e-6,
                "epsr": 1e-3,
                "nuclide_compare": ["0092235", "0094239"],
            },
        ],
    }


def sequencer(
    sequence: List[dict],
    _model: Model,
    _env: Env,
    dry_run: bool = False,
    _type: Literal[_TYPE_SEQUENCER] = None,
):
    """Run a sequence of checks.

    Args:
        sequence: List of checks to run by name.

        _model: Reference model data

        _env: Environment data.

    """
    output = []
    if dry_run:
        return {"test_pass": False, "output": output}

    test_pass = True
    try:
        # Process all the input.
        run_list = []
        i = 0
        for s in sequence:
            # Set the full name.
            t = s["_type"]
            if t.find(":") == -1:
                t = "scale.olm.check:" + t
            s["_type"] = t

            internal.logger.info("Checking options for", type=t, index=i)
            i += 1

            # Initialize the class.
            this_class = internal._fn_redirect(**s, _env=_env, _model=_model)
            run_list.append(this_class)

        # Read the reactor_library.
        work_dir = Path(_env["work_dir"])
        arpdata_txt = work_dir / "arpdata.txt"
        name = _model["name"]
        if arpdata_txt.exists():
            reactor_library = core.ReactorLibrary(arpdata_txt, name)
        else:
            reactor_library = core.ReactorLibrary(Path(f"{name}.arc.h5"))

        # Execute in sequence.
        i = 0
        for r in run_list:
            internal.logger.info("Running checking sequence={}".format(i))

            info = r.run(reactor_library)
            output.append(info.__dict__)
            i += 1

            if not info.test_pass:
                test_pass = False

        internal.logger.info(
            "Finished without exception test_pass={}".format(test_pass)
        )

    except ValueError as ve:
        internal.logger.error(str(ve))

    return {"test_pass": test_pass, "sequence": output}


# -----------------------------------------------------------------------------------------

_TYPE_GRIDGRADIENT = "scale.olm.check:GridGradient"


def _schema_GridGradient(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_GRIDGRADIENT, with_state=with_state)
    return _schema


def _test_args_GridGradient(with_state: bool = False):
    args = {"_type": _TYPE_GRIDGRADIENT}
    args.update(GridGradient.default_params())
    return args


class GridGradient:
    """Compute the grid gradients

    Computes the absolute and relative gradients of the reaction coefficient data
    in each dimension at each point and collects them into a data structure.

    The fraction of relative gradients which fall below the specified limit :code:`epsr`
    is the first quality score, :code:`q1=1-fr` where :code:`fr` is the failed fraction.
    The test passes quality check 1 if the :code:`q1<=target_q1`.

    Most often, we care less about relative differences when the absolute values are
    very small, e.g. a 10% difference in a 1e-12 barn cross section is not as big
    a deal as a 1% difference in a 100 barn cross section. Quality score :code:`q2`
    takes this into account by considering the fraction of points which fail the
    pure relative test, :code:`q1`, and those that fail a combined test where the
    relative gradient must exceed :code:`epsr` and the absolute gradient must exceed
    :code:`epsa`. The failed fraction is :code:`fa` and the combined score for
    :code:`q2=1-0.9*fa-0.1*fr`. In this way, one cannot get a perfect 1.0 for either
    score if there are any failures in a relative sense, but the second score penalizes
    them less. The second test passes if :code:`q2<=target_q2`.

    Args:
        eprs: The limit for the relative gradient.
        epsa: The limit for the absolute gradient.
        target_q1: The target for the q1 (relative only) score.
        target_g2: The target for the q2 (weighted relative and absolute) score.
        eps0: The minimum gradient to care about.

    """

    @staticmethod
    def describe_params():
        return {
            "eps0": "minimum value",
            "epsa": "absolute epsilon",
            "epsr": "relative epsilon",
            "target_q1": "target for quality score 1",
            "target_q2": "target for quality score 2",
        }

    @staticmethod
    def default_params():
        c = GridGradient()
        return {
            "eps0": c.eps0,
            "epsa": c.epsa,
            "epsr": c.epsr,
            "target_q2": c.target_q2,
            "target_q1": c.target_q1,
        }

    def __init__(
        self,
        _model: dict = None,
        _env: dict = {},
        eps0: float = 1e-20,
        epsa: float = 1e-1,
        epsr: float = 1e-1,
        target_q1: float = 0.5,
        target_q2: float = 0.7,
        _type: Literal[_TYPE_GRIDGRADIENT] = None,
    ):
        self.eps0 = eps0
        self.epsa = epsa
        self.epsr = epsr
        self.target_q1 = target_q1
        self.target_q2 = target_q2
        self.nprocs = _env.get("nprocs", 3)

    def run(self, reactor_library):
        """Run the calculation and return post-processed results"""

        internal.logger.info(
            "Running "
            + self.__class__.__name__
            + " check with params={}".format(json.dumps(self.__dict__))
        )
        self.__calc(reactor_library)

        # After calc the self.ahist, rhist, khist, and rel_axes variables are ready to
        # compute metrics.
        info = self.info()
        internal.logger.info(
            "Completed "
            + self.__class__.__name__
            + " with q1={:.2f} and q2={:.2f}".format(info.q1, info.q2)
        )

        return info

    def info(self):
        """Recalculate and return the score information."""
        info = CheckInfo()
        info.name = self.__class__.__name__
        info.eps0 = self.eps0
        info.epsa = self.epsa
        info.epsr = self.epsr
        info.target_q1 = self.target_q1
        info.target_q2 = self.target_q2
        info.wa = int(
            np.logical_and((self.ahist > self.epsa), (self.rhist > self.epsr)).sum()
        )
        info.wr = int((self.rhist > self.epsr).sum())
        info.m = int(len(self.ahist))
        info.fr = float(info.wr) / info.m
        info.q1 = 1.0 - info.fr
        info.fa = float(info.wa) / info.m
        info.q2 = 1.0 - 0.9 * info.fa - 0.1 * info.fr

        info.test_pass_q1 = info.q1 >= info.target_q1
        info.test_pass_q2 = info.q2 >= info.target_q2
        info.test_pass = info.test_pass_q1 and info.test_pass_q2

        return info

    def __calc(self, reactor_library):
        """Drives the set up for the kernel with reactor_library as input"""

        self.rel_axes = list()
        for x_list in reactor_library.axes_values:
            dx = x_list[-1] - x_list[0]
            x0 = x_list[0]
            z = list()
            for x in x_list:
                z.append((x - x0) / dx)
            self.rel_axes.append(z)
        internal.logger.info("Finished computing relative values on axes")

        self.yreshape = np.moveaxis(reactor_library.coeff, [-1], [0])
        internal.logger.info("Finished reshaping coefficients")

        internal.logger.info("Computing grid gradients ...")
        self.ahist, self.rhist, self.khist = GridGradient.__kernel(
            self.rel_axes, self.yreshape, self.eps0
        )
        internal.logger.info("Finished computing grid gradients")

    @staticmethod
    def __kernel(rel_axes, yreshape, eps0):
        """Lowest level kernel for the calculation"""
        # Number of dimensions.
        n = len(rel_axes)

        # Number of coefficients.
        ncoeff = np.shape(yreshape)[0]

        # Initialize histogram variables.
        nd = np.sum([len(a) - 1 for a in rel_axes])
        rhist = np.zeros(n * nd * ncoeff)
        ahist = np.zeros(n * nd * ncoeff)
        khist = np.zeros(n * nd * ncoeff)

        # For each coefficient in the transition matrix.
        for k in tqdm(range(ncoeff)):
            # Get just the grid of values for this coefficient.
            y = yreshape[k, ...]

            # Compute the min/max magnitude at any point in the grid.
            max_y = np.amax(np.absolute(y))
            if max_y <= 0:
                max_y = eps0
            min_y = np.amin(np.absolute(y))
            if min_y <= 0:
                min_y = eps0
            mid_y = 0.5 * (min_y + max_y)

            for i in range(n):
                # First and second derivatives.
                yp = np.asarray(np.gradient(y, rel_axes[i], axis=i))
                ypp = np.asarray(np.gradient(yp, rel_axes[i], axis=i))

                # Move the axis `i` to the last dimension so we can use ... below.
                # Go ahead and take absolute value to simplify too.
                ypp_abs = np.absolute(np.moveaxis(ypp, i, -1))

                # Evaluate Rolle's theorem for each interval along axis=i
                # March through each interval explicitly
                dx_list = np.diff(rel_axes[i])
                for j in range(len(dx_list)):
                    dx = dx_list[j]

                    # Take max of left and right ypp for this interval.
                    max_ypp = max(
                        np.amax(ypp_abs[..., j]), np.amax(ypp_abs[..., j + 1])
                    )

                    # Compute error for this interval
                    error = (dx**2) * max_ypp / 8.0

                    # Flat index.
                    iu = k * n * nd + i * nd + j

                    # Update absolute and relative versions.
                    ahist[iu] = error
                    rhist[iu] = error / (mid_y * mid_y)

                    # Remember the coefficient index of this particular gradient difference.
                    khist[iu] = k

        return ahist, rhist, khist


# -----------------------------------------------------------------------------------------

_TYPE_LOWORDERCONSISTENCY = "scale.olm.check:LowOrderConsistency"


def _schema_LowOrderConsistency(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_LOWORDERCONSISTENCY, with_state=with_state)
    return _schema


def _test_args_LowOrderConsistency(with_state: bool = False):
    args = {"_type": _TYPE_LOWORDERCONSISTENCY}
    args.update(LowOrderConsistency.default_params())
    return args


class LowOrderConsistency:
    """Check that we are consistent with the original calculation.

    The ORIGEN library approach can be viewed as a high-order/low-order methodology
    where the ORIGEN library interpolation represents a low-order method which
    should agree with the high-order method.

    This check assumes that we already have high-order (e.g. TRITON) nuclide
    inventory results available. We use each of the libraries in the interpolation
    space in a new low-order (ORIGAMI) calculation. Consistent inputs are automatically
    constructed from available data. We then compare all nuclide inventory differences
    in the same way as for the :obj:`GridGradient` method, instead of relative and
    absolute gradients, we have relative and absolute differences in nuclide inventory.

    A number of plots are produced as side effects, referenced in the dictionary
    returned from the run() method.

    Args:
        name: Name of the test.
        template: Template file to use for the low-order calculation.
        metric: Primary inventory metric to use for quality scores.
        nuclide_compare: List of nuclide identifiers for the detailed error plots.
        eprs: The limit for the relative gradient.
        epsa: The limit for the absolute gradient.
        target_q1: The target for the q1 (relative only) score.
        target_g2: The target for the q2 (weighted relative and absolute) score.
        eps0: The minimum gradient to care about.

    """

    @staticmethod
    def describe_params():
        return {
            "metric": "primary inventory metric",
            "eps0": "minimum value",
            "epsa": "absolute epsilon",
            "epsr": "relative epsilon",
            "target_q1": "target for quality score 1",
            "target_q2": "target for quality score 2",
            "nlib_start": "initial ORIGAMI nlib value",
            "nlib_max": "maximum ORIGAMI nlib value",
            "nburn_start": "initial ORIGAMI nburn value",
            "nburn_max": "maximum ORIGAMI nburn value",
            "q1_stop_criteria": "stop when q1 changes by no more than this",
            "q2_stop_criteria": "stop when q2 changes by no more than this",
            "nuclide_compare": "plot me",
            "assembly_average": "assembly-average data for the low-order template",
            "template": "template file name",
            "name": "name for test",
        }

    @staticmethod
    def default_params():
        import inspect

        # Use inspect to get required arguments.
        defaults = {}
        fn = internal._get_function_handle(_TYPE_LOWORDERCONSISTENCY)
        for k, v in inspect.signature(fn).parameters.items():
            if k.startswith("_"):
                continue
            defaults[k] = v.default
        return defaults

    def __init__(
        self,
        name: str = "",
        template: str = "",
        metric: Literal[
            "grams_per_initial_hm", "atom_fraction"
        ] = "grams_per_initial_hm",
        eps0: float = 1e-12,
        epsa: float = 1e-6,
        epsr: float = 1e-3,
        target_q1: float = 0.9,
        target_q2: float = 0.95,
        nlib_start: int = 1,
        nlib_max: int = 1,
        nburn_start: int = 1,
        nburn_max: int = 1,
        q1_stop_criteria: float = 0.0,
        q2_stop_criteria: float = 0.0,
        nuclide_compare: List[str] = ["u235", "pu239"],
        assembly_average: Dict = None,
        _model: Model = None,
        _env: Env = None,
        _type: Literal[_TYPE_LOWORDERCONSISTENCY] = None,
        _dry_run: bool = False,
    ):
        self._env = _env
        self._model = _model
        self.name = name
        self.nuclide_compare = nuclide_compare
        self.metric = metric
        self.eps0 = eps0
        self.epsa = epsa
        self.epsr = epsr
        self.target_q1 = target_q1
        self.target_q2 = target_q2
        if nlib_start <= 0:
            raise ValueError("LowOrderConsistency nlib_start must be positive.")
        if nlib_max < nlib_start:
            raise ValueError(
                "LowOrderConsistency nlib_max must be greater than or equal to "
                "nlib_start."
            )
        if nburn_start <= 0:
            raise ValueError("LowOrderConsistency nburn_start must be positive.")
        if nburn_max < nburn_start:
            raise ValueError(
                "LowOrderConsistency nburn_max must be greater than or equal to "
                "nburn_start."
            )
        if q1_stop_criteria < 0.0 or q2_stop_criteria < 0.0:
            raise ValueError(
                "LowOrderConsistency q1_stop_criteria and q2_stop_criteria "
                "must be nonnegative."
            )
        self.nlib_start = nlib_start
        self.nlib_max = nlib_max
        self.nburn_start = nburn_start
        self.nburn_max = nburn_max
        self.q1_stop_criteria = q1_stop_criteria
        self.q2_stop_criteria = q2_stop_criteria
        self.assembly_average = (
            {} if assembly_average is None else copy.deepcopy(assembly_average)
        )
        if self.metric not in ["grams_per_initial_hm", "atom_fraction"]:
            raise ValueError(
                "Unsupported LowOrderConsistency metric="
                f"{self.metric!r}; expected 'grams_per_initial_hm' or 'atom_fraction'."
            )

        if _dry_run:
            return

        if _env == None:
            dir = Path.cwd()
        else:
            dir = Path(_env["config_file"]).parent

        tm = core.TemplateManager([dir])

        self.template_path = tm.path(template)
        self.template_paths = tm.paths
        internal.logger.info(
            "check " + __class__.__name__, template_file=self.template_path
        )

        self.work_path = Path(_env["work_dir"])
        self.base_check_path = self.work_path / "check" / name
        self.check_path = self.base_check_path
        self.check_dir = self.check_path.relative_to(self.work_path)

    @staticmethod
    def make_diff_plot(
        identifier,
        image,
        time,
        min_diff,
        max_diff,
        max_diff0,
        perms,
        ylabel="(lo-hi)/max(|hi|) (%)",
    ):
        """Make the difference plot."""
        import matplotlib.pyplot as plt

        plt.rcParams.update({"font.size": 18})
        plt.figure()
        color = core.NuclideInventory._nuclide_color(identifier)
        plt.fill_between(
            np.asarray(time) / 86400.0,
            100 * np.asarray(min_diff),
            100 * np.asarray(max_diff),
            alpha=0.3,
            color=color,
        )

        for perm in perms:
            plt.plot(
                np.asarray(time) / 86400.0,
                100 * np.asarray(perm["(lo-hi)/max(|hi|)"]),
                "k-",
                alpha=0.4,
            )

        plt.xlabel("time (days)")
        plt.ylabel(ylabel)
        plt.legend(["{} (max error: {:.2f} %)".format(identifier, 100 * max_diff0)])
        plt.savefig(image, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _nuclide_mass_vector(names, nuclide_data):
        return np.array([nuclide_data[name]["mass"] for name in names], dtype=float)

    @staticmethod
    def _metric_units(metric):
        if metric == "grams_per_initial_hm":
            return "g/gIHM"
        if metric == "atom_fraction":
            return "atom fraction"
        return None

    @staticmethod
    def _amounts_to_grams_per_initial_hm(amounts, masses, initialhm):
        amounts = np.asarray(amounts, dtype=float)
        masses = np.asarray(masses, dtype=float)
        initialhm = np.asarray(initialhm, dtype=float)
        if np.any(initialhm <= 0.0):
            raise ValueError(
                "LowOrderConsistency requires positive initial heavy metal values "
                "to calculate g/gIHM."
            )
        initialhm_grams = initialhm * 1.0e6
        return amounts * masses[None, None, :] / initialhm_grams[:, None, None]

    @staticmethod
    def _amounts_to_atom_fraction(amounts):
        amounts = np.asarray(amounts, dtype=float)
        totals = amounts.sum(axis=2)
        if np.any(totals == 0.0):
            raise ValueError("Cannot calculate atom fractions with zero total atoms.")
        return amounts / totals[:, :, None]

    @staticmethod
    def _difference_arrays(lo, hi, eps0):
        ahist = np.absolute(lo - hi)
        rhist = np.absolute((lo + eps0) / (hi + eps0) - 1.0)
        return ahist, rhist

    def _quality_summary(self, ahist, rhist):
        ahist = np.ndarray.flatten(ahist)
        rhist = np.ndarray.flatten(rhist)
        wa = int(np.logical_and((ahist > self.epsa), (rhist > self.epsr)).sum())
        wr = int((rhist > self.epsr).sum())
        m = int(len(ahist))
        fr = float(wr) / m
        fa = float(wa) / m
        q1 = 1.0 - fr
        q2 = 1.0 - 0.9 * fa - 0.1 * fr
        return {
            "wa": wa,
            "wr": wr,
            "m": m,
            "fr": fr,
            "q1": q1,
            "target_q1": self.target_q1,
            "fa": fa,
            "q2": q2,
            "target_q2": self.target_q2,
            "test_pass_q1": q1 >= self.target_q1,
            "test_pass_q2": q2 >= self.target_q2,
            "test_pass": q1 >= self.target_q1 and q2 >= self.target_q2,
            "mean_abs_diff": float(np.mean(ahist)),
            "mean_rel_diff": float(np.mean(rhist)),
            "std_abs_diff": float(np.std(ahist)),
            "std_rel_diff": float(np.std(rhist)),
        }

    def _time_zero_quality_summary(self, ahist, rhist):
        return self._quality_summary(ahist[:, 0:1, :], rhist[:, 0:1, :])

    @staticmethod
    def _time_zero_scores_pass(summary):
        return summary["q1"] == 1.0 and summary["q2"] == 1.0

    def _require_time_zero_consistency(self, summary):
        summary["test_pass_time0"] = self._time_zero_scores_pass(summary["time0"])
        summary["test_pass"] = summary["test_pass"] and summary["test_pass_time0"]

    def _empty_nuclide_compare(self, ntime):
        import sys

        comparison = dict()
        for nuclide in self.nuclide_compare:
            eam = self.composition_manager.eam(nuclide)
            izzzaaa = self.composition_manager.izzzaaa(nuclide)
            i = self.names.index(izzzaaa)
            internal.logger.info(
                f"Found nuclide={nuclide} at index {i} for detailed comparison"
            )
            comparison[eam] = {
                "nuclide_index": i,
                "nuclide": eam,
                "nuclide_izzzaaa": izzzaaa,
                "time": self.time_list,
                "max_diff": [-sys.float_info.max] * ntime,
                "min_diff": [sys.float_info.max] * ntime,
                "perms": [],
                "image": "",
            }
        return comparison

    def _populate_nuclide_compare(self, comparison, lo, hi, image_dir):
        image_dir.mkdir(parents=True, exist_ok=True)
        for n in comparison:
            i_nuclide = comparison[n]["nuclide_index"]
            for k in range(len(lo)):
                lo_nuclide = lo[k, :, i_nuclide]
                hi_nuclide = hi[k, :, i_nuclide]
                err = (lo_nuclide - hi_nuclide) / (
                    self.eps0 + np.amax(np.absolute(hi_nuclide))
                )
                comparison[n]["perms"].append(
                    {
                        "hi_ii_json": str(
                            self.ii_json_list[k][0].relative_to(self.work_path)
                        ),
                        "lo_ii_json": str(
                            self.ii_json_list[k][1].relative_to(self.work_path)
                        ),
                        "point_index": k,
                        "lo": list(lo_nuclide),
                        "hi": list(hi_nuclide),
                        "(lo-hi)/max(|hi|)": list(err),
                    }
                )

        for n, d in comparison.items():
            for k in range(len(lo)):
                err = d["perms"][k]["(lo-hi)/max(|hi|)"]
                for j in range(len(self.time_list)):
                    d["max_diff"][j] = np.amax([err[j], d["max_diff"][j]])
                    d["min_diff"][j] = np.amin([err[j], d["min_diff"][j]])

            d["max_diff0"] = np.amax(
                [np.absolute(d["max_diff"]), np.absolute(d["min_diff"])]
            )
            image = image_dir / (n + "-diff.png")
            internal.logger.info(
                "creating nuclide diff", image=str(image.relative_to(self.work_path))
            )
            comparison[n]["image"] = str(image)

            label = core.NuclideInventory._nice_label0(self.composition_manager, n)
            LowOrderConsistency.make_diff_plot(
                label,
                image,
                d["time"],
                d["min_diff"],
                d["max_diff"],
                d["max_diff0"],
                d["perms"],
            )

        return comparison

    def _plot_metric_histogram(self, ahist, rhist, hist_image, ylabel):
        hist_image.parent.mkdir(parents=True, exist_ok=True)
        plot_data = CheckInfo()
        plot_data.ahist = np.ndarray.flatten(ahist)
        plot_data.rhist = np.ndarray.flatten(rhist)
        internal.logger.info(
            "creating histogram ", image=str(hist_image.relative_to(self.work_path))
        )
        core.RelAbsHistogram.plot_hist(
            plot_data,
            hist_image,
            xlabel=r"$\log_{10} |lo/hi-1|$",
            ylabel=ylabel,
        )
        return str(hist_image)

    def info(self):
        """Recalculate test statistics."""

        # set number of permutations, timesteps, and nuclides for error array
        info = CheckInfo()
        info.name = self.__class__.__name__

        info.eps0 = self.eps0
        info.epsa = self.epsa
        info.epsr = self.epsr
        info.target_q1 = self.target_q1
        info.target_q2 = self.target_q2
        info.metric = self.metric
        info.units = self._metric_units(self.metric)
        if not self.run_success:
            info.test_pass = False
            return info

        # Create a base comparison data structure to repeat for every permutation.
        internal.logger.info("Setting up detailed comparison structures...")
        ntime = len(self.time_list)

        hi_amount = np.asarray(self.hi_list, dtype=float)
        lo_amount = np.asarray(self.lo_list, dtype=float)

        atom_hi = self._amounts_to_atom_fraction(hi_amount)
        atom_lo = self._amounts_to_atom_fraction(lo_amount)
        atom_ahist, atom_rhist = self._difference_arrays(atom_lo, atom_hi, self.eps0)
        atom_hist_image = self.check_path / "atom_fraction" / "hist.png"
        atom_summary = self._quality_summary(atom_ahist, atom_rhist)
        atom_summary["time0"] = self._time_zero_quality_summary(atom_ahist, atom_rhist)
        self._require_time_zero_consistency(atom_summary)
        atom_summary["metric"] = "atom_fraction"
        atom_summary["units"] = self._metric_units("atom_fraction")
        atom_summary["hist_image"] = self._plot_metric_histogram(
            atom_ahist,
            atom_rhist,
            atom_hist_image,
            ylabel=r"$\log_{10} |hi-lo|$",
        )
        atom_summary["nuclide_compare"] = self._populate_nuclide_compare(
            self._empty_nuclide_compare(ntime),
            atom_lo,
            atom_hi,
            self.check_path / "atom_fraction",
        )

        masses = self._nuclide_mass_vector(self.names, self.nuclide_data)
        mass_hi = self._amounts_to_grams_per_initial_hm(
            hi_amount, masses, self.initialhm_list
        )
        mass_lo = self._amounts_to_grams_per_initial_hm(
            lo_amount, masses, self.initialhm_list
        )
        mass_ahist, mass_rhist = self._difference_arrays(mass_lo, mass_hi, self.eps0)
        mass_summary = self._quality_summary(mass_ahist, mass_rhist)
        mass_summary["time0"] = self._time_zero_quality_summary(mass_ahist, mass_rhist)
        self._require_time_zero_consistency(mass_summary)
        mass_summary["metric"] = "grams_per_initial_hm"
        mass_summary["units"] = self._metric_units("grams_per_initial_hm")
        mass_summary["nuclide_compare"] = self._populate_nuclide_compare(
            self._empty_nuclide_compare(ntime),
            mass_lo,
            mass_hi,
            self.check_path,
        )

        hist_image = self.check_path / "hist.png"
        info.hist_image = self._plot_metric_histogram(
            mass_ahist,
            mass_rhist,
            hist_image,
            ylabel=r"$\log_{10} |hi-lo|$ [g/gIHM]",
        )
        mass_summary["hist_image"] = info.hist_image

        metric_data = {
            "atom_fraction": {
                "summary": atom_summary,
                "hi": atom_hi,
                "lo": atom_lo,
                "ahist": atom_ahist,
                "rhist": atom_rhist,
            },
            "grams_per_initial_hm": {
                "summary": mass_summary,
                "hi": mass_hi,
                "lo": mass_lo,
                "ahist": mass_ahist,
                "rhist": mass_rhist,
            },
        }
        primary = metric_data[self.metric]
        secondary = (
            "atom_fraction"
            if self.metric == "grams_per_initial_hm"
            else "grams_per_initial_hm"
        )

        for key, value in primary["summary"].items():
            setattr(info, key, value)
        setattr(info, secondary, metric_data[secondary]["summary"])

        self.hi = primary["hi"]
        self.lo = primary["lo"]
        self.ahist = np.ndarray.flatten(primary["ahist"])
        self.rhist = np.ndarray.flatten(primary["rhist"])

        return info

    def _use_convergence_subdirectories(self):
        return self.nlib_max > self.nlib_start or self.nburn_max > self.nburn_start

    def _set_check_path_for_convergence_control(self, nlib, nburn):
        if not self._use_convergence_subdirectories():
            self.check_path = self.base_check_path
        else:
            check_path = self.base_check_path
            if self.nburn_max > self.nburn_start:
                check_path = check_path / f"nburn{nburn:04d}"
            if self.nlib_max > self.nlib_start:
                check_path = check_path / f"nlib{nlib:04d}"
            self.check_path = check_path
        self.check_dir = self.check_path.relative_to(self.work_path)

    @staticmethod
    def _matching_time_indices(reference_time, candidate_time):
        indices = []
        candidate_time = np.asarray(candidate_time, dtype=float)
        for time in np.asarray(reference_time, dtype=float):
            matches = np.where(np.isclose(candidate_time, time, rtol=0.0, atol=1.0e-6))[
                0
            ]
            if len(matches) != 1:
                raise ValueError(
                    "HIGH fidelity time="
                    f"{time} did not match exactly one LOWER fidelity time."
                )
            indices.append(int(matches[0]))
        return indices

    @staticmethod
    def _convergence_summary(info):
        keys = [
            "nlib",
            "nburn",
            "q1",
            "q2",
            "test_pass",
            "test_pass_q1",
            "test_pass_q2",
            "mean_abs_diff",
            "mean_rel_diff",
        ]
        return {key: getattr(info, key) for key in keys if hasattr(info, key)}

    def _scores_converged(self, previous_info, current_info, fixed_grid):
        if previous_info is None:
            return fixed_grid
        dq1 = abs(current_info.q1 - previous_info.q1)
        dq2 = abs(current_info.q2 - previous_info.q2)
        return dq1 <= self.q1_stop_criteria and dq2 <= self.q2_stop_criteria

    def __run_lo_fidelity(self, do_run, nlib, nburn):
        """Run the LOWER fidelity calculation which should be consistent as possible with
        the already-complete higher order calculation."""

        # Load the template file.
        with open(self.template_path, "r") as f:
            template_text = f.read()

        # Load the assemble data.
        assemble_json = self.work_path / "assemble.olm.json"
        with open(assemble_json, "r") as f:
            assemble_d = json.load(f)

        # For each point in space.
        ii_json_list = list()
        f71_list = list()
        input_list = list()
        self.initialhm_list = []
        for point in assemble_d["points"]:
            # Create the check input path.
            lib = Path(point["files"]["lib"])
            base = lib.stem
            check_input = self.check_path / base / (base + ".inp")

            # Save the list.
            hi_ii_json = self.work_path / point["files"]["ii_json"]
            lo_ii_json = check_input.with_suffix(".ii.json")
            f71_list.append(check_input.with_suffix(".f71"))
            ii_json_list.append((hi_ii_json, lo_ii_json))
            initialhm = point["history"]["initialhm"]
            if initialhm <= 0.0:
                raise ValueError(
                    "LowOrderConsistency requires positive initial heavy metal "
                    f"for point={base}"
                )
            self.initialhm_list.append(initialhm)

            # Create the directory.
            check_input.parent.mkdir(parents=True, exist_ok=True)

            # Populate data.
            check_data = {
                **point,
                "name": self.name,
                "convergence_control": {
                    "nburn": nburn,
                    "nlib": nlib,
                },
                "assembly_average": copy.deepcopy(self.assembly_average),
                "_": {"env": self._env, "model": self._model},
            }

            # Write out data file.
            check_data_file = check_input.parent / "data.olm.json"
            with open(check_data_file, "w") as f:
                f.write(json.dumps(check_data, indent=4))
            internal.logger.debug(
                "Writing LowOrderConsistency check", data_file=check_data_file
            )

            # Fill the template.
            filled_text = core.TemplateManager.expand_text(
                template_text,
                check_data,
                src_path=str(self.template_path),
                search_paths=self.template_paths,
            )

            # Write the check input file.
            internal.logger.debug(
                "Writing LowOrderConsistency check", input_file=check_input
            )
            input_list.append(str(check_input.relative_to(self.check_path)))
            with open(check_input, "w") as f:
                f.write(filled_text)

        # Use the makefile execution strategy for now.
        runinfo = internal._execute_makefile(
            dry_run=not do_run,
            _env=self._env,
            base_path=self.check_path,
            input_list=input_list,
        )

        # Actually generate the ii.json for the low fidelity calcs we just ran.
        if do_run:
            for f71 in f71_list:
                lo = internal.run_command(
                    f"{self._env['obiwan']} view -format=ii.json {f71} -cases='[{self.lo_case}]'",
                    echo=False,
                )
                lo_ii_json = f71.with_suffix(".ii.json")
                with open(lo_ii_json, "w") as f:
                    f.write(lo)

        return ii_json_list

    def __load_ii_json(self, ii_json_list):
        """Load the ii.json data that exists on disk for HIGH and LOWER fidelity into memory."""
        # We want nuclide data from one of the ii.json files.
        self.composition_manager = None

        # Convert the f71 to ii.json and extract the relevant information into memory.
        self.hi_list = list()
        self.lo_list = list()
        self.nuclide_data = None
        for hi_ii_json, lo_ii_json in ii_json_list:
            internal.logger.debug(f"loading HI {hi_ii_json}")
            # Load the json data into HIGH fidelity and LOWER fidelity data structures.
            # Note there's a little duplicate code here, but probably not worth refactoring.
            with open(hi_ii_json, "r") as f:
                jt = json.load(f)
                case = jt["responses"]["system"]

                # Just load once for the first available.
                if self.composition_manager == None:
                    self.composition_manager = core.CompositionManager(
                        jt["data"]["nuclides"]
                    )
                    self.nuclide_data = jt["data"]["nuclides"]

                hi = np.array(case["amount"])
                hi_vector = case["nuclideVectorHash"]
                self.hi_list.append(hi)
                self.names = jt["definitions"]["nuclideVectors"][hi_vector]
                self.time_list = case["time"]

            internal.logger.debug(f"loading LO {lo_ii_json}")

            with open(lo_ii_json, "r") as f:
                jo = json.load(f)
                case = jo["responses"][f"case({self.lo_case})"]
                lo = np.array(case["amount"])
                lo_time = case["time"]
                self.lo_list.append(lo)
                # Check consistency.
                if not np.array_equal(lo_time, self.time_list):
                    indices = self._matching_time_indices(self.time_list, lo_time)
                    lo = lo[indices, :]
                    lo_time = [lo_time[i] for i in indices]
                lo_vector = case["nuclideVectorHash"]
                if not lo_vector == hi_vector:
                    raise ValueError(
                        f"HIGH fidelity nuclide vector hash {hi_vector} is not the same as LOWER fidelity vector hash {lo_vector}, meaning the two nuclide sets are somehow inconsistent, which should not be possible."
                    )
                self.lo_list[-1] = lo

    def _run_once(self, do_run, nlib, nburn):
        self._set_check_path_for_convergence_control(nlib, nburn)
        self.ii_json_list = self.__run_lo_fidelity(do_run, nlib, nburn)
        self.__load_ii_json(self.ii_json_list)
        self.run_success = True
        info = self.info()
        info.nlib = nlib
        info.nburn = nburn
        info.nlib_converged = self.nlib_start == self.nlib_max
        info.nburn_converged = self.nburn_start == self.nburn_max
        info.test_pass_nlib = True
        info.test_pass_nburn = True
        return info

    def _run_nlib_convergence(self, do_run, nburn):
        previous_info = None
        current_info = None
        nlib_history = []
        nlib = self.nlib_start

        while True:
            current_info = self._run_once(do_run, nlib, nburn)
            converged = self._scores_converged(
                previous_info,
                current_info,
                fixed_grid=self.nlib_start == self.nlib_max,
            )
            if previous_info is not None:
                current_info.nlib_delta_q1 = abs(current_info.q1 - previous_info.q1)
                current_info.nlib_delta_q2 = abs(current_info.q2 - previous_info.q2)
            current_info.nlib_converged = converged
            current_info.test_pass_nlib = converged
            nlib_history.append(self._convergence_summary(current_info))

            if converged or nlib >= self.nlib_max:
                break

            previous_info = current_info
            nlib = min(nlib * 2, self.nlib_max)

        current_info.nlib_history = nlib_history
        return current_info

    def run(self, reactor_library):
        """Run a consistent set of LOWER fidelity calculations which also produce an
        f71--typically ORIGAMI."""

        # TODO: The reactor_library is not explicitly used because it was already expanded
        # into the Low Order/ORIGAMI input file. We may need to force some kind of
        # consistency here.

        # TODO: Allow input to change this or other smart way to determine if the data
        # does not need to be regenerated. Here, this is just for development iterations
        # to disable long SCALE runs while trying to debug checking.
        do_run = os.environ.get("SCALE_OLM_DO_RUN", "True") in ["True"]
        if not do_run:
            internal.logger.warning(
                "Runs suppressed by environment variable SCALE_OLM_DO_RUN!"
            )

        self.lo_case = 1
        previous_nburn_info = None
        current_info = None
        nburn_history = []
        nburn = self.nburn_start

        try:
            while True:
                current_info = self._run_nlib_convergence(do_run, nburn)
                converged = self._scores_converged(
                    previous_nburn_info,
                    current_info,
                    fixed_grid=self.nburn_start == self.nburn_max,
                )
                if previous_nburn_info is not None:
                    current_info.nburn_delta_q1 = abs(
                        current_info.q1 - previous_nburn_info.q1
                    )
                    current_info.nburn_delta_q2 = abs(
                        current_info.q2 - previous_nburn_info.q2
                    )
                current_info.nburn_converged = converged
                current_info.test_pass_nburn = converged
                nburn_history.append(self._convergence_summary(current_info))

                if not current_info.nlib_converged:
                    break
                if converged or nburn >= self.nburn_max:
                    break

                previous_nburn_info = current_info
                nburn = min(nburn * 2, self.nburn_max)

        except ValueError as ve:
            self.run_success = False
            internal.logger.error(str(ve))
            current_info = self.info()

        current_info.nlib = getattr(current_info, "nlib", self.nlib_start)
        current_info.nburn = getattr(current_info, "nburn", nburn)
        current_info.nlib_start = self.nlib_start
        current_info.nlib_max = self.nlib_max
        current_info.nburn_start = self.nburn_start
        current_info.nburn_max = self.nburn_max
        current_info.q1_stop_criteria = self.q1_stop_criteria
        current_info.q2_stop_criteria = self.q2_stop_criteria
        current_info.nlib_history = getattr(current_info, "nlib_history", [])
        current_info.nburn_history = nburn_history
        current_info.nlib_converged = getattr(
            current_info, "nlib_converged", self.nlib_start == self.nlib_max
        )
        current_info.nburn_converged = getattr(
            current_info, "nburn_converged", self.nburn_start == self.nburn_max
        )
        current_info.test_pass_nlib = getattr(
            current_info, "test_pass_nlib", current_info.nlib_converged
        )
        current_info.test_pass_nburn = getattr(
            current_info, "test_pass_nburn", current_info.nburn_converged
        )
        current_info.test_pass = (
            current_info.test_pass
            and current_info.test_pass_nlib
            and current_info.test_pass_nburn
        )

        return current_info
