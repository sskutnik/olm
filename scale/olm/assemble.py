import scale.olm.internal as internal
import scale.olm.core as core
import os
import json
from pathlib import Path
import os
import copy
import shutil
import numpy as np
import subprocess
import datetime
from typing import Literal

__all__ = ["arpdata_txt"]

_TYPE_ARPDATA_TXT = "scale.olm.assemble:arpdata_txt"
_POLARIS_BURNUP_RTOL = 5e-3
_TRITON_BURNUP_RTOL = 2e-2
_TRITON_FUEL_CASEID = -2
_POLARIS_FUEL_MATERIAL = "FUEL"
_POLARIS_FUEL_LIBRARY_SUFFIX = f".{_POLARIS_FUEL_MATERIAL}.f33"


def _schema_arpdata_txt(with_state: bool = False):
    _schema = internal._infer_schema(_TYPE_ARPDATA_TXT, with_state=with_state)
    return _schema


def _test_args_arpdata_txt(with_state: bool = False):
    return {
        "_type": _TYPE_ARPDATA_TXT,
        "dry_run": False,
        "fuel_type": "UOX",
        "dim_map": {"mod_dens": "mod_dens", "enrichment": "enrichment"},
    }


def arpdata_txt(
    fuel_type: str,
    dim_map: dict,
    keep_every: int,
    _model: dict = {},
    _env: dict = {},
    dry_run: bool = False,
    _type: Literal[_TYPE_ARPDATA_TXT] = None,
):
    """Build an ORIGEN reactor library in arpdata.txt format.

    Args:
        fuel_type: Which type of fuel: UOX/MOX.

        dim_map: arpdata.txt requires specially named dimensions. These may exist in the
                 state or you may need to map them from the state variables.

                 if fuel_type=='UOX', enrichment, mod_dens must be mapped to state variables
                 if fuel_type=='MOX', pu239_frac, pu_frac, mod_dens must be mapped to state variables


    """

    if dry_run:
        return {}

    # Get working directory.
    work_path = Path(_env["work_dir"])

    # Get library info data structure.
    arpinfo = _get_arpinfo(
        _env["obiwan"], work_path, _model["name"], fuel_type, dim_map
    )

    # Generate thinned burnup list.
    thinned_burnup_list = _generate_thinned_burnup_list(keep_every, arpinfo.burnup_list)

    # Process libraries into their final places.
    archive_file, points = _process_libraries(
        _env["obiwan"], work_path, arpinfo, thinned_burnup_list
    )

    return {
        "archive_file": archive_file,
        "points": points,
        "work_dir": str(work_path),
        "date": datetime.datetime.utcnow().isoformat(" ", "minutes"),
        "space": arpinfo.get_space(),
    }


def archive(model):
    """Build an ORIGEN reactor library in HDF5 archive format.

    Args:
        model (dict): A dictionary containing the following keys:
            - archive_file (str): The path and filename of the reactor archive to be created.
            - work_dir (str): The path to the working directory.
            - name (str): The name of the reactor.
            - obiwan (str): The path to the OBIWAN executable.

    Returns:
        dict: relevant data on the result of creating an archive
    """
    archive_file = model["archive_file"]
    config_file = model["work_dir"] + os.path.sep + "generate.olm.json"

    # Load the permuation data
    with open(config_file, "r") as f:
        data = json.load(f)

    assem_tag = "assembly_type={:s}".format(model["name"])
    lib_paths = []

    # Tag each permutation's libraries
    for perm in data["perms"]:
        perm_dir = Path(perm["input_file"]).parent
        perm_name = Path(perm["input_file"]).stem
        statevars = perm["state"]
        lib_path = os.path.join(perm_dir, perm_name + ".system.f33")
        lib_paths.append(lib_path)
        internal.logger.debug(f"Now tagging {lib_path}")

        ts = ",".join(key + "=" + str(value) for key, value in statevars.items())
        try:
            subprocess.run(
                [
                    model["obiwan"],
                    "tag",
                    lib_path,
                    f"-interptags={ts}",
                    f"-idtags={assem_tag}",
                ],
                capture_output=True,
                check=True,
            )
        except subprocess.SubprocessError as error:
            print(error)
            print("OBIWAN library tagging failed; cannot assemble archive")

    to_consolidate = " ".join(lib for lib in lib_paths)
    internal.logger.info(f"Building archive at {archive_file} ... ")
    try:
        subprocess.run(
            [
                model["obiwan"],
                "convert",
                "-format=hdf5",
                "-name={archive_file}",
                to_consolidate,
            ],
            check=True,
        )
    except subprocess.SubprocessError as error:
        print(error)
        print("OBIWAN library conversion to archive format failed")

    return {"archive_file": archive_file}


def _generate_thinned_burnup_list(keep_every, y_list, always_keep_ends=True):
    """Generate a thinned list using every point (1), every other point (2),
    every third point (3), etc."""

    if not keep_every > 0:
        raise ValueError(
            "The thinning parameter keep_every={keep_every} must be an integer >0!"
        )

    thinned_burnup_list = list()
    j = 0
    rm = 1
    for y in y_list:
        if always_keep_ends and (j == 0 or j == len(y_list) - 1):
            p = True
        elif rm >= keep_every:
            p = True
        else:
            p = False
        if p:
            thinned_burnup_list.append(y)
            rm = 0
        rm += 1
        j += 1
    return thinned_burnup_list


def _get_scale_metadata(work_dir, perm):
    if "_scale" in perm:
        return perm["_scale"]

    input_file = work_dir / perm["input_file"]
    if not input_file.exists():
        raise ValueError(
            "permutation is missing _scale input classification and "
            f"input file={input_file} does not exist"
        )

    with open(input_file, "r") as f:
        return core.ScaleInput.classify_text(f.read())


def _get_artifact_contract(work_dir, perm):
    scale_metadata = _get_scale_metadata(work_dir, perm)
    if "artifact_contract" not in scale_metadata:
        raise ValueError(
            "SCALE input classification is missing artifact_contract "
            f"for input_file={perm['input_file']}"
        )
    artifact_contract = scale_metadata["artifact_contract"]
    if artifact_contract not in ["TRITON", "Polaris"]:
        raise ValueError(
            f"Unsupported SCALE artifact contract={artifact_contract} "
            f"for input_file={perm['input_file']}"
        )
    return artifact_contract


def _get_library_suffix(default_suffix, artifact_contract):
    if artifact_contract == "Polaris":
        return _POLARIS_FUEL_LIBRARY_SUFFIX
    return default_suffix


def _get_files(work_dir, suffix, perms):
    """Get list of files by using the generate.olm.json output and changing the suffix to the
    expected library file. Note this is in permutation order, not state space order."""

    file_list = list()
    for perm in perms:
        input = perm["input_file"]
        artifact_contract = _get_artifact_contract(work_dir, perm)
        lib_suffix = _get_library_suffix(suffix, artifact_contract)

        # Convert from .inp to expected suffix.
        lib = work_dir / Path(input)
        lib = lib.with_suffix(lib_suffix)
        if not lib.exists():
            raise ValueError(f"library file={lib} does not exist!")

        output = work_dir / Path(input).with_suffix(".out")
        if not output.exists():
            raise ValueError(
                f"output file={output} does not exist! Maybe run was not complete successfully?"
            )

        file_info = {
            "lib": lib,
            "output": output,
            "artifact_contract": artifact_contract,
        }

        f71 = work_dir / Path(input).with_suffix(".f71")
        if not f71.exists():
            raise ValueError(f"f71 file={f71} does not exist!")
        file_info["f71"] = f71

        file_list.append(file_info)

    return file_list


def _get_burnup_list(obiwan, file_list):
    """Extract a burnup list from the output file and make sure they are all the same."""
    burnup_list = list()
    previous_output_file = ""
    for i in range(len(file_list)):
        output_file = file_list[i]["output"]
        artifact_contract = file_list[i]["artifact_contract"]

        if artifact_contract == "Polaris":
            if "f71" not in file_list[i]:
                raise ValueError(
                    f"Polaris file info is missing f71 for output_file={output_file}"
                )
            caseid = _get_polaris_fuel_caseid_from_output(output_file)
            output_file = file_list[i]["f71"]
            bu = core.Obiwan.get_burnups_from_f71(obiwan, output_file, caseid)
        elif artifact_contract == "TRITON":
            bu = core.ScaleOutfile.parse_burnups_from_triton_output(output_file)
        else:
            raise ValueError(
                f"Unsupported SCALE artifact contract={artifact_contract} "
                f"for output_file={output_file}"
            )

        if len(burnup_list) == 0:
            burnup_list = bu
        elif not _burnup_lists_match(burnup_list, bu, artifact_contract):
            raise ValueError(
                f"Output file={output_file} burnups deviated from previous {previous_output_file}!"
            )
        previous_output_file = output_file

    return burnup_list


def _burnup_lists_match(reference, candidate, artifact_contract):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        return False
    if artifact_contract == "Polaris":
        return np.allclose(
            reference,
            candidate,
            rtol=_POLARIS_BURNUP_RTOL,
            atol=1e-6,
        )
    if artifact_contract == "TRITON":
        return np.allclose(
            reference,
            candidate,
            rtol=_TRITON_BURNUP_RTOL,
            atol=1e-6,
        )
    return np.array_equal(reference, candidate)


def _get_triton_fuel_caseid(work_dir, perm):
    return _TRITON_FUEL_CASEID, False


def _get_polaris_fuel_caseid_from_output(output_file):
    caseid = core.ScaleOutfile.parse_polaris_state_table(
        output_file, _POLARIS_FUEL_MATERIAL
    )
    if caseid == -2:
        raise ValueError(
            f"Cannot identify Polaris {_POLARIS_FUEL_MATERIAL} case "
            f"from output file={output_file}"
        )
    return caseid


def _get_polaris_fuel_caseid(work_dir, perm):
    outfile = (work_dir / perm["input_file"]).with_suffix(".out")
    return _get_polaris_fuel_caseid_from_output(outfile), True


def _get_fuel_caseid(work_dir, perm):
    artifact_contract = _get_artifact_contract(work_dir, perm)

    if artifact_contract == "TRITON":
        return _get_triton_fuel_caseid(work_dir, perm)

    return _get_polaris_fuel_caseid(work_dir, perm)


def _validate_fuel_caseid(work_dir, perm, ii, caseid, is_polaris):
    if f"case({caseid})" not in ii["responses"]:
        if is_polaris:
            outfile = (work_dir / perm["input_file"]).with_suffix(".out")
            raise ValueError(
                f"Polaris {_POLARIS_FUEL_MATERIAL} case {caseid} "
                f"from output file={outfile} "
                "not found in F71 table"
            )
        raise ValueError(
            "Cannot identify TRITON fuel case; case -2 not found in F71 table"
        )

    return caseid, is_polaris


def _get_fuel_caseid_from_ii(work_dir, perm, ii):
    caseid, is_polaris = _get_fuel_caseid(work_dir, perm)
    return _validate_fuel_caseid(work_dir, perm, ii, caseid, is_polaris)


def _get_fuel_ii_json(obiwan, work_dir, perm):
    caseid, is_polaris = _get_fuel_caseid(work_dir, perm)
    f71 = (work_dir / perm["input_file"]).with_suffix(".f71")
    text = internal.run_command(
        f"{obiwan} view -format=ii.json {f71} -cases='[{caseid}]'",
        echo=False,
    )
    ii = json.loads(text)
    _validate_fuel_caseid(work_dir, perm, ii, caseid, is_polaris)
    ii["responses"]["system"] = ii["responses"].pop(f"case({caseid})")
    return ii, caseid, is_polaris


def _get_triton_history_from_output(obiwan, output_file, f71, caseid):
    rows = core.ScaleOutfile.parse_triton_library_table(output_file)
    initialhm = core.Obiwan.get_initialhm_from_f71(obiwan, f71, caseid)
    burndata = [{"power": row["power"], "burn": row["burn"]} for row in rows]
    return {"burndata": burndata, "initialhm": initialhm}


def _get_history(obiwan, work_dir, perm, f71, caseid, is_polaris):
    artifact_contract = _get_artifact_contract(work_dir, perm)
    if artifact_contract == "TRITON":
        output_file = (work_dir / perm["input_file"]).with_suffix(".out")
        return _get_triton_history_from_output(obiwan, output_file, f71, caseid)

    return core.Obiwan.get_history_from_f71(obiwan, f71, caseid, is_polaris)


def _get_arpinfo_uox(name, perms, file_list, dim_map):
    """For UOX, get the relative ARP interpolation information."""

    # Get the names of the keys in the state.
    key_e = dim_map["enrichment"]
    key_m = dim_map["mod_dens"]

    # Build these lists for each permutation to use in init_uox below.
    enrichment_list = []
    mod_dens_list = []
    lib_list = []
    for i in range(len(perms)):
        # Get the interpolation variables from the state.
        state = perms[i]["state"]
        e = state[key_e]
        enrichment_list.append(e)
        m = state[key_m]
        mod_dens_list.append(m)

        # Get the library name.
        lib_list.append(file_list[i]["lib"])

    # Create and return arpinfo.
    arpinfo = core.ArpInfo()
    arpinfo.init_uox(name, lib_list, enrichment_list, mod_dens_list)
    return arpinfo


def _get_arpinfo_mox(name, perms, file_list, dim_map):
    """For MOX, get the relative ARP interpolation information."""

    # Get the names of the keys in the state.
    key_e = dim_map["pu239_frac"]
    key_p = dim_map["pu_frac"]
    key_m = dim_map["mod_dens"]

    # Build these lists for each permutation to use in init_uox below.
    pu239_frac_list = []
    pu_frac_list = []
    mod_dens_list = []
    lib_list = []
    for i in range(len(perms)):
        # Get the interpolation variables from the state.
        state = perms[i]["state"]
        e = state[key_e]
        pu239_frac_list.append(e)
        p = state[key_p]
        pu_frac_list.append(p)
        m = state[key_m]
        mod_dens_list.append(m)

        # Get the library name.
        lib_list.append(file_list[i]["lib"])

    # Create and return arpinfo.
    arpinfo = core.ArpInfo()
    arpinfo.init_mox(name, lib_list, pu239_frac_list, pu_frac_list, mod_dens_list)
    return arpinfo


def _get_arpinfo(obiwan, work_dir, name, fuel_type, dim_map):
    """Populate the ArpInfo data."""

    # Get generate data which has permutations list with file names.
    generate_json = work_dir / "generate.olm.json"
    with open(generate_json, "r") as f:
        generate = json.load(f)
    perms = generate["perms"]

    # Get library,input,output in one place.
    suffix = ".system.f33"
    file_list = _get_files(work_dir, suffix, perms)

    # Initialize info based on fuel type.
    if fuel_type == "UOX":
        arpinfo = _get_arpinfo_uox(name, perms, file_list, dim_map)
    elif fuel_type == "MOX":
        arpinfo = _get_arpinfo_mox(name, perms, file_list, dim_map)
    else:
        raise ValueError(
            "Unknown fuel_type={fuel_type} (only MOX/UOX is supported right now)"
        )

    # Get the burnups.
    arpinfo.burnup_list = _get_burnup_list(obiwan, file_list)

    # Set new canonical file names.
    arpinfo.set_canonical_filenames(".h5")

    return arpinfo


def _get_comp_system(ii_data):
    """Extract the following information from the inventory interface (ii) data."""

    x = ii_data["responses"]["system"]
    volume = x["volume"]
    amount_list = x["amount"][0]  # Initial amount
    data_map = ii_data["data"]["nuclides"]
    vh = x["nuclideVectorHash"]
    nuclide_list = ii_data["definitions"]["nuclideVectors"][vh]

    x = dict()
    total_mass = 0.0
    for i in range(len(nuclide_list)):
        name = nuclide_list[i]
        data = data_map[name]
        amount = amount_list[i]
        molar_mass = data["mass"]
        mass = amount * molar_mass
        total_mass += mass
        z = data["atomicNumber"]
        e = data["element"]
        m = data["isomericState"]
        a = data["massNumber"]
        mstr = ""
        if m >= 1:
            mstr = "m"
        elif m >= 2:
            mstr = "m" + str(m)
        eam = "{}{}{}".format(e.lower(), int(a), mstr)
        if z >= 92:
            x[eam] = amount * molar_mass

    comp = core.CompositionManager.calculate_hm_oxide_breakdown(x)
    comp["info"] = core.CompositionManager.approximate_hm_info(comp)
    comp["density"] = total_mass / volume

    return comp


def _process_libraries(obiwan, work_dir, arpinfo, thinned_burnup_list):
    """Process libraries with OBIWAN, including copying, thinning, setting tags, etc."""

    # Create the arplibs directory and clear data files inside.
    d = work_dir / "arplibs"
    if d.exists():
        shutil.rmtree(d)
    os.mkdir(d)

    # Generate burnup string.
    bu_str = ",".join([str(bu) for bu in arpinfo.burnup_list])

    # Generate idtags.
    idtags = "assembly_type={:s},fuel_type={:s}".format(arpinfo.name, arpinfo.fuel_type)

    # Generate burnup string for thin list.
    thin_bu_str = ",".join([str(bu) for bu in thinned_burnup_list])
    internal.logger.info("burnup thinning:", original_bu=bu_str, thinned_bu=thin_bu_str)
    arpinfo.burnup_list = thinned_burnup_list

    # Create a temporary directory for libraries in process.
    tmp = d / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    # Get generate data which has permutations list with file names.
    generate_json = work_dir / "generate.olm.json"
    with open(generate_json, "r") as f:
        generate = json.load(f)
    perms = generate["perms"]

    # Use obiwan to perform most of the processes.
    points = list()
    for i in range(arpinfo.num_libs()):
        new_lib = Path(arpinfo.get_lib_by_index(i))
        old_lib = Path(arpinfo.origin_lib_list[i])
        tmp_lib = tmp / old_lib.name
        internal.logger.debug(f"Copying original library {old_lib} to {tmp_lib}")
        shutil.copyfile(old_lib, tmp_lib)

        # Set burnups on file using obiwan (should only be necessary in earlier SCALE versions).
        internal.run_command(
            f"{obiwan} convert -i -setbu='[{bu_str}]' {tmp_lib}", echo=False
        )
        bad_local = Path(tmp_lib.with_suffix(".f33").name)
        if bad_local.exists():
            internal.logger.warning("Fixup: relocating local", file=str(bad_local))
            shutil.move(bad_local, tmp_lib)

        # Perform burnup thinning.
        if bu_str != thin_bu_str:
            internal.run_command(
                f"{obiwan} convert -i -thin=1 -tvals='[{thin_bu_str}]' {tmp_lib}",
                check_return_code=False,
                echo=False,
            )
            if bad_local.exists():
                internal.logger.warning("Fixup: relocating local", file=str(bad_local))
                shutil.move(bad_local, tmp_lib)

        # Set tags.
        interptags = arpinfo.interptags_by_index(i)
        internal.run_command(
            f"{obiwan} tag -interptags='{interptags}' -idtags='{idtags}' {tmp_lib}",
            echo=False,
        )

        # Convert to HDF5.
        internal.run_command(
            f"{obiwan} convert -format=hdf5 -type=f33 {tmp_lib} -dir={tmp}", echo=False
        )

        # Move the local library to the new proper place.
        new_lib = d / arpinfo.get_lib_by_index(i)
        shutil.move(tmp_lib.with_suffix(".h5"), new_lib)

        # Generate the system composition information from the system ii.json.
        k = arpinfo.get_perm_by_index(i)
        perm = perms[k]
        f71 = (work_dir / perm["input_file"]).with_suffix(".f71")

        # Load into data structure and rename.
        ii_json = new_lib.with_suffix(".ii.json")
        internal.logger.debug(f"Converting {f71} to {ii_json}")

        ii, caseid, is_polaris = _get_fuel_ii_json(obiwan, work_dir, perm)
        with open(ii_json, "w") as f:
            f.write(json.dumps(ii, indent=4))

        # Get the special composition data structure.
        comp_system = _get_comp_system(ii)

        # Save relevant permutation data in a list.
        points.append(
            {
                "files": {
                    "origin": {
                        "lib": str(old_lib.relative_to(work_dir)),
                        "f71": str(f71.relative_to(work_dir)),
                    },
                    "lib": str(new_lib.relative_to(work_dir)),
                    "ii_json": str(ii_json.relative_to(work_dir)),
                },
                "comp": {
                    "system": comp_system,
                },
                "history": _get_history(
                    obiwan, work_dir, perm, f71, caseid, is_polaris
                ),
                "_": {"perm": perm},
                "_arpinfo": {
                    "interpvars": {**arpinfo.interpvars_by_index(i)},
                    "burnup_list": arpinfo.burnup_list,
                },
            }
        )

    # Remove temporary files.
    shutil.rmtree(tmp)

    # Write arpdata.txt.
    arpdata_txt = work_dir / "arpdata.txt"
    internal.logger.info(f"Writing arpdata.txt at {arpdata_txt} ... ")
    with open(arpdata_txt, "w") as f:
        f.write(arpinfo.get_arpdata())
    archive_file = "arpdata.txt:" + arpinfo.name

    return archive_file, points
