##  Required imports
import matplotlib

matplotlib.use("Agg")
import sdesk
import os
from VAMAS import parseVAMAS, extract_vamas_data_and_meta
from data_transformation import (
    aggregate_to_sample_data,
    convert_data_layers_to_img,
    GAF_transform,
)
import numpy as np
from scipy.optimize import curve_fit

##
global global_deltas_ke


def get_optimizaton_parameters(ydata, e_b):
    max_y = np.max(ydata)
    low_intensity_bound = np.zeros(e_b["num_peaks"])
    high_intensity_bound = np.ones(e_b["num_peaks"]) * max_y
    low_bkg_params_bound = [-max_y / 100, 0]
    high_bkg_params_bound = [max_y / 100, max_y]

    low_bound = [
        e_b["low_ke_bound"],
        e_b["low_peak_width_bound"],
        *low_intensity_bound,
        *low_bkg_params_bound,
    ]
    high_bound = [
        e_b["high_ke_bound"],
        e_b["high_peak_width_bound"],
        *high_intensity_bound,
        *high_bkg_params_bound,
    ]
    p0 = (np.array(high_bound) + np.array(low_bound)) / 2.0

    return (low_bound, high_bound), p0


def get_energy_params_from_user_input(input_form):
    ref_peaks = np.array(
        [
            float(r)
            for r in input_form["reference_peaks"].replace(",", ".").split("\n")
            if r.strip()
        ]
    )
    photon_energy = float(str(input_form["photon_energy"]).replace(",", "."))
    ke_refs_np = photon_energy - ref_peaks
    min_spectra_range = [np.min(ke_refs_np) - 1, np.max(ke_refs_np) + 1]

    fixed_deltas_ke = [ke - ke_refs_np[0] for ke in ke_refs_np]
    low_ke_bound = np.min(ke_refs_np) - 0.2  # ev
    high_ke_bound = np.max(ke_refs_np) + 5.0  # ev

    return {
        "num_peaks": len(ref_peaks),
        "photon_energy": photon_energy,
        "ke_refs_np": ke_refs_np,
        "low_ke_bound": low_ke_bound,
        "high_ke_bound": high_ke_bound,
        "fixed_deltas_ke": fixed_deltas_ke,
        "min_spectra_range": min_spectra_range,
        "work_function": input_form.get("work_function", None),
        "low_peak_width_bound": 1e-3,
        "high_peak_width_bound": 2,
    }


def func(xx, *p):
    global global_deltas_ke
    mu_s = [p[0] + d for d in global_deltas_ke]
    sig = p[1]
    intensities = p[2:-2]
    a = p[-2]
    b = p[-1]

    y = 0
    for i in range(0, len(mu_s)):
        y = y + intensities[i] * np.exp(
            -np.power(xx - mu_s[i], 2.0) / (2 * np.power(sig, 2.0))
        )
    y = y + a * (xx - mu_s[0]) + b
    return y


# Define your method main()
def main():
    global global_deltas_ke
    process = sdesk.Process()

    # LOAD INPUT FILES OR SAMPLES
    input_file = process.input_files[0]
    filename, file_extension = os.path.splitext(input_file.properties["name"])
    if file_extension.lower() not in [".vms", ".vamas"]:
        """STOP AND DO NOTHING"""
        return 0

    # LOAD PARAMETERS FROM USER INPUT FORM
    input_form = process.arguments
    e_b = get_energy_params_from_user_input(input_form)
    photon_energy = e_b["photon_energy"]
    ke_reference = e_b["ke_refs_np"][0]
    global_deltas_ke = e_b["fixed_deltas_ke"]
    work_function = e_b["work_function"]

    # PROCESS INPUT FILE AND EXTRACT DATA
    output_files = []

    def is_in_range(minx, maxx, spectra_range):
        return minx < spectra_range[0] and maxx > spectra_range[1]

    with open(input_file.path, "r") as fp:
        vamas_data = parseVAMAS(fp)
        for dict_data in vamas_data["blocks"]:
            jHeader, jInfo, data = extract_vamas_data_and_meta(dict_data)
            data = np.array(data)
            xdata = data[:, 0]
            minx = np.min(xdata)
            maxx = np.max(xdata)
            print("block_identifier")
            print(dict_data["block_identifier"])

            output_files.append(
                {
                    "name": "{}_be.txt".format(dict_data["block_identifier"]),
                    "columns": jInfo["columnNames"],
                    "data": data,
                    "is_in_range": is_in_range(minx, maxx, e_b["min_spectra_range"]),
                    "header": sdesk.json_to_text(jHeader),
                    "zoom": 1 / (maxx - minx + 1),
                }
            )

    # Calculate binding energy and update metadata
    files_to_be_fit = [f for f in output_files if f["is_in_range"]]
    if len(files_to_be_fit):
        files_to_be_fit = sorted(files_to_be_fit, key=lambda f: (-f["zoom"]))

        # Use the file with best zoom
        xdata = files_to_be_fit[0]["data"][:, 0]
        ydata = files_to_be_fit[0]["data"][:, 1]

        bounds, p0 = get_optimizaton_parameters(ydata, e_b)
        popt, pcov = curve_fit(func, xdata, ydata, bounds=bounds, p0=p0)
        work_function = popt[0] - ke_reference

        # Convert KE to BE for all spectra
        for file in output_files:
            file["columns"][0] = "Binding Energy (eV)"
            xdata = file["data"][:, 0]
            file["data"][:, 0] = photon_energy - xdata + work_function

    # UPDATE CUSTOM PROPERTIES OF INPUT FILE
    new_properties = {
        "phonton_energy": photon_energy,
        "work_function": work_function,
        "energy references": input_form.get("reference_peaks", "").replace("\n", " "),
    }
    if input_file.custom_properties:
        input_file.custom_properties.update(new_properties)
    else:
        input_file.custom_properties = new_properties
    input_file.save_custom_properties()

    # Create output files derived from the processing of input file
    for out_file in output_files:
        sdesk_output_file = process.create_output_file(out_file["name"])
        sdesk.write_tsv_file(
            sdesk_output_file.path,
            out_file["columns"],
            out_file["data"],
            out_file["header"],
        )

    # Aggregate data to sample and output for visualization
    linked_subject = input_file.subject
    if linked_subject:
        data_obj, path = aggregate_to_sample_data(
            linked_subject, [o["data"] for o in output_files], "xps"
        )
        linked_subject.save_as_aggregated_data(path)

        sdesk_output_file = process.create_output_file("aggregated_xps_table.txt")
        sdesk.write_tsv_file(
            sdesk_output_file.path, ["energy", "yield"], data_obj["xps"]
        )

        gaf_xps, _, _ = GAF_transform(data_obj["xps"][:, 1])

        if data_obj.get("eels", None) is not None:
            gaf_eels, _, _ = GAF_transform(data_obj["eels"][:, 1])
        else:
            gaf_eels = np.zeros(gaf_xps.shape)

        img = convert_data_layers_to_img(gaf_xps, gaf_eels, gaf_eels * 0)
        aggregated_output_file = process.create_output_file(
            f"GAF_sample_{linked_subject.uid}.jpeg"
        )
        img.save(aggregated_output_file.path)


# Call method main()
main()
