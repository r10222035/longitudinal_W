"""Common table-format printers for expected yields."""

from __future__ import annotations

import numpy as np


def print_expected_event_counts_table(
    results,
    luminosity_fb_inv,
    *,
    title_prefix="Expected event counts comparison table",
    sample_width=16,
    include_na=False,
    diagnostics=None,
    extended_mode=False,
):
    """Print a unified expected-yield table.

    Parameters
    ----------
    results : dict
        Mapping sample_name -> result dict with keys
        n_expected_sr / n_expected_low_mjj / n_expected_wz.
        Optional extended keys: sigma_gen_fb, sigma_sr_fb, n_total_events, 
        n_passed_events.
    luminosity_fb_inv : float
        Integrated luminosity in fb^-1.
    title_prefix : str, optional
        Prefix shown before luminosity string.
    sample_width : int, optional
        Width of sample-name column.
    include_na : bool, optional
        If True, allow NaN rows and print as N/A.
    diagnostics : dict or None, optional
        Optional Sherpa diagnostics block to print below table.
    extended_mode : bool, optional
        If True and extended keys are present, show additional columns
        (Initial XS, Total Events, Passed Events, Pass Rate, Final XS).
    """
    title = f"{title_prefix} (L = {luminosity_fb_inv:.0f} fb^-1)"
    print(title)

    # Check if we have extended data (sigma_gen_fb, n_total_events, etc.)
    has_extended = extended_mode and any(
        v.get("sigma_gen_fb") is not None or v.get("n_total_events") is not None
        for v in results.values()
    )

    if has_extended:
        # Extended table format
        line_len = 135
        print("=" * line_len)
        print(
            f"{'Sample':<{sample_width}} {'Region':<8} {'Initial XS (fb)':>15} "
            f"{'Total Events':>15} {'Passed Events':>15} {'Pass Rate':>12} "
            f"{'Final XS (fb)':>15} {'Expected Events (139/fb)':>24}"
        )
        print("-" * line_len)

        if include_na:
            for sample_name, vals in sorted(results.items()):
                region = vals.get("region", "SR")
                if np.isnan(vals.get("n_expected_sr", np.nan)):
                    print(
                        f"{sample_name:<{sample_width}} {region:<8} "
                        f"{'N/A':>15} {'N/A':>15} {'N/A':>15} {'N/A':>12} "
                        f"{'N/A':>15} {'N/A':>24}"
                    )
                else:
                    init_xs = vals.get("sigma_gen_fb", np.nan)
                    final_xs = vals.get("sigma_sr_fb", np.nan)
                    n_total = vals.get("n_total_events", int(vals.get("n_expected_sr", 0)))
                    n_passed = vals.get("n_passed_events", int(vals.get("n_expected_sr", 0)))
                    pass_rate = (n_passed / n_total * 100) if n_total > 0 else 0.0
                    n_exp = vals.get("n_expected_sr", 0.0)

                    init_xs_str = f"{init_xs:.6e}" if not np.isnan(init_xs) else "N/A"
                    final_xs_str = f"{final_xs:.6e}" if not np.isnan(final_xs) else "N/A"

                    print(
                        f"{sample_name:<{sample_width}} {region:<8} "
                        f"{init_xs_str:>15} {n_total:>15} {n_passed:>15} "
                        f"{pass_rate:>11.2f}% {final_xs_str:>15} {n_exp:>24.2f}"
                    )
        else:
            for sample_name, vals in sorted(results.items()):
                region = vals.get("region", "SR")
                init_xs = vals.get("sigma_gen_fb", np.nan)
                final_xs = vals.get("sigma_sr_fb", np.nan)
                n_total = vals.get("n_total_events", int(vals.get("n_expected_sr", 0)))
                n_passed = vals.get("n_passed_events", int(vals.get("n_expected_sr", 0)))
                pass_rate = (n_passed / n_total * 100) if n_total > 0 else 0.0
                n_exp = vals.get("n_expected_sr", 0.0)

                init_xs_str = f"{init_xs:.6e}" if not np.isnan(init_xs) else "N/A"
                final_xs_str = f"{final_xs:.6e}" if not np.isnan(final_xs) else "N/A"

                print(
                    f"{sample_name:<{sample_width}} {region:<8} "
                    f"{init_xs_str:>15} {n_total:>15} {n_passed:>15} "
                    f"{pass_rate:>11.2f}% {final_xs_str:>15} {n_exp:>24.2f}"
                )

        print("-" * line_len)
        valid_values = [v for v in results.values() if not np.isnan(v.get("n_expected_sr", np.nan))]
        sum_n_exp = sum(v.get("n_expected_sr", 0) for v in valid_values)
        print(
            f"{'SUM':<{sample_width}} {'':8} {'':>15} {'':>15} {'':>15} {'':>12} "
            f"{'':>15} {sum_n_exp:>24.2f}"
        )
    else:
        # Standard compact table format
        line_len = max(75, sample_width + 56)
        print("=" * line_len)
        print(f"{'Sample':<{sample_width}} {'N_exp(SR)':>15} {'N_exp(Low-mjj CR)':>19} {'N_exp(WZ CR)':>15}")
        print("-" * line_len)

        if include_na:
            for sample_name, vals in sorted(results.items()):
                if np.isnan(vals.get("n_expected_sr", np.nan)):
                    sr_text = "N/A"
                    low_text = "N/A"
                    wz_text = "N/A"
                else:
                    sr_text = f"{vals['n_expected_sr']:>15.2f}"
                    low_text = f"{vals['n_expected_low_mjj']:>19.2f}"
                    wz_text = f"{vals['n_expected_wz']:>15.2f}"
                print(f"{sample_name:<{sample_width}} {sr_text} {low_text} {wz_text}")

            print("-" * line_len)
            valid_values = [v for v in results.values() if not np.isnan(v.get("n_expected_sr", np.nan))]
            sum_n_sr = sum(v["n_expected_sr"] for v in valid_values)
            sum_n_low_mjj = sum(v["n_expected_low_mjj"] for v in valid_values)
            sum_n_wz = sum(v["n_expected_wz"] for v in valid_values)
        else:
            for sample_name, vals in sorted(results.items()):
                print(
                    f"{sample_name:<{sample_width}} "
                    f"{vals['n_expected_sr']:>15.2f} "
                    f"{vals['n_expected_low_mjj']:>19.2f} "
                    f"{vals['n_expected_wz']:>15.2f}"
                )

            print("-" * line_len)
            sum_n_sr = sum(v["n_expected_sr"] for v in results.values())
            sum_n_low_mjj = sum(v["n_expected_low_mjj"] for v in results.values())
            sum_n_wz = sum(v["n_expected_wz"] for v in results.values())

        print(
            f"{'SUM':<{sample_width}} "
            f"{sum_n_sr:>15.2f} "
            f"{sum_n_low_mjj:>19.2f} "
            f"{sum_n_wz:>15.2f}"
        )

    if diagnostics is not None:
        print("\nDiagnostics:")
        for sample_name, diag in diagnostics.items():
            pairing = diag.get("pairing", {})
            xsec_stats = diag.get("xsec_stats", {})
            if xsec_stats.get("n_valid", 0) == 0:
                print(f"  {sample_name:<10} missing reconstructed ROOT data")
                continue
            print(
                f"  {sample_name:<10} pairs={pairing.get('n_pairs', 0):<4} "
                f"valid_xsec={xsec_stats.get('n_valid', 0):<6} "
                f"mean_xsec_pb={xsec_stats.get('mean_xsec_pb', np.nan):.6e}"
            )