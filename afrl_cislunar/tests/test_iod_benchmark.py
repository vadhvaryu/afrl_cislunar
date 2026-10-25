"""IOD benchmark harness (B3): job lists and summaries. The heavy runs are not part of the test suite."""

from od_stack import iod_benchmark as b


def test_job_list_follows_the_trimmed_plan():
    js = b.jobs(["cr3bp_dc", "gauss", "halo_catalog"], ["current", "dense_6h"], [1.0, 10.0], range(3))
    count = {m: sum(j["method"] == m for j in js) for m in ("cr3bp_dc", "gauss", "halo_catalog")}
    assert count == {"cr3bp_dc": 1 * 2 * 2 * 3, "gauss": 4 * 2 * 2 * 3, "halo_catalog": 1 * 2 * 2 * 3}
    assert {j["triple"] for j in js if j["method"] == "halo_catalog"} == {""}


def test_summary_and_cross_check():
    rows = []
    for s in range(10):
        rows.append({"method": "m1", "arc": "current", "triple": "", "noise_arcsec": "10.0", "seed": str(s),
                     "iod_ok": "True", "iod_within_10000km": "True", "outcome": "converged" if s < 9 else "wrong_orbit",
                     "iod_position_error_km": "100", "iod_seconds": "1", "total_seconds": "5",
                     "iod_rx_m": "1e8", "iod_ry_m": "0", "iod_rz_m": "0", "iod_epoch_days": "4.0"})
        rows.append({**rows[-1], "method": "m2", "iod_within_10000km": "False", "outcome": "converged", "iod_rx_m": "1.01e8"})
    summ = {a["method"]: a for a in b.summarize(rows)}
    assert summ["m1"]["works"] and not summ["m2"]["works"]           # 9/10 converged, 10/10 within; m2 0/10 within
    cc = b.cross_check(rows)
    assert len(cc) == 10 and abs(cc[0]["distance_km"] - 1000.0) < 1e-6
