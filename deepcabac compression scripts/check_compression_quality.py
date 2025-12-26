import os
import csv
import glob

def safe_float(x):
    try:
        return float(x)
    except:
        return None

def safe_int(x):
    try:
        return int(float(x))
    except:
        return None


def analyze_csv(path):
    bad_ratio = []
    bad_gain = []
    overhead_records = []

    total_rows = 0

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1

            pname = row.get("param_name", "UNKNOWN")
            bits = row.get("bits_quant", "??")

            ratio = safe_float(row.get("compression_ratio"))
            gain = safe_float(row.get("compression_gain_pct"))
            raw_b = safe_int(row.get("raw_bytes"))
            comp_b = safe_int(row.get("compressed_bytes"))

            # ----------------------------------------------------------
            # Detect bad compression
            # ----------------------------------------------------------
            condition_bad = False

            if ratio is not None and ratio < 1:
                condition_bad = True
                bad_ratio.append((pname, bits, ratio))

            if gain is not None and gain < 0:
                condition_bad = True
                bad_gain.append((pname, bits, gain))

            # ----------------------------------------------------------
            # For every "bad" tensor, compute overhead
            # ----------------------------------------------------------
            if condition_bad and raw_b is not None and comp_b is not None and comp_b > 0:
                overhead = comp_b - raw_b
                overhead_records.append((pname, bits, raw_b, comp_b, overhead))

    return total_rows, bad_ratio, bad_gain, overhead_records



def summarize_overhead(overhead_records):
    """Return a dict with distribution counts."""
    dist = {
        1: 0,
        2: 0,
        3: 0,
        ">=4": 0
    }
    for _, _, raw_b, comp_b, overhead in overhead_records:
        if overhead == 1:
            dist[1] += 1
        elif overhead == 2:
            dist[2] += 1
        elif overhead == 3:
            dist[3] += 1
        elif overhead >= 4:
            dist[">=4"] += 1
    return dist



def main():
    print("\n=== Compression Quality Checker WITH Overhead Distribution ===\n")

    files = sorted(glob.glob("gpt_quant_eval_mixed/*compression*.csv"))
    if not files:
        print("No compression CSVs found in gpt_quant_eval_mixed/")
        return

    for csv_path in files:
        print(f"\n--- Analyzing: {csv_path} ---")

        total, bad_ratio, bad_gain, overhead_records = analyze_csv(csv_path)

        print(f"Total rows: {total}")

        # ----------------------
        # Bad Ratio (<1)
        # ----------------------
        if bad_ratio:
            print(f"\n ⚠ Found {len(bad_ratio)} tensors with ratio < 1:")
            for name, bits, ratio in bad_ratio[:20]:
                print(f"   - {name} ({bits}-bit), ratio={ratio:.3f}")
        else:
            print(" ✔ No compression ratio < 1")

        # ----------------------
        # Bad Gain (<0)
        # ----------------------
        if bad_gain:
            print(f"\n ⚠ Found {len(bad_gain)} tensors with negative gain:")
            for name, bits, gain in bad_gain[:20]:
                print(f"   - {name} ({bits}-bit), gain={gain:.2f}%")
        else:
            print(" ✔ No negative gain")

        # ----------------------
        # Overhead: Detailed Info
        # ----------------------
        if overhead_records:
            print(f"\n 🔍 Found {len(overhead_records)} BAD tensors (ratio <1 or gain <0).")
            print("   Showing first 30 with overhead details:\n")

            for name, bits, raw_b, comp_b, overhead in overhead_records[:30]:
                print(
                    f"   - {name} ({bits}-bit): raw={raw_b}B → comp={comp_b}B "
                    f"(overhead={overhead} bytes)"
                )

            if len(overhead_records) > 30:
                print(f"   ... + {len(overhead_records)-30} more")

            # ----------------------
            # Summary distribution
            # ----------------------
            dist = summarize_overhead(overhead_records)

            print("\n 📊 Overhead distribution summary:")
            print(f"   1 byte  : {dist[1]}")
            print(f"   2 bytes : {dist[2]}")
            print(f"   3 bytes : {dist[3]}")
            print(f"   ≥4 bytes: {dist['>=4']}")

        else:
            print("\n ✔ No overhead issues (no bad tensors).")

        # ----------------------
        # Decision summary
        # ----------------------
        if bad_ratio or bad_gain:
            print(" ⇒ ❌ This model likely needs BYPASS mode for some tensors.")
        else:
            print(" ⇒ ✔ This model seems OK.")

    print("\n=== Done ===\n")


if __name__ == "__main__":
    main()
