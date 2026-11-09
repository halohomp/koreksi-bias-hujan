# ==============================================================================
# SCRIPT KOREKSI BIAS HUJAN HARIAN & BULANAN (GAMMA QM & QDM)
# Skenario: MIROC6 Historis & Proyeksi SSP245 vs Observasi CHIRPS
# ==============================================================================

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# 1. Hubungkan ke Google Drive & Set Direktori Kerja
try:
    from google.colab import drive
    drive.mount("/content/drive")
    WORK_DIR = Path("/content/drive/MyDrive/Koreksi Bias Hujan")
except ImportError:
    WORK_DIR = Path(".")

if not WORK_DIR.exists():
    raise FileNotFoundError(f"Folder Google Drive tidak ditemukan di: {WORK_DIR}")

os.chdir(WORK_DIR)
print(f"Direktori kerja aktif: {os.getcwd()}")

# Buat folder output otomatis di dalam folder 'Koreksi Bias Hujan'
OUT_DIR = WORK_DIR / "Hasil_Koreksi_Gamma_QM_QDM"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. KONFIGURASI NAMA FILE (SESUAI GOOGLE DRIVE ANDA)
# ==============================================================================
CHIRPS_FILES = [
    "Tanah Pasir_Bulanan_CHIRPS_1995_2004.csv",
    "Tanah Pasir_Bulanan_CHIRPS_2005_2014.csv",
]

FILE_MOD_HIST = "Hujan_Tanah_Pasir_1995_2014_historis_miroc.csv"

# Deteksi otomatis file proyeksi SSP245 di dalam folder
FILES_PROJ = {}
for f in sorted(WORK_DIR.glob("Hujan_Tanah_Pasir_SSP245_*.csv")):
    label = f.stem.replace("Hujan_Tanah_Pasir_", "")
    FILES_PROJ[label] = f.name

print("\nFile Proyeksi SSP245 yang terdeteksi:")
for lbl, fname in FILES_PROJ.items():
    print(f"  - {lbl} : {fname}")


# ==============================================================================
# 3. FUNGSI PERSIAPAN DATA (CHIRPS & MIROC)
# ==============================================================================
def load_and_merge_chirps(file_list):
    """Menggabungkan data bulanan CHIRPS 1995-2004 dan 2005-2014."""
    dfs = []
    for f in file_list:
        f_path = WORK_DIR / f
        if not f_path.exists():
            raise FileNotFoundError(f"File CHIRPS tidak ditemukan: {f}")
        dfs.append(pd.read_csv(f_path))

    df_obs = pd.concat(dfs, ignore_index=True)
    df_obs["YearMonth"] = pd.to_datetime(df_obs["month"]).dt.to_period("M")
    df_obs = df_obs.sort_values("YearMonth").reset_index(drop=True)
    df_obs["Month"] = df_obs["YearMonth"].dt.month
    df_obs.rename(columns={"precipitation": "Curah_Hujan_mm"}, inplace=True)
    return df_obs[["month", "YearMonth", "Month", "Curah_Hujan_mm"]]


def load_and_aggregate_miroc(filepath):
    """Membaca data harian MIROC dan menghitung akumulasi bulanan (mm/bulan)."""
    f_path = WORK_DIR / filepath
    df_daily = pd.read_csv(f_path)
    df_daily["Tanggal"] = pd.to_datetime(df_daily["Tanggal"])
    df_daily = df_daily.sort_values("Tanggal").reset_index(drop=True)
    df_daily["YearMonth"] = df_daily["Tanggal"].dt.to_period("M")
    df_daily["Month"] = df_daily["Tanggal"].dt.month

    df_mon = (
        df_daily.groupby("YearMonth", as_index=False)["Curah_Hujan_mm"]
        .sum()
        .sort_values("YearMonth")
        .reset_index(drop=True)
    )
    df_mon["month"] = df_mon["YearMonth"].astype(str)
    df_mon["Month"] = df_mon["YearMonth"].dt.month
    return df_daily, df_mon[["month", "YearMonth", "Month", "Curah_Hujan_mm"]]


# ==============================================================================
# 4. FUNGSI INTI: DISTRIBUSI GAMMA, QM, & QDM
# ==============================================================================
def fit_gamma_params(data):
    """Fitting parameter Gamma (shape=alpha, loc=0, scale=beta) dengan MLE + fallback MoM."""
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    try:
        alpha, loc, beta = stats.gamma.fit(arr, floc=0, method="MLE")
        if not np.all(np.isfinite([alpha, beta])) or alpha <= 0 or beta <= 0:
            raise ValueError("MLE Gamma tidak konvergen")
        return alpha, beta
    except Exception:
        mean_val = np.mean(arr)
        var_val = np.var(arr, ddof=1)
        alpha = (mean_val ** 2) / max(var_val, 1e-6)
        beta = max(var_val, 1e-6) / mean_val
        return alpha, beta


def QM_gamma(target_train, input_train, input_eval):
    """Quantile Mapping (QM) Gamma untuk periode Historis (1995-2014)."""
    alpha_t, beta_t = fit_gamma_params(target_train)
    alpha_i, beta_i = fit_gamma_params(input_train)

    tau = stats.gamma.cdf(input_eval, alpha_i, loc=0, scale=beta_i)
    tau = np.clip(tau, 0.001, 0.999)
    return stats.gamma.ppf(tau, alpha_t, loc=0, scale=beta_t)


def QDM_gamma(target_train, input_train, input_eval):
    """Quantile Delta Mapping (QDM) Gamma Multiplikatif untuk Proyeksi."""
    alpha_t, beta_t = fit_gamma_params(target_train)
    alpha_i, beta_i = fit_gamma_params(input_train)
    alpha_e, beta_e = fit_gamma_params(input_eval)

    tau = stats.gamma.cdf(input_eval, alpha_e, loc=0, scale=beta_e)
    tau = np.clip(tau, 0.001, 0.999)

    x_qm = stats.gamma.ppf(tau, alpha_t, loc=0, scale=beta_t)
    x_hist_analog = stats.gamma.ppf(tau, alpha_i, loc=0, scale=beta_i)

    # Faktor perubahan relatif (rasio multiplikatif)
    delta_ratio = input_eval / np.maximum(x_hist_analog, 1e-3)
    return x_qm * delta_ratio


def apply_monthly_correction(df_obs_mon, df_hist_mon, df_eval_mon, method="QM"):
    """Menjalankan koreksi bias dikelompokkan per bulan kalender (1..12)."""
    df_out = df_eval_mon.copy()
    df_out.rename(columns={"Curah_Hujan_mm": "MIROC_Raw_mm"}, inplace=True)
    df_out["MIROC_Corrected_mm"] = np.nan

    for m in range(1, 13):
        obs_m = df_obs_mon.loc[df_obs_mon["Month"] == m, "Curah_Hujan_mm"].values
        hist_m = df_hist_mon.loc[df_hist_mon["Month"] == m, "Curah_Hujan_mm"].values
        mask_m = df_out["Month"] == m
        eval_m = df_out.loc[mask_m, "MIROC_Raw_mm"].values

        if method.upper() == "QM":
            corr_m = QM_gamma(obs_m, hist_m, eval_m)
        else:
            corr_m = QDM_gamma(obs_m, hist_m, eval_m)

        df_out.loc[mask_m, "MIROC_Corrected_mm"] = np.round(corr_m, 2)

    return df_out


def disaggregate_monthly_to_daily(df_daily, df_mon_corr):
    """Mendisagregasi hasil koreksi bulanan kembali ke skala harian secara proporsional."""
    df_d = df_daily.copy()
    ratio_series = (
        df_mon_corr.set_index("YearMonth")["MIROC_Corrected_mm"]
        / df_mon_corr.set_index("YearMonth")["MIROC_Raw_mm"].replace(0, np.nan)
    )
    df_d["Faktor_Skala"] = df_d["YearMonth"].map(ratio_series).fillna(1.0)
    df_d["MIROC_Raw_mm"] = df_d["Curah_Hujan_mm"]
    df_d["MIROC_Corrected_mm"] = (df_d["MIROC_Raw_mm"] * df_d["Faktor_Skala"]).round(4)
    return df_d


# ==============================================================================
# 5. EKSEKUSI UTAMA & EKSPOR FILE CSV
# ==============================================================================
if __name__ == "__main__":
    print("\n[1/4] Menggabungkan data Observasi CHIRPS (1995-2014)...")
    df_obs_mon = load_and_merge_chirps(CHIRPS_FILES)
    df_obs_mon[["month", "Curah_Hujan_mm"]].to_csv(OUT_DIR / "CHIRPS_Bulanan_Gabungan_1995_2014.csv", index=False)

    print("[2/4] Memproses Historis MIROC (1995-2014) dengan Gamma QM...")
    hist_daily, hist_mon = load_and_aggregate_miroc(FILE_MOD_HIST)
    hist_mon_qm = apply_monthly_correction(df_obs_mon, hist_mon, hist_mon, method="QM")
    hist_daily_qm = disaggregate_monthly_to_daily(hist_daily, hist_mon_qm)

    # Tabel Evaluasi Historis
    obs_v, raw_v, cor_v = df_obs_mon["Curah_Hujan_mm"].values, hist_mon_qm["MIROC_Raw_mm"].values, hist_mon_qm["MIROC_Corrected_mm"].values
    eval_table = pd.DataFrame(
        {
            "Observasi CHIRPS": [np.mean(obs_v), np.std(obs_v, ddof=1), np.max(obs_v), 0.0, 0.0, 0.0],
            "MIROC Historis Raw": [np.mean(raw_v), np.std(raw_v, ddof=1), np.max(raw_v), 100.0 * np.sum(raw_v - obs_v) / np.sum(obs_v), mean_absolute_error(obs_v, raw_v), np.sqrt(mean_squared_error(obs_v, raw_v))],
            "MIROC Historis Gamma QM": [np.mean(cor_v), np.std(cor_v, ddof=1), np.max(cor_v), 100.0 * np.sum(cor_v - obs_v) / np.sum(obs_v), mean_absolute_error(obs_v, cor_v), np.sqrt(mean_squared_error(obs_v, cor_v))],
        },
        index=["Mean (mm/bln)", "Std Dev (mm/bln)", "Max (mm/bln)", "PBIAS (%)", "MAE (mm/bln)", "RMSE (mm/bln)"],
    )
    print("\n" + "=" * 68)
    print("EVALUASI KOREKSI BIAS PERIODE HISTORIS (1995-2014)")
    print("=" * 68)
    print(eval_table.round(2).to_string())
    print("=" * 68)

    compiled_monthly_list = [pd.DataFrame({
        "Bulan_Tahun": hist_mon_qm["month"], "Periode": "Historis_1995_2014", "Metode": "Gamma_QM",
        "Observasi_CHIRPS_mm": df_obs_mon["Curah_Hujan_mm"].round(2),
        "MIROC_Raw_mm": hist_mon_qm["MIROC_Raw_mm"].round(2),
        "MIROC_Corrected_mm": hist_mon_qm["MIROC_Corrected_mm"].round(2),
    })]

    compiled_daily_list = [pd.DataFrame({
        "Tanggal": hist_daily_qm["Tanggal"].dt.strftime("%Y-%m-%d"), "Periode": "Historis_1995_2014", "Metode": "Gamma_QM",
        "MIROC_Raw_mm": hist_daily_qm["MIROC_Raw_mm"].round(4),
        "MIROC_Corrected_mm": hist_daily_qm["MIROC_Corrected_mm"].round(4),
    })]

    proj_daily_only_list = []
    proj_monthly_dict = {}

    print("\n[3/4] Memproses Proyeksi SSP245 (Near & Mid) dengan Gamma QDM...")
    for label, proj_file in FILES_PROJ.items():
        proj_daily, proj_mon = load_and_aggregate_miroc(proj_file)
        proj_mon_qdm = apply_monthly_correction(df_obs_mon, hist_mon, proj_mon, method="QDM")
        proj_daily_qdm = disaggregate_monthly_to_daily(proj_daily, proj_mon_qdm)
        proj_monthly_dict[label] = proj_mon_qdm

        proj_mon_qdm[["month", "MIROC_Raw_mm", "MIROC_Corrected_mm"]].to_csv(OUT_DIR / f"Bulanan_Gamma_QDM_{label}.csv", index=False)
        proj_daily_qdm[["Tanggal", "MIROC_Corrected_mm"]].rename(columns={"MIROC_Corrected_mm": "Curah_Hujan_mm"}).to_csv(OUT_DIR / f"Harian_Gamma_QDM_{label}.csv", index=False)

        compiled_monthly_list.append(pd.DataFrame({
            "Bulan_Tahun": proj_mon_qdm["month"], "Periode": label, "Metode": "Gamma_QDM",
            "Observasi_CHIRPS_mm": np.nan, "MIROC_Raw_mm": proj_mon_qdm["MIROC_Raw_mm"].round(2),
            "MIROC_Corrected_mm": proj_mon_qdm["MIROC_Corrected_mm"].round(2),
        }))

        compiled_daily_list.append(pd.DataFrame({
            "Tanggal": proj_daily_qdm["Tanggal"].dt.strftime("%Y-%m-%d"), "Periode": label, "Metode": "Gamma_QDM",
            "MIROC_Raw_mm": proj_daily_qdm["MIROC_Raw_mm"].round(4), "MIROC_Corrected_mm": proj_daily_qdm["MIROC_Corrected_mm"].round(4),
        }))

        proj_daily_only_list.append(pd.DataFrame({
            "Tanggal": proj_daily_qdm["Tanggal"].dt.strftime("%Y-%m-%d"),
            "Curah_Hujan_mm": proj_daily_qdm["MIROC_Corrected_mm"].round(4),
        }))
        print(f"  -> Selesai memproses: {label}")

    print("\n[4/4] Menyimpan File Kompilasi CSV & Grafik Evaluasi...")
    pd.concat(compiled_monthly_list, ignore_index=True).to_csv(OUT_DIR / "Kompilasi_Bulanan_Koreksi_Bias_Gamma_1995_2060.csv", index=False)
    pd.concat(compiled_daily_list, ignore_index=True).to_csv(OUT_DIR / "Kompilasi_Harian_Koreksi_Bias_Gamma_1995_2060.csv", index=False)
    pd.concat(proj_daily_only_list, ignore_index=True).sort_values("Tanggal").to_csv(OUT_DIR / "Hujan_Tanah_Pasir_SSP245_Corrected_Harian_2021_2060.csv", index=False)

    # Plot Grafik Klimatologi
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    months_lbl = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

    axes[0].plot(range(1, 13), df_obs_mon.groupby("Month")["Curah_Hujan_mm"].mean(), "o-", lw=2.5, label="CHIRPS (Obs)")
    axes[0].plot(range(1, 13), hist_mon_qm.groupby("Month")["MIROC_Raw_mm"].mean(), "s--", lw=2, label="MIROC Hist (Raw)")
    axes[0].plot(range(1, 13), hist_mon_qm.groupby("Month")["MIROC_Corrected_mm"].mean(), "^-.", lw=2, label="MIROC Hist (Gamma QM)")
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(months_lbl)
    axes[0].set_title("Klimatologi Bulanan Historis (1995–2014)")
    axes[0].set_ylabel("Curah Hujan (mm/bulan)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, 13), df_obs_mon.groupby("Month")["Curah_Hujan_mm"].mean(), "o-", lw=2.2, label="Baseline CHIRPS (1995–2014)")
    for label, df_p in proj_monthly_dict.items():
        axes[1].plot(range(1, 13), df_p.groupby("Month")["MIROC_Corrected_mm"].mean(), "s-", lw=2, label=f"{label} (Gamma QDM)")
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xticklabels(months_lbl)
    axes[1].set_title("Klimatologi Proyeksi SSP245 (2021–2060)")
    axes[1].set_ylabel("Curah Hujan (mm/bulan)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "Grafik_Evaluasi_Gamma_QM_QDM.png", dpi=300)
    plt.show()

    print(f"\nSELESAI! Seluruh file tersimpan rapi di Google Drive dalam folder:\n  {OUT_DIR}")