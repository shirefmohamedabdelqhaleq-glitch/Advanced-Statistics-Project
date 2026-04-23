import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Page Config
st.set_page_config(page_title="Advanced Statistics & Probability", layout="wide")

def plot_dist(dist, crit, test_stat=None, df=None, title="", x_center=None, error_val=None, tail="Two-Tailed (≠)"):
    fig, ax = plt.subplots(figsize=(13, 5))

    y = None  # حماية من الخطأ

    # --- 1. Confidence Interval ---
    if x_center is not None and error_val is not None:
        low, high = x_center - error_val, x_center + error_val
        std_plot = error_val / (crit if crit != 0 else 1.96)

        x = np.linspace(x_center - 4*std_plot, x_center + 4*std_plot, 1000)

        if dist == "Z":
            y = stats.norm.pdf(x, loc=x_center, scale=std_plot)
        else:
            y = stats.t.pdf(x, df, loc=x_center, scale=std_plot)

        ax.plot(x, y, linewidth=3)
        ax.fill_between(x, y, where=(x >= low) & (x <= high), alpha=0.2)

        ax.axvline(low, linestyle='--')
        ax.axvline(high, linestyle='--')

        ax.text(low, max(y)*1.02, f"{low:.2f}", ha='center')
        ax.text(high, max(y)*1.02, f"{high:.2f}", ha='center')

    # --- 2. Hypothesis Testing ---
    elif test_stat is not None:
        if dist == "Chi":
            upper = max(stats.chi2.ppf(0.99, df), test_stat * 1.2)
            x = np.linspace(0, upper, 1000)
            y = stats.chi2.pdf(x, df)
        else:
            limit = max(abs(test_stat), abs(crit if isinstance(crit, float) else max(crit)), 4) + 1
            x = np.linspace(-limit, limit, 1000)

            if dist == "Z":
                y = stats.norm.pdf(x)
            else:
                y = stats.t.pdf(x, df)

        ax.plot(x, y, linewidth=3)

        if dist != "Chi":
            if "Two" in tail:
                ax.fill_between(x, y, where=(x < -crit) | (x > crit), alpha=0.4)
                ax.axvline(crit, linestyle='--')
                ax.axvline(-crit, linestyle='--')
            elif "Right" in tail:
                ax.fill_between(x, y, where=(x > crit), alpha=0.4)
                ax.axvline(crit, linestyle='--')
            elif "Left" in tail:
                ax.fill_between(x, y, where=(x < -crit), alpha=0.4)
                ax.axvline(-crit, linestyle='--')

        ax.axvline(test_stat, linewidth=3)
        ax.text(test_stat, max(y)*1.1, f"TS: {test_stat:.2f}", ha='center')

    # --- 3. Chi-Square Interval (FIX 🔥) ---
    elif dist == "Chi" and isinstance(crit, list):
        lower, upper = crit

        x = np.linspace(0, upper * 1.5, 1000)
        y = stats.chi2.pdf(x, df)

        ax.plot(x, y, linewidth=3)

        ax.fill_between(x, y, where=(x >= lower) & (x <= upper), alpha=0.2)

        ax.axvline(lower, linestyle='--')
        ax.axvline(upper, linestyle='--')

        ax.text(lower, max(y)*1.02, f"{lower:.2f}", ha='center')
        ax.text(upper, max(y)*1.02, f"{upper:.2f}", ha='center')

    # --- حماية نهائية ---
    if y is None:
        return

    # --- الشكل ---
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_ylim(0, max(y)*1.25)

    st.pyplot(fig)
# --- بقية الكود كما هو بدون تعديل في المنطق ---

def handle_raw_data(key_suffix):
    raw_input = st.text_area("Enter data separated by commas (e.g., 10, 12.5, 11, 14):", key=f"raw_{key_suffix}")
    if raw_input:
        try:
            data = [float(x.strip()) for x in raw_input.split(",")]
            return np.array(data)
        except ValueError:
            st.error("Please enter valid numbers separated by commas!")
    return None

def main():
    st.sidebar.title("📊 Control Panel")
    lesson = st.sidebar.selectbox("Select Lecture:", ["L4: Mean Estimation (μ)", "L5: Proportion & Variance", "L6-L7: Hypothesis Testing"])
    alpha = st.sidebar.selectbox("Alpha (α):", [0.01, 0.05, 0.10], index=1)

    if lesson == "L4: Mean Estimation (μ)":
        st.header("📌 Mean Estimation (μ)")
        c1, c2 = st.columns([1, 1.5])
        with c1:
            input_method = st.radio("Data Input Method:", ["Summary Statistics", "Raw Data"])
            if input_method == "Raw Data":
                data = handle_raw_data("L4")
                if data is not None:
                    n = len(data); x_bar = np.mean(data); calc_sd = np.std(data, ddof=1)
                    st.info(f"Calculated: n={n}, x̄={x_bar:.2f}, s={calc_sd:.2f}")
                else: n, x_bar, calc_sd = 1, 0.0, 0.0
            else:
                n = st.number_input("n", value=20); x_bar = st.number_input("Mean (x̄)", value=32.8); calc_sd = None

            sigma_known = st.radio("σ Known?", ["Known", "Unknown"])
            s_type = st.radio("Input Type:", ["SD (s OR σ)", "Variance (s² OR σ²)"])
            if input_method == "Summary Statistics":
                val = st.number_input("Value:", value=4.51)
                sd = val if "SD" in s_type else np.sqrt(val)
            else: sd = calc_sd if calc_sd is not None else 0.0

            if sigma_known == "Known" or n >= 30:
                dist = "Z"; auto_c = round(stats.norm.ppf(1 - alpha/2), 2)
            else:
                dist = "T"; auto_c = round(stats.t.ppf(1 - alpha/2, n-1), 3)
            crit = st.number_input(f"{dist} Critical Value:", value=float(auto_c))

        with c2:
            if n > 0:
                error = crit * (sd / np.sqrt(n))
                st.write(f"**Margin of Error (E):** {error:.4f}")
                st.success(f"**Confidence Interval:** \n### [{x_bar-error:.2f} < μ < {x_bar+error:.2f}]")
                plot_dist(dist, crit, df=n-1, x_center=x_bar, error_val=error)

    elif lesson == "L5: Proportion & Variance":
        tab1, tab2 = st.tabs(["🎯 Proportion (P)", "⚖️ Variance (σ²)"])
        with tab1:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                n_p = st.number_input("n", value=100, key="n_p")
                p_input_method = st.radio("Method:", ["Use X and n", "Use p-hat directly"])
                if p_input_method == "Use X and n":
                    x_val = st.number_input("X:", value=60); p_v = x_val / n_p
                else: p_v = st.number_input("p-hat:", value=0.60)
                z_auto = round(stats.norm.ppf(1 - alpha/2), 2)
                z_p = st.number_input("Z Value:", value=z_auto, key="z_p")
            with c2:
                me = z_p * np.sqrt((p_v*(1-p_v))/n_p)
                st.success(f"**Proportion Interval:** \n### [{p_v-me:.4f} < P < {p_v+me:.4f}]")
                plot_dist("Z", z_p, x_center=p_v, error_val=me)

        with tab2:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                v_method = st.radio("Input Method:", ["Summary Statistics", "Raw Data"], key="v_meth")
                if v_method == "Raw Data":
                    data_v = handle_raw_data("L5_v")
                    if data_v is not None:
                        n_v = len(data_v); s2 = np.var(data_v, ddof=1)
                        st.info(f"Calculated: n={n_v}, s²={s2:.4f}")
                    else: n_v, s2 = 10, 0.25
                else:
                    n_v = st.number_input("n", value=10, key="n_v")
                    v_type = st.radio("Input Type:", ["SD (s)", "Variance (s²)"], key="v_type")
                    v_input = st.number_input("Value:", value=0.25, format="%.5f", key="v_input")
                    s2 = v_input**2 if "SD" in v_type else v_input
                
                df = n_v - 1
                cl_a = round(stats.chi2.ppf(alpha/2, df), 3)
                cu_a = round(stats.chi2.ppf(1 - alpha/2, df), 3)
                cl = st.number_input("Chi Lower (L):", value=cl_a)
                cu = st.number_input("Chi Upper (R):", value=cu_a)
            with c2:
                st.success(f"**Variance Interval:** \n### [{(df*s2)/cu:.5f} < σ² < {(df*s2)/cl:.5f}]")
                plot_dist("Chi", [cl, cu], df=df)

    elif lesson == "L6-L7: Hypothesis Testing":
        st.header("🧪 Hypothesis Testing")
        target = st.selectbox("Test for:", ["μ (Mean)", "P (Proportion)", "σ² (Variance)"])
        c1, c2 = st.columns([1, 1.5])
        with c1:
            h0 = st.number_input("H0 Value", value=8.0)
            tail = st.radio("Alternative (H1):", ["Two-Tailed (≠)", "Right-Tailed (>)", "Left-Tailed (<)"])
            test_input_method = st.radio("Data Input Method:", ["Summary Stats", "Raw Data"], key="test_in")
            
            if test_input_method == "Raw Data" and target != "P (Proportion)":
                data_t = handle_raw_data("L6")
                if data_t is not None:
                    n_t = len(data_t); obs = np.mean(data_t) if "Mean" in target else np.var(data_t, ddof=1)
                    input_val = np.std(data_t, ddof=1) if "Mean" in target else np.var(data_t, ddof=1)
                    st_type = "SD (σ/s)" if "Mean" in target else "Var (σ²/s²)"
                else: n_t, obs, input_val, st_type = 32, 8.2, 0.6, "SD (σ/s)"
            else:
                n_t = st.number_input("n", value=32, key="n_t")
                if target == "P (Proportion)":
                    p_method_t = st.radio("Method for p-hat:", ["Input X", "Input p-hat directly"], key="p_meth_t")
                    if p_method_t == "Input X": x_obs = st.number_input("X:", value=20); obs = x_obs / n_t
                    else: obs = st.number_input("p-hat Observed:", value=0.625)
                else:
                    obs = st.number_input("Observed (x̄ or s²):", value=8.2)
                    st_type = st.radio("Input Type:", ["SD (σ/s)", "Var (σ²/s²)"], key="st_t")
                    input_val = st.number_input("SD/Var Value:", value=0.6, key="inv")

            dist = "Z" if (target != "μ (Mean)" or n_t >= 30) else "T"
            if target == "σ² (Variance)": dist = "Chi"
            if dist == "Z": cr_a = round(stats.norm.ppf(1 - (alpha/2 if "Two" in tail else alpha)), 2)
            elif dist == "T": cr_a = round(stats.t.ppf(1 - (alpha/2 if "Two" in tail else alpha), n_t-1), 3)
            else: cr_a = round(stats.chi2.ppf(1 - alpha, n_t-1), 3)
            cr = st.number_input(f"Critical {dist}:", value=float(cr_a))

        with c2:
            if n_t > 0:
                if "Mean" in target:
                    current_sd = input_val if "SD" in st_type else np.sqrt(input_val)
                    stat = (obs - h0) / (current_sd / np.sqrt(n_t))
                elif "Proportion" in target:
                    h0_p = h0 if h0 <= 1 else h0/100
                    stat = (obs - h0_p) / np.sqrt((h0_p*(1-h0_p))/n_t)
                else:
                    current_s2 = input_val**2 if "SD" in st_type else input_val
                    stat = ((n_t-1)*current_s2) / h0

                st.write(f"**Test Statistic:** {stat:.3f}")
                if "Two" in tail: reject = abs(stat) > cr
                elif "Right" in tail: reject = stat > cr
                else: reject = stat < -cr if dist != "Chi" else stat < stats.chi2.ppf(alpha, n_t-1)
                
                if reject: st.error("**Decision: Reject H0 (Significant)**")
                else: st.success("**Decision: Accept H0 (Not Significant)**")
                
                plot_dist(dist, cr if dist != "Chi" else [0, cr], test_stat=stat, df=n_t-1, tail=tail)

if __name__ == "__main__":
    main()
