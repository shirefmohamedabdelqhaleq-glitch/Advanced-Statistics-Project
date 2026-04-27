import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# --- Page Config & Custom Design ---
st.set_page_config(page_title="Advanced Statistics Pro", layout="wide", page_icon="📊")

st.markdown("""
<style>

/* ===== Theme-aware base ===== */
:root {
    --bg: var(--background-color);
    --text: var(--text-color);
    --primary: var(--primary-color);
    --border: rgba(128,128,128,0.2);
}

/* ===== Global ===== */
html, body, [class*="css"] {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    color: var(--text);
}

/* ===== Remove header ===== */
header {
    background: transparent;
}

[data-testid="collapsedControl"] {
    display: block !important;
}

/* ===== Layout spacing ===== */
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background-color: var(--bg);
    border-right: 1px solid var(--border);
}

/* ===== Titles ===== */
h1, h2, h3 {
    font-weight: 600;
    letter-spacing: -0.5px;
}

/* ===== Cards (clean, adaptive) ===== */
.stat-card {
    background: transparent;
    padding: 18px;
    border-radius: 10px;
    border: 1px solid var(--border);
    transition: all 0.2s ease;
}
.stat-card:hover {
    border-color: var(--primary);
}

/* ===== Buttons ===== */
.stButton>button {
    background: var(--primary);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px;
    font-weight: 500;
    transition: 0.2s;
}
.stButton>button:hover {
    opacity: 0.85;
}

/* ===== Inputs ===== */
.stNumberInput input,
.stTextInput input,
.stTextArea textarea {
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
}

/* ===== Selectbox & dropdown ===== */
[data-baseweb="select"] {
    border-radius: 6px !important;
}

/* ===== Tabs (clean style) ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 6px;
    padding: 8px 14px;
    background: transparent;
    border: 1px solid transparent;
}

.stTabs [data-baseweb="tab"]:hover {
    border: 1px solid var(--border);
}

.stTabs [aria-selected="true"] {
    background: var(--primary) !important;
    color: white !important;
}

/* ===== Metrics ===== */
[data-testid="stMetric"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
}

/* ===== Expander ===== */
details {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px;
}

/* ===== Alerts ===== */
.stAlert {
    border-radius: 10px;
    border: 1px solid var(--border);
}

/* ===== Divider ===== */
hr {
    border-color: var(--border);
}

/* ===== Scrollbar (optional but nice) ===== */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: rgba(128,128,128,0.4);
    border-radius: 4px;
}

/* ===== Fix matplotlib bg ===== */
canvas {
    background-color: transparent !important;
}

</style>
""", unsafe_allow_html=True)

def plot_dist(dist, crit, test_stat=None, df=None, title="", x_center=None, error_val=None, tail="Two-Tailed (≠)"):
    fig, ax = plt.subplots(figsize=(13, 5))
    y = None  

    try:
        # حماية df
        if df is not None and df <= 0:
            df = 1

        if x_center is not None and error_val is not None:
            low, high = x_center - error_val, x_center + error_val
            std_plot = error_val / (crit if crit != 0 else 1.96)

            if std_plot == 0 or np.isnan(std_plot) or np.isinf(std_plot):
                std_plot = 1

            x = np.linspace(x_center - 4*std_plot, x_center + 4*std_plot, 1000)

            if dist == "Z":
                y = stats.norm.pdf(x, loc=x_center, scale=std_plot)
            else:
                y = stats.t.pdf(x, df, loc=x_center, scale=std_plot)

            y = np.nan_to_num(y)

            ax.plot(x, y, linewidth=3)
            ax.fill_between(x, y, where=(x >= low) & (x <= high), alpha=0.2)
            ax.axvline(low, linestyle='--', color='red')
            ax.axvline(high, linestyle='--', color='red')

            ymax = np.max(y) if len(y) > 0 else 1
            ax.text(low, ymax*1.05, f"{low:.2f}", ha='center', color='red')
            ax.text(high, ymax*1.05, f"{high:.2f}", ha='center', color='red')

        elif test_stat is not None:
            if dist == "Chi":
                upper = max(stats.chi2.ppf(0.99, df), test_stat * 1.2 if test_stat else 1)
                x = np.linspace(0, upper, 1000)
                y = stats.chi2.pdf(x, df)
            else:
                crit_val = crit if isinstance(crit, float) else max(crit)
                limit = max(abs(test_stat), abs(crit_val), 4) + 1
                x = np.linspace(-limit, limit, 1000)

                if dist == "Z":
                    y = stats.norm.pdf(x)
                else:
                    y = stats.t.pdf(x, df)

            y = np.nan_to_num(y)

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

            ax.axvline(test_stat, linewidth=3, color='red')

            ymax = np.max(y) if len(y) > 0 else 1
            ax.text(test_stat, ymax*1.1, f"TS: {test_stat:.2f}", ha='center', color='red')

        elif dist == "Chi" and isinstance(crit, list):
            lower, upper = crit
            x = np.linspace(0, upper * 1.5 if upper > 0 else 1, 1000)
            y = stats.chi2.pdf(x, df)

            y = np.nan_to_num(y)

            ax.plot(x, y, linewidth=3)
            ax.fill_between(x, y, where=(x >= lower) & (x <= upper), alpha=0.2)
            ax.axvline(lower, linestyle='--')
            ax.axvline(upper, linestyle='--')

            ymax = np.max(y) if len(y) > 0 else 1
            ax.text(lower, ymax*1.05, f"{lower:.2f}", ha='center')
            ax.text(upper, ymax*1.05, f"{upper:.2f}", ha='center')

        if y is None:
            return

        y = np.nan_to_num(y)

        ymax = np.max(y) if len(y) > 0 else 1
        if ymax == 0 or np.isnan(ymax) or np.isinf(ymax):
            ymax = 1

        ax.set_ylim(0, ymax * 1.25)

        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Plot Error: {e}")

def handle_raw_data(key_suffix):
    raw_input = st.text_area("Enter data (commas):", placeholder="10, 12.5, 11...", key=f"raw_{key_suffix}")
    if raw_input:
        try:
            data = [float(x.strip()) for x in raw_input.split(",")]
            return np.array(data)
        except ValueError:
            st.error("Invalid numbers!")
    return None

def main():
    
    # --- Sidebar Styling ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
        st.title("Settings")
        lesson = st.selectbox("Select Lecture:", ["L4: Mean Estimation (μ)", "L5: Proportion & Variance", "L6-L7: Hypothesis Testing"])
        alpha = st.select_slider("Significance Level (α):", options=[0.01, 0.05, 0.10], value=0.05)
        st.divider()
        st.caption("Advanced Statistics Tool v2.0")

    # --- Main Content ---
    if lesson == "L4: Mean Estimation (μ)":
        st.title("📌 Mean Estimation (μ)")
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            with st.expander("📥 Data Input", expanded=True):
                input_method = st.radio("Method:", ["Summary Statistics", "Raw Data"])
                if input_method == "Raw Data":
                    data = handle_raw_data("L4")
                    if data is not None:
                        n, x_bar, calc_sd = len(data), np.mean(data), np.std(data, ddof=1)
                        st.info(f"n={n} | x̄={x_bar:.2f} | s={calc_sd:.2f}")
                    else: n, x_bar, calc_sd = 1, 0.0, 0.0
                else:
                    n = st.number_input("Sample Size (n)", value=20)
                    x_bar = st.number_input("Sample Mean (x̄)", value=32.8)
                    calc_sd = None

                sigma_known = st.radio("Standard Deviation:", ["Known", "Unknown"], horizontal=True)
                s_type = st.radio("Input Type:", ["SD (s OR σ)", "Variance (s² OR σ²)"], horizontal=True)
                
                if input_method == "Summary Statistics":
                    val = st.number_input("Value:", value=4.51)
                    sd = val if "SD" in s_type else np.sqrt(val)
                else: sd = calc_sd if calc_sd is not None else 0.0

                dist = "Z" if (sigma_known == "Known" or n >= 30) else "T"
                auto_c = round(stats.norm.ppf(1 - alpha/2), 2) if dist == "Z" else round(stats.t.ppf(1 - alpha/2, n-1), 3)
                crit = st.number_input(f"{dist} Critical Value:", value=float(auto_c))

        with col2:
            if n > 0:
                error = crit * (sd / np.sqrt(n))
                st.subheader("Results")
                res_c1, res_c2 = st.columns(2)
                res_c1.metric("Margin of Error (E)", f"{error:.4f}")
                res_c2.metric("Confidence Level", f"{(1-alpha)*100}%")
                
                st.success(f"**Confidence Interval:** \n### {x_bar-error:.4f} < μ < {x_bar+error:.4f}")
                plot_dist(dist, crit, df=n-1, x_center=x_bar, error_val=error)

    elif lesson == "L5: Proportion & Variance":
        st.title("🎯 Proportion & Variance")
        tab1, tab2 = st.tabs(["🎯 Population Proportion (P)", "⚖️ Population Variance (σ²)"])
        
        with tab1:
            c1, c2 = st.columns([1, 2], gap="large")
            with c1:
                with st.expander("Parameters", expanded=True):
                    n_p = st.number_input("Sample Size (n)", value=100, key="n_p")
                    p_input_method = st.radio("Input Method:", ["Use X and n", "Use p-hat"], horizontal=True)
                    if p_input_method == "Use X and n":
                        x_val = st.number_input("Successes (X):", value=60); p_v = x_val / n_p
                    else: p_v = st.number_input("p-hat:", value=0.60)
                    z_p = st.number_input("Z Critical Value:", value=round(stats.norm.ppf(1 - alpha/2), 2), key="z_p")
            with c2:
                me = z_p * np.sqrt((p_v*(1-p_v))/n_p)
                st.metric("p-hat", f"{p_v:.4f}")
                st.success(f"**Proportion Interval:** \n### {p_v-me:.4f} < P < {p_v+me:.4f}")
                plot_dist("Z", z_p, x_center=p_v, error_val=me)

        with tab2:
            c1, c2 = st.columns([1, 2], gap="large")
            with c1:
                with st.expander("Variance Data", expanded=True):
                    v_method = st.radio("Input Method:", ["Summary Statistics", "Raw Data"], key="v_meth")
                    if v_method == "Raw Data":
                        data_v = handle_raw_data("L5_v")
                        if data_v is not None:
                            n_v, s2 = len(data_v), np.var(data_v, ddof=1)
                        else: n_v, s2 = 10, 0.25
                    else:
                        n_v = st.number_input("n", value=10, key="n_v")
                        v_type = st.radio("Type:", ["SD (s)", "Variance (s²)"], key="v_type", horizontal=True)
                        v_input = st.number_input("Value:", value=0.25, format="%.5f", key="v_input")
                        s2 = v_input**2 if "SD" in v_type else v_input
                    
                    df = n_v - 1
                    cl = st.number_input("Chi Lower (L):", value=round(stats.chi2.ppf(alpha/2, df), 3))
                    cu = st.number_input("Chi Upper (R):", value=round(stats.chi2.ppf(1 - alpha/2, df), 3))
            with c2:
                st.success(f"**Variance Interval:** \n### {(df*s2)/cu:.5f} < σ² < {(df*s2)/cl:.5f}")
                plot_dist("Chi", [cl, cu], df=df)

    elif lesson == "L6-L7: Hypothesis Testing":
        st.title("🧪 Hypothesis Testing")
        target = st.selectbox("What are you testing for?", ["μ (Mean)", "P (Proportion)", "σ² (Variance)"])
        
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            with st.expander("Test Configuration", expanded=True):
                h0 = st.number_input("H0 (Null Hypothesis) Value", value=8.0)
                tail = st.selectbox("Alternative Hypothesis (H1):", ["Two-Tailed (≠)", "Right-Tailed (>)", "Left-Tailed (<)"])
                test_input_method = st.radio("Input Method:", ["Summary Stats", "Raw Data"], key="test_in")
                
                if test_input_method == "Raw Data" and target != "P (Proportion)":
                    data_t = handle_raw_data("L6")
                    if data_t is not None:
                        n_t = len(data_t)
                        obs = np.mean(data_t) if "Mean" in target else np.var(data_t, ddof=1)
                        input_val = np.std(data_t, ddof=1) if "Mean" in target else np.var(data_t, ddof=1)
                        st_type = "SD (σ/s)" if "Mean" in target else "Var (σ²/s²)"
                    else: n_t, obs, input_val, st_type = 32, 8.2, 0.6, "SD (σ/s)"
                else:
                    n_t = st.number_input("Sample Size (n)", value=32, key="n_t")
                    if target == "P (Proportion)":
                        p_method_t = st.radio("p-hat Method:", ["Input X", "Direct"], key="p_meth_t", horizontal=True)
                        obs = st.number_input("X:", value=20)/n_t if p_method_t == "Input X" else st.number_input("p-hat Observed:", value=0.625)
                    else:
                        obs = st.number_input("Observed (x̄ or s²):", value=8.2)
                        st_type = st.radio("Type:", ["SD (σ/s)", "Var (σ²/s²)"], key="st_t", horizontal=True)
                        input_val = st.number_input("SD/Var Value:", value=0.6)

                dist = "Z" if (target != "μ (Mean)" or n_t >= 30) else "T"
                if target == "σ² (Variance)": dist = "Chi"
                
                if dist == "Z": cr_a = stats.norm.ppf(1 - (alpha/2 if "Two" in tail else alpha))
                elif dist == "T": cr_a = stats.t.ppf(1 - (alpha/2 if "Two" in tail else alpha), n_t-1)
                else: cr_a = stats.chi2.ppf(1 - alpha, n_t-1)
                cr = st.number_input(f"Critical {dist}:", value=float(round(cr_a, 3)))

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

                st.subheader("Test Results")
                st.metric("Test Statistic (TS)", f"{stat:.4f}")
                
                if "Two" in tail: reject = abs(stat) > cr
                elif "Right" in tail: reject = stat > cr
                else: reject = stat < -cr if dist != "Chi" else stat < stats.chi2.ppf(alpha, n_t-1)
                
                if reject: st.error("### Decision: Reject H0 (Significant) ")
                else: st.success("### Decision: Accept H0 (Not Significant) ")
                
                plot_dist(dist, cr if dist != "Chi" else [0, cr], test_stat=stat, df=n_t-1, tail=tail)

if __name__ == "__main__":
    main()
