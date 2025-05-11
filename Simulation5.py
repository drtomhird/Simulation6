import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(layout="wide")
st.title("Market Simulation Tool")
st.write("\u2705 App loaded into memory")

np.random.seed(42)

# --- Shared Inputs ---
st.markdown("### Shared Simulation Settings")
col1, col2, col3 = st.columns(3)
with col1:
    cv = st.number_input("Choose Coefficient of Variation", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
    dist_type = st.selectbox("Choose distribution type", ["Normal", "Truncated Normal", "Uniform"])
with col2:
    num_simulations = st.number_input("Choose # of simulations", min_value=100, value=5000, step=100)
    truncation = st.number_input("Choose truncation (if applicable)", min_value=0.0, max_value=100.0, value=20.0, step=0.5)
with col3:
    merger_synergy = st.number_input("Choose a merger synergy effect (%)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

def simulate_shares(means, sim_iters):
    stds = means * cv
    wins = np.zeros(len(means))
    alpha_count = 0
    beta_gaps = []
    merged_second_count = 0

    firm_A = 0  # First firm
    firm_B = 1  # Second firm

    for _ in range(min(sim_iters, 20000)):
        bids = []
        for m, s in zip(means, stds):
            if dist_type == "Normal":
                bid = np.random.normal(m, s)
            elif dist_type == "Truncated Normal":
                lower = m * (1 - truncation / 100)
                upper = m * (1 + truncation / 100)
                bid = np.clip(np.random.normal(m, s), lower, upper)
            elif dist_type == "Uniform":
                lower = m * (1 - truncation / 100)
                upper = m * (1 + truncation / 100)
                bid = np.random.uniform(lower, upper)
            bids.append(bid)

        sorted_idx = np.argsort(bids)
        wins[sorted_idx[0]] += 1

        if set(sorted_idx[:2]) == {firm_A, firm_B}:
            alpha_count += 1
            second = bids[sorted_idx[1]]
            third = bids[sorted_idx[2]]
            if second != 0:
                beta_gaps.append(100 * (third - second) / second)

        merged_set = {firm_A, firm_B}
        if sorted_idx[1] in merged_set and sorted_idx[0] not in merged_set:
            merged_second_count += 1

    shares = wins / sim_iters
    alpha = alpha_count / sim_iters
    beta = np.mean(beta_gaps) if beta_gaps else 0.0
    merged_second_rate = merged_second_count / sim_iters
    return shares, alpha, beta, stds, beta_gaps, merged_second_rate

mode = st.radio("Choose which simulation to run:", ["MPI from mean costs", "MPI from market shares", "Debug symmetric test"], horizontal=True)

if mode == "MPI from mean costs":
    mean_costs_input = st.text_input("Enter mean costs for each firm (comma-separated)", value="100,100,100")
    if st.button("Run Simulation (from mean costs)"):
        mean_costs = list(map(float, mean_costs_input.split(",")))
        means = np.array(mean_costs)
        shares, alpha, beta, stds, beta_gaps, merged_second_rate = simulate_shares(means, sim_iters=num_simulations)
        alpha_beta = alpha * beta

        results = pd.DataFrame({
            "Firm": [chr(65 + i) for i in range(len(means))],
            "Mean Cost": [f"{m:.0f}" for m in means],
            "Market Share (%)": [f"{s*100:.0f}" for s in shares]
        })
        st.dataframe(results)

        st.markdown(f"**α (A & B lowest)**: `{alpha:.1%}`")
        st.markdown(f"**β (3rd vs 2nd lowest | when A & B lowest)**: `{beta:.1f}%`")
        st.markdown(f"**α × β**: `{alpha_beta:.1f}%`")

        st.markdown(f"**Percentage of tenders the merged entity is 2nd lowest bid**: `{merged_second_rate:.1%}`")
        st.markdown(f"**Merger synergy cost reduction**: `{merger_synergy:.1f}%`")
        merger_synergy_effect = - merged_second_rate * merger_synergy
        st.markdown(f"**Merger synergy effect**: `{merger_synergy_effect:.1f}%`")
        net_mpi = alpha_beta + merger_synergy_effect
        st.markdown(f"**Net MPI**: `{net_mpi:.1f}%`")

        # Plotting code as before
        fig, ax = plt.subplots(figsize=(5, 3))
        x = np.linspace(min(means) - 4 * max(stds), max(means) + 4 * max(stds), 1000)
        for i, mu in enumerate(means):
            sigma = mu * cv
            label = f"Firm {chr(65+i)}"
            if dist_type == "Normal":
                y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            elif dist_type == "Truncated Normal":
                lower = mu * (1 - truncation / 100)
                upper = mu * (1 + truncation / 100)
                y = np.where((x >= lower) & (x <= upper),
                             1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                             0)
            elif dist_type == "Uniform":
                lower = mu * (1 - truncation / 100)
                upper = mu * (1 + truncation / 100)
                y = np.where((x >= lower) & (x <= upper),
                             1 / (upper - lower),
                             0)
            ax.plot(x, y, label=label)
        ax.set_title("Cost Distributions")
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        ax.legend()

        col_dist, col_hist = st.columns(2)
        with col_dist:
            st.pyplot(fig)

        if beta_gaps:
            
            fig_hist, ax_hist = plt.subplots(figsize=(5, 3))
            ax_hist.hist(beta_gaps, bins=20, color="skyblue", edgecolor="black")
            ax_hist.set_title("Histogram of β (3rd vs 2nd lowest when A & B lowest)")
            ax_hist.set_xlabel("% Gap")
            ax_hist.set_ylabel("Frequency")
            with col_hist:
                st.pyplot(fig_hist)

if mode == "MPI from market shares":
    target_shares_input = st.text_input("Enter target market shares (comma-separated %)", value="33.3,33.3,33.3")
    tolerance = st.number_input("Tolerance (% deviation from target)", min_value=0.5, max_value=20.0, value=5.0, step=0.5)
    if st.button("Run Simulation (from market shares)"):
        target_shares = np.array(list(map(float, target_shares_input.split(",")))) / 100
        num_firms = len(target_shares)

        def objective(means):
            shares, _, _, _, _, _ = simulate_shares(np.abs(means), sim_iters=num_simulations)
            rel_error = np.abs((shares - target_shares) / target_shares)
            if np.all(rel_error <= tolerance / 100):
                return 0
            return np.sum(rel_error)

        result = minimize(objective, x0=np.full(num_firms, 100), method="Nelder-Mead", options={"maxiter": 300})

        if result.success:
            means = np.abs(result.x)
            shares, alpha, beta, stds, beta_gaps, merged_second_rate = simulate_shares(means, sim_iters=num_simulations)
            alpha_beta = alpha * beta

            results = pd.DataFrame({
                "Firm": [chr(65 + i) for i in range(num_firms)],
                "Mean Cost": [f"{m:.0f}" for m in means],
                "Market Share (%)": [f"{s*100:.0f}" for s in shares]
            })
            st.dataframe(results)

            st.markdown(f"**α (A & B lowest)**: `{alpha:.1%}`")
            st.markdown(f"**β (3rd vs 2nd lowest | when A & B lowest)**: `{beta:.1f}%`")
            st.markdown(f"**α × β**: `{alpha_beta:.1f}%`")

            st.markdown(f"**Percentage of tenders the merged entity is 2nd lowest bid**: `{merged_second_rate:.1%}`")
            st.markdown(f"**Merger synergy cost reduction**: `{merger_synergy:.1f}%`")
            merger_synergy_effect = - merged_second_rate * merger_synergy
            st.markdown(f"**Merger synergy effect**: `{merger_synergy_effect:.1f}%`")
            net_mpi = alpha_beta + merger_synergy_effect
            st.markdown(f"**Net MPI**: `{net_mpi:.1f}%`")

            fig, ax = plt.subplots(figsize=(5, 3))
            x = np.linspace(min(means) - 4 * max(stds), max(means) + 4 * max(stds), 1000)
            for i, mu in enumerate(means):
                sigma = mu * cv
                label = f"Firm {chr(65+i)}"
                if dist_type == "Normal":
                    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                elif dist_type == "Truncated Normal":
                    lower = mu * (1 - truncation / 100)
                    upper = mu * (1 + truncation / 100)
                    y = np.where((x >= lower) & (x <= upper),
                                 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                                 0)
                elif dist_type == "Uniform":
                    lower = mu * (1 - truncation / 100)
                    upper = mu * (1 + truncation / 100)
                    y = np.where((x >= lower) & (x <= upper),
                                 1 / (upper - lower),
                                 0)
                ax.plot(x, y, label=label)
            ax.set_title("Cost Distributions")
            ax.set_xlabel("Value")
            ax.set_ylabel("Probability Density")
            ax.legend()

            col_dist, col_hist = st.columns(2)
            with col_dist:
                st.pyplot(fig)

            if beta_gaps:
                
                fig_hist, ax_hist = plt.subplots(figsize=(5, 3))
                ax_hist.hist(beta_gaps, bins=20, color="skyblue", edgecolor="black")
                ax_hist.set_title("Histogram of β (3rd vs 2nd lowest when A & B lowest)")
                ax_hist.set_xlabel("% Gap")
                ax_hist.set_ylabel("Frequency")
                with col_hist:
                    st.pyplot(fig_hist)