import streamlit as st
import numpy as np
from algorithm.GeneticAlgorithm import GeneticAlgorithmDEAP
from algorithm.GaussianSurface import GaussianSurface
from utils import suggest_ga_params

# --- Streamlit UI ---
st.set_page_config(page_title="Genetic Algorithm Visualizer", layout="wide")
st.title("Genetic Algorithm on Multi-Peak Gaussian Surface")

# --- Layout: Parameters Card ---
with st.form("ga_params_form"):
    st.markdown("### Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Basic Options")
        num_peaks = st.number_input("Number of Peaks", min_value=1, max_value=20, value=5)
        x_min = st.number_input("X Min", value=-40.0)
        x_max = st.number_input("X Max", value=40.0)
        y_min = st.number_input("Y Min", value=-20.0)
        y_max = st.number_input("Y Max", value=20.0)
        x_step = st.number_input("X Step Size", min_value=0.01, value=0.5)
        y_step = st.number_input("Y Step Size", min_value=0.01, value=0.3)
        sigma = st.number_input("Sigma (Peak Width)", min_value=0.1, value=6.0)
        amp_min = st.number_input("Amplitude Min", min_value=0.01, value=1.0)
        amp_max = st.number_input("Amplitude Max", min_value=0.01, value=3.0)
    with col2:
        st.markdown("#### Optional / Advanced Options")
        cxpb = st.slider("Crossover Probability", min_value=0.0, max_value=1.0, value=0.5)
        mutpb = st.slider("Mutation Probability", min_value=0.0, max_value=1.0, value=0.2)
        seed = st.number_input("Random Seed", min_value=0, value=42)
        user_population_size = st.number_input("Population Size (leave 0 for auto)", min_value=0, max_value=1000, value=0)
        user_generations = st.number_input("Number of Generations (leave 0 for auto)", min_value=0, max_value=1000, value=0)
        early_stopping = st.checkbox("Enable Early Stopping (stop if population converges)", value=True)

    # Centered button
    colb1, colb2, colb3 = st.columns([1,2,1])
    with colb2:
        submitted = st.form_submit_button("Run Genetic Algorithm", use_container_width=True)

# --- Results and Animations ---
if submitted:
    xy_bounds = [(x_min, x_max), (y_min, y_max)]
    step_sizes = [x_step, y_step]
    amplitude_bounds = (amp_min, amp_max)

    surface = GaussianSurface(
        num_peaks=num_peaks,
        xy_bounds=xy_bounds,
        amplitude_bounds=amplitude_bounds,
        sigma=sigma,
        seed=seed
    )

    x = np.linspace(xy_bounds[0][0], xy_bounds[0][1], 400)
    y = np.linspace(xy_bounds[1][0], xy_bounds[1][1], 400)
    X, Y = np.meshgrid(x, y)
    Z = surface.evaluate(X, Y)

    # Determine population size and generations
    auto_population_size, auto_generations = suggest_ga_params(xy_bounds, step_sizes)

    population_size = user_population_size if user_population_size > 0 else auto_population_size
    generations = user_generations if user_generations > 0 else auto_generations

    ga = GeneticAlgorithmDEAP(
        surface=surface,
        bounds=xy_bounds,
        step_sizes=step_sizes,
        population_size=population_size,
        generations=generations,
        cxpb=cxpb,
        mutpb=mutpb,
        early_stopping=early_stopping,
        seed=seed
    )
    best = ga.run()

    # Results card in the center
    st.markdown("---")
    colr1, colr2, colr3 = st.columns([1, 2, 1])
    with colr2:
        st.markdown("### Results")
        card1, card2 = st.columns(2)
        with card1:
            st.success("**Best Solution**")
            st.markdown(f"<div style='font-size:1.2em'><b>{best}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#999'><b>Function value:</b> {surface.evaluate_point(best[0], best[1]):.4f}</span>", unsafe_allow_html=True)
        with card2:
            st.success("**GA Parameters Used**")
            st.markdown(f"""
            <ul style='font-size:1.1em'>
                <li><b>Population size:</b> {population_size}</li>
                <li><b>Max generations:</b> {generations}</li>
                <li><b>Converged in:</b> {ga.converged_generation} generations</li>
            </ul>
            """, unsafe_allow_html=True)

    # Animations side by side
    cola, colb = st.columns(2)
    with cola:
        st.subheader("Population Evolution Animation")
        pop_buf = ga.animate_population(X, Y, Z)
        st.image(pop_buf, caption="Population Evolution", use_container_width=True)
    with colb:
        st.subheader("Fitness Trend Animation")
        fit_buf = ga.animate_fitness_trend()
        st.image(fit_buf, caption="Fitness Trend", use_container_width=True)