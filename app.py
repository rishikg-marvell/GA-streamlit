import streamlit as st
import pandas as pd
from algorithm.GA import HBERGeneticAlgorithm
from utils import suggest_ga_params

st.set_page_config(page_title="GA HBER Optimizer", layout="wide")
st.title("Genetic Algorithm for PRE, MAIN, POST Optimization")

# --- File Upload ---
st.markdown("### 1. Upload your CSV file")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
df_new = None

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("First 5 rows of your data:")
    st.dataframe(df.head())

    # --- Grouping operation ---
    st.markdown("#### Grouping by PRE, MAIN, POST and aggregating avg_HBER and worst_HBER")
    if all(col in df.columns for col in ['PRE', 'MAIN', 'POST', 'HBER']):
        df_new = df.groupby(['PRE', 'MAIN', 'POST']).agg(
            avg_HBER=('HBER', 'mean'),
            worst_HBER=('HBER', 'max')
        ).reset_index()
        st.dataframe(df_new.head())
    else:
        st.error("CSV must contain columns: PRE, MAIN, POST, HBER")

if df_new is not None:
    st.markdown("### 2. Set Genetic Algorithm Parameters")

    # --- Parameter Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        pre_min = st.number_input("PRE min", value=int(df_new['PRE'].min()))
        pre_max = st.number_input("PRE max", value=int(df_new['PRE'].max()))
        main_min = st.number_input("MAIN min", value=int(df_new['MAIN'].min()))
        main_max = st.number_input("MAIN max", value=int(df_new['MAIN'].max()))
        post_min = st.number_input("POST min", value=int(df_new['POST'].min()))
        post_max = st.number_input("POST max", value=int(df_new['POST'].max()))
    with col2:
        pre_step = st.number_input("PRE step size", value=50)
        main_step = st.number_input("MAIN step size", value=50)
        post_step = st.number_input("POST step size", value=50)
        
        st.markdown("**Set weights for fitness function:**")
        w1 = st.slider("Weight for avg_HBER", 0.0, 1.0, 0.7, step=0.01)
        w2 = 1.0 - w1
        st.markdown(f"Weight for worst_HBER: **{w2:.2f}** (sum always 1)")

        seed = st.number_input("Random Seed", min_value=0, value=42)
        user_population_size = st.number_input("Population Size (0 for auto)", min_value=0, value=0)
        user_generations = st.number_input("Generations (0 for auto)", min_value=0, value=0)
        cxpb = st.slider("Crossover Probability", 0.0, 1.0, 0.5)
        mutpb = st.slider("Mutation Probability", 0.0, 1.0, 0.2)
        early_stopping = st.checkbox("Enable Early Stopping", value=True)

    bounds = {'PRE': (pre_min, pre_max), 'MAIN': (main_min, main_max), 'POST': (post_min, post_max)}
    step_sizes = {'PRE': pre_step, 'MAIN': main_step, 'POST': post_step}

    auto_population_size, auto_generations = suggest_ga_params(bounds, step_sizes)
    population_size = user_population_size if user_population_size > 0 else auto_population_size
    generations = user_generations if user_generations > 0 else auto_generations

    st.markdown(f"**Suggested Population Size:** {auto_population_size} &nbsp;&nbsp; **Suggested Generations:** {auto_generations}")

    # --- Run GA Button ---
    run_ga = st.button("Run Genetic Algorithm")

    if run_ga:
        hber_ga = HBERGeneticAlgorithm(
            df_new,
            bounds,
            step_sizes,
            fitness_weights=(w1, w2),
            population_size=population_size,
            generations=generations,
            cxpb=cxpb,
            mutpb=mutpb,
            seed=seed,
            early_stopping=early_stopping
        )
        best = hber_ga.run()

        st.success(f"**Best Solution:** PRE={best[0]}, MAIN={best[1]}, POST={best[2]}")
        st.info(f"Fitness value: {hber_ga.evaluate_point(*best):.4e}")

        # --- Animated Population Evolution ---
        st.markdown("#### Population Evolution (3D Animation)")
        gif_bytes = hber_ga.animate_population()
        st.image(gif_bytes.getvalue(), caption="Population Evolution", use_container_width=True)

        # --- Animated Fitness Trend ---
        st.markdown("#### Fitness Trend Over Generations")
        gif_bytes2 = hber_ga.animate_fitness_trend()
        st.image(gif_bytes2.getvalue(), caption="Fitness Trend", use_container_width=True)