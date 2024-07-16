import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def solve_transportation_problem(supply, demand, costs):
    supply = np.array(supply)
    demand = np.array(demand)
    costs = np.array(costs)

    total_supply = sum(supply)
    total_demand = sum(demand)

    if total_supply != total_demand:
        st.error("Total supply must equal total demand")
        return None, None

    num_sources = len(supply)
    num_destinations = len(demand)

    c = costs.flatten()
    A_eq = []
    b_eq = []

    for i in range(num_sources):
        A_eq_row = [0] * (num_sources * num_destinations)
        for j in range(num_destinations):
            A_eq_row[i * num_destinations + j] = 1
        A_eq.append(A_eq_row)
        b_eq.append(supply[i])

    for j in range(num_destinations):
        A_eq_row = [0] * (num_sources * num_destinations)
        for i in range(num_sources):
            A_eq_row[i * num_destinations + j] = 1
        A_eq.append(A_eq_row)
        b_eq.append(demand[j])

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, method='highs')

    if result.success:
        return result.x.reshape((num_sources, num_destinations)), result.fun
    else:
        st.error("No solution found")
        return None, None

def plot_allocation(allocation, supply, demand):
    fig, ax = plt.subplots()
    cax = ax.matshow(allocation, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(demand)))
    ax.set_yticks(np.arange(len(supply)))

    ax.set_xticklabels([f"Destination {i+1}" for i in range(len(demand))])
    ax.set_yticklabels([f"Source {i+1}" for i in range(len(supply))])

    for i in range(len(supply)):
        for j in range(len(demand)):
            text = ax.text(j, i, int(allocation[i, j]), ha="center", va="center", color="black")

    plt.xlabel("Destinations")
    plt.ylabel("Sources")
    plt.title("Optimal Allocation")
    return fig

def main():
    st.title("Transportation Problem Solver")
    
    st.sidebar.header("Input Parameters")
    num_sources = st.sidebar.number_input("Number of Sources", min_value=1, value=3)
    num_destinations = st.sidebar.number_input("Number of Destinations", min_value=1, value=3)
    
    supply = []
    demand = []
    costs = []

    st.sidebar.subheader("Supply")
    for i in range(num_sources):
        supply.append(st.sidebar.number_input(f"Supply of Source {i+1}", min_value=0, value=10))

    st.sidebar.subheader("Demand")
    for j in range(num_destinations):
        demand.append(st.sidebar.number_input(f"Demand of Destination {j+1}", min_value=0, value=10))

    st.sidebar.subheader("Costs")
    for i in range(num_sources):
        row = []
        for j in range(num_destinations):
            row.append(st.sidebar.number_input(f"Cost from Source {i+1} to Destination {j+1}", min_value=0, value=1))
        costs.append(row)

    if st.sidebar.button("Solve"):
        allocation, total_cost = solve_transportation_problem(supply, demand, costs)
        
        if allocation is not None:
            st.subheader("Results")
            st.write("Optimal Allocation:")
            st.write(pd.DataFrame(allocation, columns=[f"Destination {j+1}" for j in range(num_destinations)], index=[f"Source {i+1}" for i in range(num_sources)]))
            st.write(f"Total Cost: ${total_cost:.2f}")
            
            # Plot the allocation matrix
            fig = plot_allocation(allocation, supply, demand)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
