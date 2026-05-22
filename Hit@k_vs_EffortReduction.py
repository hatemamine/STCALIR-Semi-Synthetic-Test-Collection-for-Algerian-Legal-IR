import matplotlib.pyplot as plt

# Given Hit@k values
hit_at_k = {
    1000: 0.9121184088806661,
    100: 0.902867715078631,
    50: 0.8852913968547641,
    30: 0.8621646623496763,
    20: 0.8436632747456059,
    10: 0.7853839037927844
}

# Sort k values
k_values = sorted(hit_at_k.keys())

# Compute effort reduction: 1 - k/1000
effort_reduction = [1 - (k / 1000) for k in k_values]

# Get corresponding Hit@k
hit_values = [hit_at_k[k] for k in k_values]

# Plot
plt.figure()
plt.plot(effort_reduction, hit_values, marker='o')

# Labels
plt.xlabel('Effort Reduction (1 - k/1000)')
plt.ylabel('Hit@k')
plt.title('Hit@k vs Effort Reduction')

# Annotate points with k values
for i, k in enumerate(k_values):
    plt.text(effort_reduction[i], hit_values[i], f'k={k}')

plt.grid(True)
plt.show()
