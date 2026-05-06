import numpy as np

def calc_mean_and_sigma(counts, shots, measure, N_sites, df=0):
    # Initialize lists to store the mean and standard deviation for each qubit
    mean_values = []
    std_values = []

    # Loop over all the qubits
    for qubit in range(N_sites):
        # Initialize lists to store the Pauli outcomes for this qubit
        outcomes = []

        # Loop over all the outcomes in the counts
        for outcome, count in counts.items():
            # Ensure the outcome string is padded to the correct length
            outcome = outcome.zfill(N_sites)  # Pad with leading zeros if necessary

            # Map the outcome to +1 or -1 for Z-basis measurement
            if measure == 'Z':
                value = 1 if outcome[qubit] == '0' else -1  # For Z-basis measurement (0 -> +1, 1 -> -1)
            elif measure == 'X':
                # For X-basis, apply Hadamard transformation logic (which corresponds to equal superposition of |0> and |1>)
                value = 1 if outcome[qubit] == '0' else -1
            elif measure == 'Y':
                # For Y-basis, apply phase shift (S gate), so same logic as X, but with a phase difference
                value = 1 if outcome[qubit] == '0' else -1

            # Add the outcome to the list
            outcomes.extend([value] * count)  # Extend the list by the count of this outcome

        # Normalize the outcomes (mean for this qubit)
        mean_sigma = np.mean(outcomes)

        # Calculate the standard deviation
        std_sigma = np.std(outcomes, ddof=df)  # Standard deviation for the Pauli operator outcomes

        # Store the calculated mean and standard deviation for this qubit
        mean_values.append(mean_sigma)
        std_values.append(std_sigma)

        # Print the results for debugging
        # print(f"Qubit {qubit} - Mean: {mean_sigma}, Std Dev: {std_sigma}")

    return mean_values, std_values

