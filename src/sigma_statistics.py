import numpy as np

def calc_mean_and_sigma(counts, params):
    # params is the run-parameter dict (see initialize_parameters in the tests).
    # Only N_sites and df are used; the former `shots` argument was never read in the
    # body, so it is gone.
    N_sites = params["N_sites"]
    df = params["df"]

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

            # Map the measured bit to a Pauli eigenvalue (0 -> +1, 1 -> -1).
            # The caller is responsible for rotating the desired basis (X/Y)
            # into the computational basis before measuring, so the mapping is
            # the same regardless of which Pauli is being measured.
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

