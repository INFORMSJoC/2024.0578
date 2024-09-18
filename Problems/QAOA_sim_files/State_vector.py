import numpy as np
#from qiskit import QuantumCircuit, execute, Aer
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer


def append_zz_term(qc, q1, q2, gamma):
    qc.cx(q1, q2)
    qc.rz(2 * gamma, q2)
    qc.cx(q1, q2)


def get_cost_operator_circuit(G, gamma):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N, N)
    for i, j in G.edges():
        append_zz_term(qc, i, j, gamma)
    return qc


def append_x_term(qc, q1, beta):
    qc.rx(2 * beta, q1)


def get_mixer_operator_circuit(G, beta):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N, N)
    for n in G.nodes():
        append_x_term(qc, n, beta)
    return qc


def maxcut_obj(x, G):
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            # the edge is cut
            cut -= 1
    return cut


def state_num2str(basis_state_as_num, nqubits):
    return "{0:b}".format(basis_state_as_num).zfill(nqubits)


def state_str2num(basis_state_as_str):
    return int(basis_state_as_str, 2)


def state_reverse(basis_state_as_num, nqubits):
    basis_state_as_str = state_num2str(basis_state_as_num, nqubits)
    new_str = basis_state_as_str[::-1]
    return state_str2num(new_str)


def get_adjusted_state(state):
    nqubits = np.log2(state.shape[0])
    if nqubits % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    nqubits = int(nqubits)

    adjusted_state = np.zeros(2**nqubits, dtype=complex)
    for basis_state in range(2**nqubits):
        adjusted_state[state_reverse(basis_state, nqubits)] = state[basis_state]
    return adjusted_state


def get_qaoa_circuit_sv(G, beta, gamma):
    assert len(beta) == len(gamma)
    p = len(beta)  # infering number of QAOA steps from the parameters passed
    N = G.number_of_nodes()
    qc = QuantumCircuit(N, N)
    # first, apply a layer of Hadamards
    qc.h(range(N))
    # second, apply p alternating operators
    for i in range(p):
        qc.append(get_cost_operator_circuit(G, gamma[i]), range(N), range(N))
        qc.append(get_mixer_operator_circuit(G, beta[i]), range(N), range(N))
    # no measurement in the end!
    return qc


def state_to_ampl_counts(vec, eps=1e-15):
    """Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = "0{}b".format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real**2 + val.imag**2 > eps:
            counts[format(kk, str_format)] = val
    return counts


def compute_maxcut_energy_sv(sv, G):
    """Compute objective from statevector
    For large number of qubits, this is slow.
    """
    counts = state_to_ampl_counts(sv)
    mean = sum(maxcut_obj(np.array([int(x) for x in k]), G) * (np.abs(v) ** 2) for k, v in counts.items())
    var = sum(((maxcut_obj(np.array([int(x) for x in k]), G) - mean)**2) * (np.abs(v) ** 2) for k, v in counts.items())
    return mean, var


def State_Vector_Simulator(G, p, theta):
    backend = Aer.get_backend("statevector_simulator")
    beta = theta[:p]
    gamma = theta[p:]
    qc = get_qaoa_circuit_sv(G, beta, gamma)
    #sv = execute(qc, backend).result().get_statevector()
    new_circuit = transpile(qc, backend)
    sv = backend.run(new_circuit).result().get_statevector()
    mean, var = compute_maxcut_energy_sv(get_adjusted_state(sv), G)
    return mean, var
