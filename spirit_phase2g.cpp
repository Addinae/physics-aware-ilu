#include <iostream>
#include <vector>
#include <complex>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <set>
#include <chrono>
#include <limits>
#include <omp.h>
#include <atomic>

using namespace std;

// Global atomic counters for physics-aware filtering diagnostics
atomic<int> physics_aware_extra_entries_total(0);
atomic<int> physics_aware_rows_processed(0);

// ---- Helper Functions (Complex Data Types) ----

// Magnitude of a complex vector
double mag(const int n, const vector<complex<double>>& vec) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; i++)
        sum += pow(real(vec[i]), 2) + pow(imag(vec[i]), 2);
    return sqrt(sum);
}

// Dot product for complex vectors - Thread-safe version
complex<double> vecvec(const int n, const vector<complex<double>>& veca, const vector<complex<double>>& vecb) {
    complex<double> sum(0.0, 0.0);
    
    #pragma omp parallel
    {
        complex<double> local_sum(0.0, 0.0);
        #pragma omp for nowait
        for (int i = 0; i < n; i++) {
            local_sum += conj(veca[i]) * vecb[i];
        }
        #pragma omp critical
        {
            sum += local_sum;
        }
    }
    return sum;
}

// Standard matrix-vector multiplication for CSR format
vector<complex<double>> matvec(const vector<complex<double>>& vals, const vector<int>& IA, const vector<int>& JA, int n, const vector<complex<double>>& X) {
    vector<complex<double>> AX(n, 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        complex<double> tmp = 0.0;
        for (int j = IA[i]; j < IA[i+1]; j++)
            tmp += vals[j] * X[JA[j]];
        AX[i] = tmp;
    }
    return AX;
}

// ---- Matrix Market Reader ----

tuple<vector<complex<double>>, vector<int>, vector<int>>
read_matrix_market(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Cannot open Matrix Market file: " + filename);
    }
    
    string line;
    bool is_complex = false, is_symmetric = false;
    
    // Parse header
    while (getline(file, line)) {
        if (line[0] != '%') break;
        if (line.find("complex") != string::npos) is_complex = true;
        if (line.find("symmetric") != string::npos) is_symmetric = true;
    }
    
    // Parse dimensions
    istringstream iss(line);
    int n_rows, n_cols, n_nnz;
    iss >> n_rows >> n_cols >> n_nnz;
    
    cout << "Loading " << n_rows << "x" << n_cols << " matrix with " << n_nnz << " entries" << endl;
    
    // Read triplets (i, j, val) - Matrix Market uses 1-based indexing
    vector<tuple<int, int, complex<double>>> triplets;
    
    for (int k = 0; k < n_nnz; k++) {
        getline(file, line);
        istringstream entry(line);
        int i, j;
        double real_part, imag_part = 0.0;
        
        entry >> i >> j >> real_part;
        if (is_complex) entry >> imag_part;
        
        i--; j--; // Convert to 0-based indexing
        triplets.emplace_back(i, j, complex<double>(real_part, imag_part));
        
        // Add symmetric entry if needed
        if (is_symmetric && i != j) {
            triplets.emplace_back(j, i, complex<double>(real_part, -imag_part)); // Hermitian
        }
    }
    
    // Convert COO to CSR
    sort(triplets.begin(), triplets.end(), 
         [](const auto& a, const auto& b) {
             return get<0>(a) < get<0>(b) || 
                    (get<0>(a) == get<0>(b) && get<1>(a) < get<1>(b));
         });
    
    vector<complex<double>> vals;
    vector<int> JA, IA(n_rows + 1, 0);
    
    int current_row = 0;
    for (const auto& [i, j, val] : triplets) {
        while (current_row <= i) IA[current_row++] = vals.size();
        vals.push_back(val);
        JA.push_back(j);
    }
    while (current_row <= n_rows) IA[current_row++] = vals.size();
    
    cout << "Successfully loaded matrix with " << vals.size() << " non-zeros" << endl;
    return {vals, JA, IA};
}

// ---- Physics-Aware Functions ----

// Calculate row norm helper function
double calculate_row_norm(const vector<complex<double>>& vals, const vector<int>& IA, int row) {
    double norm = 0.0;
    for (int j = IA[row]; j < IA[row+1]; j++) {
        norm += abs(vals[j]) * abs(vals[j]);
    }
    return sqrt(norm);
}

// PROPER surface detection based on connectivity pattern
vector<int> detect_surface_atoms(const vector<int>& IA, const vector<int>& JA, int n) {
    vector<int> surface_indices;
    vector<int> neighbor_count(n, 0);
    
    for (int i = 0; i < n; i++) {
        for (int j = IA[i]; j < IA[i+1]; j++) {
            if (JA[j] != i) { // Exclude diagonal
                neighbor_count[i]++;
            }
        }
    }
    
    // Find atoms with fewer neighbors (true surface atoms)
    double avg_neighbors = 0.0;
    for (int count : neighbor_count) avg_neighbors += count;
    avg_neighbors /= n;
    
    for (int i = 0; i < n; i++) {
        if (neighbor_count[i] < avg_neighbors * 0.7) { // Surface atoms have fewer neighbors
            surface_indices.push_back(i);
        }
    }
    
    cout << "Detected " << surface_indices.size() << " surface atoms (avg neighbors: " 
         << avg_neighbors << ")" << endl;
    return surface_indices;
}

// Define contact regions for realistic devices
vector<int> define_contacts(const vector<int>& surface_indices, int n) {
    vector<int> contacts;
    
    // For typical molecular systems, select atoms at extremes
    if (n > 1000) {
        // Larger system: select surface atoms at boundaries
        int left_contact = surface_indices[0];
        int right_contact = surface_indices.back();
        contacts.push_back(left_contact);
        contacts.push_back(right_contact);
        
        // Also include their strongly coupled neighbors
        cout << "Defined contacts at atoms: " << left_contact << ", " << right_contact << endl;
    } else {
        // Smaller system: use all surface atoms as contacts
        contacts = surface_indices;
        cout << "Using all surface atoms as contacts" << endl;
    }
    
    return contacts;
}

// Find quantum transport pathways
set<int> find_transport_pathways(const vector<int>& contacts, 
                                const vector<int>& surface_indices,
                                const vector<complex<double>>& H_vals,
                                const vector<int>& IA, const vector<int>& JA,
                                int n) {
    set<int> pathways;
    
    // Include contacts and surface atoms
    for (int contact : contacts) {
        pathways.insert(contact);
    }
    for (int surface : surface_indices) {
        pathways.insert(surface);
    }
    
    // Add atoms strongly coupled to contacts (quantum tunneling pathways)
    for (int contact : contacts) {
        for (int j = IA[contact]; j < IA[contact+1]; j++) {
            int neighbor = JA[j];
            double coupling = abs(H_vals[j]);
            if (coupling > 0.1) { // Significant quantum coupling
                pathways.insert(neighbor);
                
                // Second-nearest neighbors for important pathways
                for (int k = IA[neighbor]; k < IA[neighbor+1]; k++) {
                    int second_neighbor = JA[k];
                    double second_coupling = abs(H_vals[k]);
                    if (second_coupling > 0.05) {
                        pathways.insert(second_neighbor);
                    }
                }
            }
        }
    }
    
    cout << "Identified " << pathways.size() << " atoms in quantum transport pathways" << endl;
    return pathways;
}

// Energy-dependent physics criticality
bool is_energy_critical(int atom_id, double energy, const vector<complex<double>>& H_diagonal) {
    // Atoms near the current energy are more important for transport
    double atom_energy = real(H_diagonal[atom_id]);
    return (abs(atom_energy - energy) < 2.0); // Within 2 eV of relevant energy
}

// Extract diagonal elements from CSR matrix
vector<complex<double>> extract_diagonal(const vector<complex<double>>& vals, 
                                       const vector<int>& IA, 
                                       const vector<int>& JA, 
                                       int n) {
    vector<complex<double>> diagonal(n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = IA[i]; j < IA[i+1]; j++) {
            if (JA[j] == i) {
                diagonal[i] = vals[j];
                break;
            }
        }
    }
    return diagonal;
}

// ---- Preconditioner & Filtering ----

// Standard CSR filter (value-based only)
tuple<vector<complex<double>>, vector<int>, vector<int>>
filterCSR(const vector<complex<double>>& vals, const vector<int>& JA, const vector<int>& IA, float tol, int maxfill) {
    const int n_rows = IA.size() - 1;
    const double ABSOLUTE_THRESHOLD = 1e-8;
    vector<int> new_row_sizes(n_rows, 0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_rows; ++i) {
        double row_norm = calculate_row_norm(vals, IA, i);
        double threshold = max(tol * row_norm, ABSOLUTE_THRESHOLD);

        int diagonal_found = 0;
        int kept_elements = 0;
        
        for (int j = IA[i]; j < IA[i+1]; ++j) {
            int col = JA[j];
            complex<double> val = vals[j];
            
            if (col == i) {
                diagonal_found = 1;
            } else if (abs(val) >= threshold && kept_elements < maxfill) {
                kept_elements++;
            }
        }
        
        new_row_sizes[i] = diagonal_found + kept_elements;
    }

    vector<int> new_IA(n_rows + 1, 0);
    for (int i = 0; i < n_rows; ++i) new_IA[i + 1] = new_IA[i] + new_row_sizes[i];
    vector<complex<double>> new_vals(new_IA[n_rows]);
    vector<int> new_JA(new_IA[n_rows]);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_rows; ++i) {
        double row_norm = calculate_row_norm(vals, IA, i);
        double threshold = max(tol * row_norm, ABSOLUTE_THRESHOLD);
        
        vector<pair<double, int>> element_magnitudes; // magnitude, index
        complex<double> diagonal_val = 0.0;
        int diagonal_found = 0;
        
        for (int j = IA[i]; j < IA[i+1]; ++j) {
            int col = JA[j];
            complex<double> val = vals[j];
            double mag_val = abs(val);
            
            if (col == i) {
                diagonal_val = val;
                diagonal_found = 1;
            } else if (mag_val >= threshold) {
                element_magnitudes.emplace_back(mag_val, j);
            }
        }
        
        // Sort by magnitude and take top maxfill
        sort(element_magnitudes.rbegin(), element_magnitudes.rend());
        if (element_magnitudes.size() > maxfill) {
            element_magnitudes.resize(maxfill);
        }
        
        int output_idx = new_IA[i];
        if (diagonal_found) {
            new_vals[output_idx] = diagonal_val;
            new_JA[output_idx++] = i;
        }
        
        for (const auto& [mag, idx] : element_magnitudes) {
            new_vals[output_idx] = vals[idx];
            new_JA[output_idx++] = JA[idx];
        }
    }
    
    return {new_vals, new_JA, new_IA};
}

// Enhanced Physics-aware CSR filter
tuple<vector<complex<double>>, vector<int>, vector<int>>
filterCSR_PhysicsAware(const vector<complex<double>>& vals, const vector<int>& JA, 
                      const vector<int>& IA, float tol, int maxfill, 
                      const vector<int>& surface_indices, const vector<int>& contacts,
                      const set<int>& transport_pathways, double energy,
                      const vector<complex<double>>& H_diagonal) {
    
    const int n_rows = IA.size() - 1;
    const double ABSOLUTE_THRESHOLD = 1e-8;
    vector<int> new_row_sizes(n_rows, 0);

    // Reset counters
    physics_aware_extra_entries_total = 0;
    physics_aware_rows_processed = 0;
   
    // +++ ADDED ENERGY-DEPENDENT PHYSICS SENSITIVITY HERE +++
    // ENERGY-DEPENDENT PHYSICS SENSITIVITY
double physics_tol = tol;
if (abs(energy) < 2.0) { // Near Fermi level (±2 eV)
    // Near Fermi level: quantum transport is most sensitive
    // Be more selective about physics-based keeping
    physics_tol = max(tol * 0.3, 1e-10); // 70% more conservative + safety
    if (abs(energy) < 0.5) { // Very near Fermi level
        physics_tol = max(tol * 0.1, 1e-10); // 90% more conservative + safety
    }
} else {
    // Far from Fermi level: can be more aggressive
    physics_tol = max(tol * 2.0, 1e-10); // Less conservative + safety
}

// ADD THIS SAFETY CHECK (in case tol was already very small)
physics_tol = max(physics_tol, 1e-10);

cout << "  Energy=" << scientific << energy << " -> physics_tol=" << physics_tol << endl;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_rows; ++i) {
        double row_norm = calculate_row_norm(vals, IA, i);
        double threshold = max(tol * row_norm, ABSOLUTE_THRESHOLD);

        int diagonal_found = 0;
        int kept_elements = 0;
        int extra_physics_entries = 0;
        
        // Check if this row is physics-critical
        bool row_is_critical = (transport_pathways.find(i) != transport_pathways.end()) ||
                              is_energy_critical(i, energy, H_diagonal);
        
        for (int j = IA[i]; j < IA[i+1]; ++j) {
            int col = JA[j];
            complex<double> val = vals[j];
            double mag_val = abs(val);
            
            bool element_is_critical = (transport_pathways.find(col) != transport_pathways.end()) ||
                                      is_energy_critical(col, energy, H_diagonal);
            
            bool keep_by_physics = row_is_critical || element_is_critical;
            bool keep_by_value = (mag_val >= threshold);
            
            if (col == i) {
                diagonal_found = 1;
            } else if (keep_by_physics && kept_elements < maxfill * 1.2)  {
                if (keep_by_physics && !keep_by_value) {
                    extra_physics_entries++;
                }
                kept_elements++;
            }
        }
        
        // Update diagnostics
        physics_aware_extra_entries_total += extra_physics_entries;
        physics_aware_rows_processed++;
        
        new_row_sizes[i] = diagonal_found + min(kept_elements, 2 * maxfill);
    }

    vector<int> new_IA(n_rows + 1, 0);
    for (int i = 0; i < n_rows; ++i) new_IA[i + 1] = new_IA[i] + new_row_sizes[i];
    vector<complex<double>> new_vals(new_IA[n_rows]);
    vector<int> new_JA(new_IA[n_rows]);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_rows; ++i) {
        double row_norm = calculate_row_norm(vals, IA, i);
        double threshold = max(tol * row_norm, ABSOLUTE_THRESHOLD);
        
        vector<pair<double, int>> element_magnitudes; // magnitude, index
        complex<double> diagonal_val = 0.0;
        int diagonal_found = 0;
        
        // Check physics criticality
        bool row_is_critical = (transport_pathways.find(i) != transport_pathways.end()) ||
                              is_energy_critical(i, energy, H_diagonal);
        
        for (int j = IA[i]; j < IA[i+1]; ++j) {
            int col = JA[j];
            complex<double> val = vals[j];
            double mag_val = abs(val);
            
            bool element_is_critical = (transport_pathways.find(col) != transport_pathways.end()) ||
                                      is_energy_critical(col, energy, H_diagonal);
            
            bool keep_by_physics = row_is_critical || element_is_critical;
            
            if (col == i) {
                diagonal_val = val;
                diagonal_found = 1;
            } else if (mag_val >= threshold || keep_by_physics) {
                element_magnitudes.emplace_back(mag_val, j);
            }
        }
        
        // Sort and limit
        sort(element_magnitudes.rbegin(), element_magnitudes.rend());
        if (element_magnitudes.size() > 2 * maxfill) {
            element_magnitudes.resize(2 * maxfill);
        }
        
        int output_idx = new_IA[i];
        if (diagonal_found) {
            new_vals[output_idx] = diagonal_val;
            new_JA[output_idx++] = i;
        }
        
        for (const auto& [mag, idx] : element_magnitudes) {
            new_vals[output_idx] = vals[idx];
            new_JA[output_idx++] = JA[idx];
        }
    }
    
    return {new_vals, new_JA, new_IA};
}

// ILU(0) factorization for complex matrices (in CSR)
vector<complex<double>> ilu0(const vector<complex<double>>& A, const vector<int>& IA, const vector<int>& JA, int size) {
    vector<complex<double>> diagvals(size, 0);
    vector<complex<double>> LU(A);

    for (int i = 0; i < size; ++i) {
        // Find diagonal
        int diagind = -1;
        for (int ind = IA[i]; ind < IA[i + 1]; ++ind) {
            if (JA[ind] == i) {
                diagind = ind;
                break;
            }
        }
        if (diagind == -1) continue;
        
        for (int ind = IA[i]; ind < IA[i + 1]; ++ind) {
            int k = JA[ind];
            if (k >= i) continue;
            
            complex<double> L_ik = LU[ind] / diagvals[k];
            LU[ind] = L_ik;
            
            // Update row
            for (int j = ind + 1; j < IA[i + 1]; ++j) {
                int col_j = JA[j];
                for (int l = IA[k]; l < IA[k + 1]; ++l) {
                    if (JA[l] == col_j) {
                        LU[j] -= L_ik * LU[l];
                        break;
                    }
                }
            }
        }
        diagvals[i] = LU[diagind];
        if (abs(diagvals[i]) < 1e-12) diagvals[i] = complex<double>(1e-12, 0.0);
    }
    return LU;
}

// LU solve for CSR (complex)
vector<complex<double>> lusolve(const vector<complex<double>>& vals, const vector<int>& IA, const vector<int>& JA, int n, const vector<complex<double>>& B) {
    vector<complex<double>> X(n, 0.0);
    
    // Forward substitution
    for (int i = 0; i < n; i++) {
        complex<double> sum = B[i];
        for (int j = IA[i]; j < IA[i+1]; j++) {
            if (JA[j] < i) {
                sum -= vals[j] * X[JA[j]];
            }
        }
        X[i] = sum;
    }
    
    // Backward substitution
    for (int i = n-1; i >= 0; i--) {
        for (int j = IA[i]; j < IA[i+1]; j++) {
            if (JA[j] > i) {
                X[i] -= vals[j] * X[JA[j]];
            } else if (JA[j] == i) {
                if (abs(vals[j]) > 1e-12) {
                    X[i] /= vals[j];
                }
            }
        }
    }
    
    return X;
}

// Residual calculation for CSR
vector<complex<double>> rsolv(const vector<complex<double>>& vals, const vector<int>& IA, const vector<int>& JA, int n, const vector<complex<double>>& X, const vector<complex<double>>& B) {
    vector<complex<double>> r(n, 0.0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        complex<double> sum = 0.0;
        for (int j = IA[i]; j < IA[i+1]; j++) {
            sum += vals[j] * X[JA[j]];
        }
        r[i] = B[i] - sum;
    }
    return r;
}

// BiCGSTAB for complex matrices
int bicgstab(const vector<complex<double>>& A_vals, const vector<int>& A_IA, const vector<int>& A_JA,
             const vector<complex<double>>& LU_vals, const vector<int>& LU_IA, const vector<int>& LU_JA,
             const vector<complex<double>>& B, vector<complex<double>>& X,
             int max_iter = 1000, double tol = 1e-6) {
    int n = (int)B.size();
    vector<complex<double>> r = rsolv(A_vals, A_IA, A_JA, n, X, B);
    vector<complex<double>> r0 = r;
    vector<complex<double>> p = r;
    vector<complex<double>> v(n, 0.0); // ADD THIS LINE - v vector declaration
    complex<double> rho_old(1.0, 0.0), alpha(1.0, 0.0), omega(1.0, 0.0);
    
    for (int iter = 0; iter < max_iter; ++iter) {
        complex<double> rho_new = vecvec(n, r0, r);
        if (abs(rho_new) < 1e-14) return iter;
        
        if (iter > 0) {
            complex<double> beta = (rho_new / rho_old) * (alpha / omega);
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                p[i] = r[i] + beta * (p[i] - omega * v[i]);
            }
        }
        
        v = matvec(A_vals, A_IA, A_JA, n, p); // ADD THIS LINE - compute v
        complex<double> denom = vecvec(n, r0, v);
        if (abs(denom) < 1e-14) return iter;
        alpha = rho_new / denom;
        
        vector<complex<double>> s(n);
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            s[i] = r[i] - alpha * v[i];
        }
        
        if (mag(n, s) < tol) {
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                X[i] += alpha * p[i];
            }
            return iter + 1;
        }
        
        vector<complex<double>> t = matvec(A_vals, A_IA, A_JA, n, s);
        complex<double> omega_denom = vecvec(n, t, t);
        if (abs(omega_denom) < 1e-14) return iter;
        omega = vecvec(n, t, s) / omega_denom;
        
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            X[i] += alpha * p[i] + omega * s[i];
            r[i] = s[i] - omega * t[i];
        }
        
        if (mag(n, r) < tol) return iter + 1;
        rho_old = rho_new;
    }
    
    return max_iter;
}

// ---- NEGF System Construction ----

vector<complex<double>> create_NEGF_matrix(
    const vector<complex<double>>& H_vals, const vector<int>& H_IA, const vector<int>& H_JA,
    int n, double E, double eta, const vector<int>& surface_indices, double Gamma) {
    
    vector<complex<double>> A_vals = H_vals;
    complex<double> energy_term(E, eta);
    complex<double> self_energy(0.0, -Gamma);
    
    // Precompute surface set for faster lookup
    set<int> surface_set(surface_indices.begin(), surface_indices.end());
    
    for (int i = 0; i < n; i++) {
        for (int j = H_IA[i]; j < H_IA[i+1]; j++) {
            if (H_JA[j] == i) {
                A_vals[j] = energy_term - A_vals[j];
                
                if (surface_set.find(i) != surface_set.end()) {
                    A_vals[j] += self_energy;
                }
                break;
            }
        }
    }
    
    return A_vals;
}

tuple<double, double, double> analyze_hamiltonian(const vector<complex<double>>& H_vals) {
    double max_element = 0.0;
    double sum = 0.0;
    
    for (const auto& val : H_vals) {
        double magnitude = abs(val);
        max_element = max(max_element, magnitude);
        sum += magnitude;
    }
    
    double avg_element = sum / H_vals.size();
    double suggested_gamma = 0.01 * avg_element;
    
    cout << "Hamiltonian analysis:" << endl;
    cout << "  Max element magnitude: " << max_element << endl;
    cout << "  Average magnitude: " << avg_element << endl;
    cout << "  Suggested Γ range: " << suggested_gamma * 0.01 << " to " << suggested_gamma << endl;
    
    return {max_element, avg_element, suggested_gamma};
}

// ---- Benchmarking Framework ----

struct BenchmarkResult {
    bool converged;
    int iterations;
    double residual;
    double true_error;
    int nnz_precond;
    double solve_time;
    string method_name;
    double energy;
    double gamma;
    float tol;
    int maxfill;
    int physics_extra_entries;
};

void write_results_csv(const vector<BenchmarkResult>& results, const string& filename) {
    ofstream file(filename);
    file << "Method,Energy,Gamma,Tol,MaxFill,Converged,Iterations,Residual,TrueError,NNZ,SolveTime,PhysicsExtraEntries" << endl;
    
    for (const auto& r : results) {
        file << r.method_name << "," << r.energy << "," << r.gamma << "," << r.tol << "," << r.maxfill << ","
             << r.converged << "," << r.iterations << "," << r.residual << ","
             << r.true_error << "," << r.nnz_precond << "," << r.solve_time << ","
             << r.physics_extra_entries << endl;
    }
    
    cout << "Results written to " << filename << endl;
}

void analyze_benchmark_results(const vector<BenchmarkResult>& results) {
    map<string, int> convergence_count, total_count;
    map<string, double> avg_error, avg_time, avg_physics_extra;
    
    for (const auto& r : results) {
        total_count[r.method_name]++;
        if (r.converged) convergence_count[r.method_name]++;
        
        if (!std::isnan(r.true_error)) avg_error[r.method_name] += r.true_error;
        if (!std::isnan(r.solve_time)) avg_time[r.method_name] += r.solve_time;
        if (!std::isnan(r.physics_extra_entries)) avg_physics_extra[r.method_name] += r.physics_extra_entries;
    }
    
    cout << "\n=== BENCHMARK RESULTS ANALYSIS ===" << endl;
    cout << "Method           | Converged | Success Rate | Avg Error     | Avg Time   | Avg Physics Extra" << endl;
    cout << "-----------------|-----------|--------------|---------------|------------|------------------" << endl;
    for (const string& method : {"Standard", "Physics-Aware"}) {
        if (total_count.find(method) == total_count.end()) continue;
        
        double success_rate = 100.0 * convergence_count[method] / total_count[method];
        cout << left << setw(16) << method << " | "
             << setw(9) << convergence_count[method] << " | "
             << setw(11) << fixed << setprecision(1) << success_rate << "% | "
             << setw(13) << scientific << setprecision(2) << avg_error[method]/total_count[method] << " | "
             << setw(10) << fixed << setprecision(3) << avg_time[method]/total_count[method] << " | "
             << setw(16) << fixed << setprecision(1) << avg_physics_extra[method]/total_count[method] << endl;
    }
    
    cout << "\nKey Success Stories:" << endl;
    int physics_wins = 0;
    double max_improvement = 0.0;
    
    for (size_t i = 0; i < results.size(); i += 2) {
        if (i+1 < results.size() && results[i].method_name == "Standard" && results[i+1].method_name == "Physics-Aware") {
            if (!results[i].converged && results[i+1].converged) {
                cout << "Physics-aware FIXES convergence failure at E=" 
                     << results[i].energy << ", Γ=" << results[i].gamma << endl;
                physics_wins++;
            } else if (!std::isnan(results[i].true_error) && !std::isnan(results[i+1].true_error) 
                      && results[i+1].true_error < results[i].true_error * 0.5) {
                double improvement = results[i].true_error / results[i+1].true_error;
                max_improvement = max(max_improvement, improvement);
                physics_wins++;
            }
        }
    }
    
    cout << "Physics-aware advantages found: " << physics_wins << " cases" << endl;
    cout << "Maximum error improvement: " << max_improvement << "x" << endl;
}

// ---- Enhanced Benchmarking with Error Handling ----

void run_comprehensive_benchmark(const string& matrix_file) {
    cout << "\n=== PHYSICS-AWARE PRECONDITIONER BENCHMARK ===" << endl;
    cout << "Loading matrix: " << matrix_file << endl;
    
    // Load the real Hamiltonian
    auto [H_vals, H_JA, H_IA] = read_matrix_market(matrix_file);
    int n = H_IA.size() - 1;
    
    // Analyze system and setup physics
    auto [max_H, avg_H, suggested_gamma] = analyze_hamiltonian(H_vals);
    vector<int> surface_indices = detect_surface_atoms(H_IA, H_JA, n);
    vector<int> contacts = define_contacts(surface_indices, n);
    vector<complex<double>> H_diagonal = extract_diagonal(H_vals, H_IA, H_JA, n);
    
    // Define parameter sweeps - focus on challenging cases
    vector<double> energy_range = {-avg_H, 0.0, avg_H};
    vector<double> gamma_range = {suggested_gamma * 0.01, suggested_gamma * 0.001, suggested_gamma * 0.0001};
    vector<pair<float, int>> filtering_params = {{1e-4, 10}, {1e-5, 20}};
    
    vector<BenchmarkResult> results;
    double eta = 1e-5;
    
    cout << "\nRunning benchmark sweep..." << endl;
    cout << "Energy points: " << energy_range.size() << endl;
    cout << "Gamma points: " << gamma_range.size() << endl;
    cout << "Filtering variants: " << filtering_params.size() << endl;
    
    int test_count = 0;
    int total_tests = energy_range.size() * gamma_range.size() * filtering_params.size() * 2;
    
    for (double E : energy_range) {
        // Find transport pathways for this energy
        set<int> transport_pathways = find_transport_pathways(contacts, surface_indices, 
                                                            H_vals, H_IA, H_JA, n);
        
        for (double Gamma : gamma_range) {
            // Create NEGF matrix
            vector<complex<double>> A_vals = create_NEGF_matrix(H_vals, H_IA, H_JA, n, E, eta, surface_indices, Gamma);
            // Replace uniform RHS with physics-based RHS
            vector<complex<double>> B(n, 0.0);
            for (int contact : contacts) {
                B[contact] = complex<double>(1.0, 0.0);  // Current injection at contacts
            }    
            
            for (auto [tol, maxfill] : filtering_params) {
                cout << "Testing E=" << scientific << setprecision(2) << E 
                     << ", Gamma=" << Gamma << ", tol=" << tol << ", maxfill=" << maxfill << endl;
                
                for (int method = 0; method < 2; method++) {
                    BenchmarkResult result;
                    result.energy = E;
                    result.gamma = Gamma;
                    result.tol = tol;
                    result.maxfill = maxfill;
                    result.method_name = (method == 0) ? "Standard" : "Physics-Aware";
                    
                    auto start_time = chrono::high_resolution_clock::now();
                    
                    try {
                        // Apply filtering
                        vector<complex<double>> filt_vals;
                        vector<int> filt_JA, filt_IA;
                        
                        if (method == 0) {
                            tie(filt_vals, filt_JA, filt_IA) = filterCSR(A_vals, H_JA, H_IA, tol, maxfill);
                            result.physics_extra_entries = 0;
                        } else {
                            tie(filt_vals, filt_JA, filt_IA) = filterCSR_PhysicsAware(
                                A_vals, H_JA, H_IA, tol, maxfill, 
                                surface_indices, contacts, transport_pathways, E, H_diagonal);
                            result.physics_extra_entries = physics_aware_extra_entries_total;
                        }
                        
                        // ILU factorization
                        vector<complex<double>> LU = ilu0(filt_vals, filt_IA, filt_JA, n);
                        
                        // Solve with BiCGSTAB
                        vector<complex<double>> X(n, 0.0);
                        result.iterations = bicgstab(A_vals, H_IA, H_JA, LU, filt_IA, filt_JA, B, X, 500, 1e-4);
                        
                        auto end_time = chrono::high_resolution_clock::now();
                        result.solve_time = chrono::duration<double>(end_time - start_time).count();
                        
                        // Compute metrics
                        result.converged = (result.iterations < 500);
                        vector<complex<double>> residual = rsolv(A_vals, H_IA, H_JA, n, X, B);
                        result.residual = mag(n, residual);
                        
                        // More realistic error estimation
                        // Better error metric for NEGF
                        // Realistic quantum transport error metrics
                        double physical_error = 0.0;
                        int count = 0;

                        for (int i = 0; i < n; i++) {
                            double magnitude = abs(X[i]);
                            double imag_part = abs(imag(X[i]));
    
                            // Realistic thresholds for quantum transport
                            if (magnitude > 100.0) { // Much lower threshold
                                physical_error += log10(magnitude/10.0); // Logarithmic penalty
                                      count++;
                            }

                            if (imag_part > 1.0) { // Much lower threshold  
                                physical_error += imag_part; // Linear penalty
                                count++;
                            }

                        // Additional: check for NaN/inf
                            if (!isfinite(real(X[i])) || !isfinite(imag(X[i]))) {
                                physical_error += 100.0; // Severe penalty
                                count++;
                            }
                        }

                        if (count > 0) {
                            result.true_error = physical_error / count;
                        } else {
                               result.true_error = 1e-16; // Minimal error for "perfect" solutions
                        }
                        
                        cout << "  " << result.method_name << ": " 
                             << (result.converged ? "CONVERGED" : "FAILED") 
                             << " in " << result.iterations << " iterations, "
                             << "residual=" << scientific << setprecision(1) << result.residual 
                             << ", error=" << result.true_error << endl;
                        
                    } catch (const exception& e) {
                        auto end_time = chrono::high_resolution_clock::now();
                        result.solve_time = chrono::duration<double>(end_time - start_time).count();
                        
                        result.converged = false;
                        result.iterations = 500;
                        result.residual = numeric_limits<double>::quiet_NaN();
                        result.true_error = numeric_limits<double>::quiet_NaN();
                        result.nnz_precond = 0;
                        result.physics_extra_entries = 0;
                        
                        cout << "  " << result.method_name << " FAILED: " << e.what() << endl;
                    }
                    
                    results.push_back(result);
                    test_count++;
                    
                    if (test_count % 5 == 0) {
                        cout << "=== Progress: " << test_count << "/" << total_tests 
                             << " (" << fixed << setprecision(1) << 100.0*test_count/total_tests << "%) ===" << endl;
                    }
                }
            }
        }
    }
    
    // Analysis and output
    cout << "\n=== BENCHMARK COMPLETE ===" << endl;
    cout << "Total tests run: " << results.size() << endl;
    
    analyze_benchmark_results(results);
    write_results_csv(results, "physics_aware_results.csv");
    
    // Additional physics insights
    int total_physics_extra = 0;
    for (const auto& result : results) {
        if (result.method_name == "Physics-Aware") {
            total_physics_extra += result.physics_extra_entries;
        }
    }
    cout << "\nTotal physics-based extra entries kept: " << total_physics_extra << endl;
}

// ---- Main Function ----

int main(int argc, char* argv[]) {
    cout << "=== ADVANCED PHYSICS-AWARE PRECONDITIONER ===" << endl;
    cout << "Incorporating quantum transport physics into ILU preconditioning" << endl;
    
    string matrix_file;
    if (argc > 1) {
        matrix_file = argv[1];
    } else {
        matrix_file = "hamiltonian.mtx";
        cout << "No matrix file specified. Using default: " << matrix_file << endl;
        cout << "Usage: " << argv[0] << " <matrix_file.mtx>" << endl;
    }
    
    try {
        run_comprehensive_benchmark(matrix_file);
    } catch (const exception& e) {
        cerr << "Fatal error: " << e.what() << endl;
        return 1;
    }
    
    cout << "\n=== ANALYSIS COMPLETE ===" << endl;
    cout << "Results saved to physics_aware_results.csv" << endl;
    cout << "finish" << endl;
    
    return 0;
}