import math
import numpy as np
from decimal import Decimal, getcontext

def calculate_pi_machin(decimals):
    """
    Dynamically calculate Pi to a specific number of decimal places 
    using Machin's formula.
    """
    # Temporarily increase precision to avoid rounding errors during calculation
    getcontext().prec = decimals + 5
    threshold = Decimal('1e-' + str(decimals + 5))
    
    def arctan_inv(x):
        """Calculate arctan(1/x) using the Taylor series."""
        x_inv = Decimal(1) / Decimal(x)
        x_inv_sq = x_inv * x_inv
        term = x_inv
        total = term
        n = 3
        sign = -1
        
        while term > threshold:
            term *= x_inv_sq
            next_term = term / n
            total += sign * next_term
            n += 2
            sign *= -1
            
        return total

    # Compute Pi using Machin's formula: pi = 16*arctan(1/5) - 4*arctan(1/239)
    pi = 16 * arctan_inv(5) - 4 * arctan_inv(239)
    
    # Truncate/round to the exact precision requested 
    # (+1 accounts for the integer '3' before the decimal point)
    getcontext().prec = decimals + 1 
    
    return +pi  # The unary plus applies the new context precision

def get_pi_float(block_height):
    """Gradually reveal more digits of Pi calculated dynamically by the CPU."""
    # Matches original logic mapping block_height to precision length.
    # Capped at 100 decimal places to prevent infinite computation.
    decimal_places = min(block_height + 1, 100)
    
    # Calculate Pi mathematically on-the-fly
    dynamic_pi = calculate_pi_machin(decimal_places)
    
    # Cast to float to match the original function signature
    # Note: Python floats cap at ~15 decimal digits of precision.
    return float(dynamic_pi)

def calculate_model_driven_weights(old_weights, scores, model_name, block_height):
    """
    Dynamically evolve criterion weights based on model confidence and block height.
    Uses actual model parameters instead of hardcoded constants.
    """
    # Parse actual model parameters from model_name
    if "70b" in model_name:
        model_version, model_size = 3.3, 70.0
    elif "8b" in model_name:
        model_version, model_size = 3.1, 8.0
    else:
        # Default fallback for unknown models
        model_version, model_size = 3.1, 8.0
    
    # Calculate delta using ACTUAL model parameters, not hardcoded constants
    baseline_version, baseline_size = 3.1, 8.0
    delta_models = abs((model_version * model_size) - (baseline_version * baseline_size))
    delta_models = max(delta_models, 1.0)  # Prevent division by zero
    
    pi_accuracy = get_pi_float(block_height)
    mean_score = np.mean(scores)
    
    new_weights = []
    for i, old_w in enumerate(old_weights):
        # Stretch scores to amplify variance (adversarial design)
        stretched_score = max(1.0, min(100.0, mean_score + (scores[i] - mean_score) * 3.0))
        
        # Weight shift based on model capability and score performance
        weight_shift = ((model_version * model_size) / (delta_models * pi_accuracy)) * ((stretched_score / 100.0) ** 2)
        
        # Exponential moving average with evolutionary pressure
        w_new = old_w * 0.85 + (1.0 + weight_shift * 0.15) * 0.15
        new_weights.append(w_new)
        
    # Normalize to sum to 8.0 (as per whitepaper constraint)
    sum_of_weights = sum(new_weights)
    if sum_of_weights == 0:
        return [1.0] * 8  # Genesis fallback
    
    return [round((w / sum_of_weights) * 8.0, 6) for w in new_weights]

def compute_logical_integrity(extracted_logic_vars):
    """
    Adversarial Logic Engine (Δ_LoLic).
    Penalizes papers where conclusions overreach evidence or contain non-sequiturs.
    """
    evidence = extracted_logic_vars.get('Evidence_Strength', 0.5)
    conclusion_reach = extracted_logic_vars.get('Conclusion_Reach', 0.5)
    jumps = extracted_logic_vars.get('Logical_Jumps', 0.5)
    premise = extracted_logic_vars.get('Premise_Validity', 0.5)
    
    # Core formula from whitepaper: exponential decay on logic gap
    logic_gap = max(0.0, conclusion_reach - evidence)
    logic_score = (premise * evidence) * np.exp(-(logic_gap * 2.0 + jumps * 1.5)) * 100
    
    return float(max(0.0, min(100.0, logic_score)))

def compute_formulaic_criteria(vars_dict):
    """
    Compute the 8 multidimensional criteria scores based on extracted proxy variables.
    Formulas aligned with the π-Index WhitePaper (Sections 5.1–5.8).
    """
    scores = {}
    
    # C1: Originality via Epistemic Gradient Fields
    H_novel = vars_dict.get('H_novel', 0.5)
    K_epi = vars_dict.get('K_epistemic', 0.5)
    zeta = vars_dict.get('zeta', 0.5)
    I_ex = vars_dict.get('I_existing', 0.5)
    c1_raw = ((H_novel * K_epi) / (zeta * I_ex + 0.1)) * 60
    scores["C1_Originality"] = min(100.0, max(0.0, c1_raw))
    
    # C2: Methodological Rigor via Error-Covariance Tensors
    sigma_err = vars_dict.get('Sigma_error', 0.2)
    mu_sig = vars_dict.get('mu_signal', 0.8)
    rho_k = vars_dict.get('rho_k', 0.5)
    rigor_matrix = max(0.0, 1.0 - (sigma_err / (mu_sig + 0.1)))
    c2_raw = rigor_matrix * rho_k * math.gamma(1.5) * 140
    scores["C2_Methodological_Rigor"] = min(100.0, max(0.0, c2_raw))
    
    # C3: Interdisciplinary Bridging via Rényi Entropy
    p_disc = np.array(vars_dict.get('p_disciplines', [1.0]))
    p_disc = p_disc / (p_disc.sum() + 1e-9)
    renyi_entropy = -np.log(np.sum(p_disc**2) + 1e-9)
    bridge = vars_dict.get('bridge_capacity', 0.5)
    c3_raw = (renyi_entropy + bridge) * 55
    scores["C3_Interdisciplinary"] = min(100.0, max(0.0, c3_raw))
    
    # C4: Societal Impact via Fractional Stochastic Integration
    q_frac = max(0.1, vars_dict.get('q_fractional', 1.5))
    utility = vars_dict.get('Utility_vector', 0.5)
    decay = vars_dict.get('decay_rate', 0.5)
    gamma_q = math.gamma(q_frac)
    c4_raw = (1.0 / gamma_q) * utility * np.exp(-decay) * 150
    scores["C4_Societal_Impact"] = min(100.0, max(0.0, c4_raw))
    
    # C5: Open Science Potential via Multi-Objective Integration
    D_open = vars_dict.get('D_open', 0.1)
    J_code = vars_dict.get('J_code', 0.1)
    P_FAIR = vars_dict.get('P_FAIR', 0.1)
    c5_raw = ((0.7 * D_open) + (0.3 * J_code)) * P_FAIR * 180
    scores["C5_Open_Science_Potential"] = min(100.0, max(0.0, c5_raw))
    
    # C6: Literature Integration via Non-Euclidean PageRank
    d_g = vars_dict.get('d_g_distance', 0.5)
    R_xi = vars_dict.get('R_xi', 0.5)
    PR_xi = vars_dict.get('PR_xi', 0.5)
    c6_raw = np.exp(-1.5 * d_g) * R_xi * PR_xi * 180
    scores["C6_Literature_Integration"] = min(100.0, max(0.0, c6_raw))
    
    # C7: Empirical Density via Fisher Information
    I_Fish = vars_dict.get('I_Fisher', 0.5)
    KL_div = vars_dict.get('KL_divergence', 0.5)
    V_base = vars_dict.get('V_baseline', 0.5)
    omega = vars_dict.get('omega_data', 0.5)
    sum_lam = vars_dict.get('sum_lambda_kappa', 1.0)
    density_inner = (I_Fish * KL_div) / (V_base * omega + 0.1)
    c7_raw = np.tanh(density_inner) * sum_lam * 80
    scores["C7_Empirical_Density"] = min(100.0, max(0.0, c7_raw))
    
    # C8: Future Actionability via Lyapunov Exponents
    eta = vars_dict.get('eta_steps', 2.0)
    lambda_lya = vars_dict.get('Lambda_Lyapunov', 0.5)
    c8_raw = (1.0 / (1.0 + np.exp(-(eta - (lambda_lya * 5))))) * 100
    scores["C8_Future_Actionability"] = min(100.0, max(0.0, c8_raw))
    
    # Round all scores
    for key in scores:
        scores[key] = round(scores[key], 2)
    
    return scores

def calculate_complex_drift(alignment, scores):
    """
    Epistemic Drift Metric.
    Enforces contextual relevance and prevents portfolio inflation with disjointed work.
    """
    if not scores or alignment is None:
        return 0.0
        
    average_score = np.mean(scores)
    standard_deviation = np.std(scores)
    alignment_gap = (100.0 - alignment) / 100.0
    
    # Complex drift formula from whitepaper Section 7
    numerator = 3.0 * (alignment_gap ** 1.5) * (1.0 + (standard_deviation / 100.0))
    denominator = 0.1 + (average_score / 100.0)
    drift_metric = 100.0 * (1.0 - np.exp(-numerator / denominator))
    
    return float(max(0.0, min(100.0, drift_metric)))

def get_recommendation_spectrum(score, drift):
    """
    Tiered recommendation based on score-drift synergy.
    """
    if drift == "N/A":
        return "N/A"
        
    synergy = score * (1.0 - (drift / 100.0)**1.5)
    
    if synergy >= 85: 
        return "Tier I: Core Paradigm (Optimal Synergy)"
    elif synergy >= 70: 
        return "Tier II: Highly Aligned Framework"
    elif synergy >= 55: 
        return "Tier III: Moderately Synergistic"
    elif synergy >= 40: 
        return "Tier IV: Tangential Relevance"
    elif synergy >= 25: 
        return "Tier V: Epistemic Divergence"
    else: 
        return "Tier VI: Orthogonal / Unrelated Noise"
