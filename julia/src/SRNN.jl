
module SRNN

using Interpolations, Plots, Random, LinearAlgebra, Statistics, Printf

# activation.jl
relu(x) = max.(0.0, x)
get_activation(fn) = (fn !== nothing) ? fn : relu

# params.jl
struct Params
    n_E::Int
    n_I::Int
    E_indices::Vector{Int}
    I_indices::Vector{Int}
    n_a_E::Int
    n_a_I::Int
    n_b_E::Int
    n_b_I::Int
    tau_a_E::Union{Vector{Float64}, Nothing}
    tau_a_I::Union{Vector{Float64}, Nothing}
    tau_b_E::Union{Vector{Float64}, Nothing}
    tau_b_I::Union{Vector{Float64}, Nothing}
    tau_d::Float64
    n::Int
    M::Matrix{Float64}
    EI_vec::Vector{Int}
    c_SFA::Vector{Float64}
    F_STD::Vector{Float64}
    tau_STD::Float64
    activation_function::Function
end

function package_params(
    n_E, n_I, E_indices, I_indices,
    n_a_E, n_a_I, n_b_E, n_b_I,
    tau_a_E, tau_a_I, tau_b_E, tau_b_I,
    tau_d, n, M, c_SFA, F_STD, tau_STD, EI_vec;
    activation_function=nothing
)
    _arr(x) = (x === nothing) ? nothing : vec(collect(Float64, x))
    _arr_int(x) = (x === nothing) ? nothing : vec(collect(Int, x))
    act_fn = (activation_function === nothing) ? relu : activation_function
    return Params(
        n_E, n_I,
        _arr_int(E_indices), _arr_int(I_indices),
        n_a_E, n_a_I, n_b_E, n_b_I,
        (n_a_E > 0) ? _arr(tau_a_E) : nothing,
        (n_a_I > 0) ? _arr(tau_a_I) : nothing,
        (n_b_E > 0) ? _arr(tau_b_E) : nothing,
        (n_b_I > 0) ? _arr(tau_b_I) : nothing,
        Float64(tau_d),
        n,
        collect(Float64, M),
        _arr_int(EI_vec),
        _arr(c_SFA),
        _arr(F_STD),
        Float64(tau_STD),
        act_fn
    )
end

# utils/get_EI_indices.jl
function get_EI_indices(EI_vec)
    E_indices = findall(x -> x == 1, EI_vec)
    I_indices = findall(x -> x == -1, EI_vec)
    n_E = length(E_indices)
    n_I = length(I_indices)
    return E_indices, I_indices, n_E, n_I
end

# utils/generate_M_no_iso.jl
function generate_M_no_iso(n::Int, w::Dict, sparsity::Float64, EI::Float64; rng=Random.default_rng())
    EI_vec = -ones(Int, n)
    EI_vec[1:round(Int, EI * n)] .= 1
    E0 = n * (n - 1)
    E_keep = round(Int, (1 - sparsity) * E0)
    if E_keep < n
        error("Requested sparsity too high; cannot ensure strong connectivity")
    end
    mask = zeros(Bool, n, n)
    perm = randperm(rng, n)
    nxt = circshift(perm, -1)
    for i in 1:n
        mask[perm[i], nxt[i]] = true
    end
    E_add = E_keep - sum(mask)
    if E_add > 0
        avail = .!mask .& .!Matrix(I, n, n)
        idxs = findall(avail)
        pick_idx = randperm(rng, length(idxs))[1:E_add]
        sel = idxs[pick_idx]
        mask[sel] .= true
    end
    A = randn(rng, n, n)
    A[.!mask] .= 0.0
    A[diagind(A)] .= 0.0
    A[:, EI_vec .== 1] = abs.(A[:, EI_vec .== 1])
    A[:, EI_vec .== -1] = -abs.(A[:, EI_vec .== -1])
    E_indices = findall(x -> x == 1, EI_vec)
    I_indices = findall(x -> x == -1, EI_vec)
    A[I_indices, E_indices] .*= get(w, "EI", 1.0)
    A[E_indices, I_indices] .*= get(w, "IE", 1.0)
    A[E_indices, E_indices] .*= get(w, "EE", 1.0)
    A[I_indices, I_indices] .*= get(w, "II", 1.0)
    A[diagind(A)] .= get(w, "selfI", 0.0)
    diag_idx_E = E_indices
    A[diagind(A)[diag_idx_E]] .= get(w, "selfE", 0.0)
    return A, EI_vec
end

# utils/get_minmax_range.jl
function get_minmax_range(params::Params)
    num_a_E = params.n_E * params.n_a_E
    num_a_I = params.n_I * params.n_a_I
    num_b_E = params.n_E * params.n_b_E
    num_b_I = params.n_I * params.n_b_I
    N_states = num_a_E + num_a_I + num_b_E + num_b_I + params.n
    if N_states == 0
        return Matrix{Float64}(undef, 0, 2)
    end
    bounds = fill(NaN, (N_states, 2))
    idx = 1 + num_a_E + num_a_I
    if num_b_E > 0
        bounds[idx:idx+num_b_E-1, :] .= [0.0 1.0]
    end
    idx += num_b_E
    if num_b_I > 0
        bounds[idx:idx+num_b_I-1, :] .= [0.0 1.0]
    end
    return bounds
end

# state.jl
function unpack_state(X, params::Params)
    X = vec(X)
    len_a_E = params.n_E * params.n_a_E
    len_a_I = params.n_I * params.n_a_I
    len_b_E = params.n_E * params.n_b_E
    len_b_I = params.n_I * params.n_b_I
    idx = 1
    a_E = (len_a_E > 0) ? reshape(X[idx:idx+len_a_E-1], params.n_E, params.n_a_E) : nothing
    idx += len_a_E
    a_I = (len_a_I > 0) ? reshape(X[idx:idx+len_a_I-1], params.n_I, params.n_a_I) : nothing
    idx += len_a_I
    b_E = (len_b_E > 0) ? reshape(X[idx:idx+len_b_E-1], params.n_E, params.n_b_E) : nothing
    idx += len_b_E
    b_I = (len_b_I > 0) ? reshape(X[idx:idx+len_b_I-1], params.n_I, params.n_b_I) : nothing
    idx += len_b_I
    u_d = X[idx:idx+params.n-1]
    return a_E, a_I, b_E, b_I, u_d
end

function unpack_trajectory(X_traj, params::Params)
    _, nt = size(X_traj)
    len_a_E = params.n_E * params.n_a_E
    len_a_I = params.n_I * params.n_a_I
    len_b_E = params.n_E * params.n_b_E
    len_b_I = params.n_I * params.n_b_I
    idx = 1
    a_E = (len_a_E > 0) ? reshape(X_traj[idx:idx+len_a_E-1, :], params.n_E, params.n_a_E, nt) : nothing
    idx += len_a_E
    a_I = (len_a_I > 0) ? reshape(X_traj[idx:idx+len_a_I-1, :], params.n_I, params.n_a_I, nt) : nothing
    idx += len_a_I
    b_E = (len_b_E > 0) ? reshape(X_traj[idx:idx+len_b_E-1, :], params.n_E, params.n_b_E, nt) : nothing
    idx += len_b_E
    b_I = (len_b_I > 0) ? reshape(X_traj[idx:idx+len_b_I-1, :], params.n_I, params.n_b_I, nt) : nothing
    idx += len_b_I
    u_d = X_traj[idx:idx+params.n-1, :]
    return a_E, a_I, b_E, b_I, u_d
end

# dependent.jl
function compute_dependent(a_E_ts, a_I_ts, b_E_ts, b_I_ts, u_d_ts, params::Params)
    if u_d_ts === nothing || isempty(u_d_ts)
        return Matrix{Float64}(undef, 0, 0), Matrix{Float64}(undef, 0, 0)
    end
    u_eff = copy(u_d_ts)
    if params.n_E > 0 && params.n_a_E > 0 && a_E_ts !== nothing
        sum_a_E = sum(a_E_ts, dims=2)
        u_eff[params.E_indices, :] .-= params.c_SFA[params.E_indices] .* dropdims(sum_a_E, dims=2)
    end
    if params.n_I > 0 && params.n_a_I > 0 && a_I_ts !== nothing
        sum_a_I = sum(a_I_ts, dims=2)
        u_eff[params.I_indices, :] .-= params.c_SFA[params.I_indices] .* dropdims(sum_a_I, dims=2)
    end
    r = params.activation_function(u_eff)
    p = copy(r)
    if params.n_E > 0 && params.n_b_E > 0 && b_E_ts !== nothing
        prod_b_E = prod(b_E_ts, dims=2)
        p[params.E_indices, :] .*= dropdims(prod_b_E, dims=2)
    end
    if params.n_I > 0 && params.n_b_I > 0 && b_I_ts !== nothing
        prod_b_I = prod(b_I_ts, dims=2)
        p[params.I_indices, :] .*= dropdims(prod_b_I, dims=2)
    end
    return r, p
end

# dynamics.jl
function make_rhs(t_ex, u_ex, params::Params)
    # Create separate interpolators for each neuron (row of u_ex)
    n_neurons = size(u_ex, 1)
    u_interps = [linear_interpolation(t_ex, u_ex[i, :], extrapolation_bc=NaN) for i in 1:n_neurons]
    
    function rhs(X, t)
        u = [u_interps[i](t) for i in 1:n_neurons]
        a_E, a_I, b_E, b_I, u_d = unpack_state(X, params)
        u_eff = copy(u_d)
        if params.n_E > 0 && params.n_a_E > 0 && a_E !== nothing
            u_eff[params.E_indices] .-= params.c_SFA[params.E_indices] .* sum(a_E, dims=2)
        end
        if params.n_I > 0 && params.n_a_I > 0 && a_I !== nothing
            u_eff[params.I_indices] .-= params.c_SFA[params.I_indices] .* sum(a_I, dims=2)
        end
        r = params.activation_function(u_eff)
        p = copy(r)
        if params.n_E > 0 && params.n_b_E > 0 && b_E !== nothing
            p[params.E_indices] .*= prod(b_E, dims=2)
        end
        if params.n_I > 0 && params.n_b_I > 0 && b_I !== nothing
            p[params.I_indices] .*= prod(b_I, dims=2)
        end
        d_a_E = Float64[]
        if params.n_E > 0 && params.n_a_E > 0 && a_E !== nothing
            d_a_E_mat = (r[params.E_indices] .- a_E) ./ params.tau_a_E'
            mask = params.c_SFA[params.E_indices] .== 0
            if any(mask); d_a_E_mat[mask, :] .= 0.0; end
            d_a_E = vec(d_a_E_mat)
        end
        d_a_I = Float64[]
        if params.n_I > 0 && params.n_a_I > 0 && a_I !== nothing
            d_a_I_mat = (r[params.I_indices] .- a_I) ./ params.tau_a_I'
            mask = params.c_SFA[params.I_indices] .== 0
            if any(mask); d_a_I_mat[mask, :] .= 0.0; end
            d_a_I = vec(d_a_I_mat)
        end
        d_b_E = Float64[]
        if params.n_E > 0 && params.n_b_E > 0 && b_E !== nothing
            d_b_E_mat = (1.0 .- b_E) ./ params.tau_b_E' .- (b_E .* (params.F_STD[params.E_indices] .* r[params.E_indices])) ./ params.tau_STD
            mask = params.F_STD[params.E_indices] .== 0
            if any(mask); d_b_E_mat[mask, :] .= 0.0; end
            d_b_E = vec(d_b_E_mat)
        end
        d_b_I = Float64[]
        if params.n_I > 0 && params.n_b_I > 0 && b_I !== nothing
            d_b_I_mat = (1.0 .- b_I) ./ params.tau_b_I' .- (b_I .* (params.F_STD[params.I_indices] .* r[params.I_indices])) ./ params.tau_STD
            mask = params.F_STD[params.I_indices] .== 0
            if any(mask); d_b_I_mat[mask, :] .= 0.0; end
            d_b_I = vec(d_b_I_mat)
        end
        d_u_d = (-u_d .+ u .+ params.M * p) ./ params.tau_d
        return vcat(d_a_E, d_a_I, d_b_E, d_b_I, d_u_d)
    end
    return rhs
end

# rk4.jl
function rk4_step(f, y, t, dt)
    k1 = dt * f(y, t)
    k2 = dt * f(y + 0.5 * k1, t + 0.5 * dt)
    k3 = dt * f(y + 0.5 * k2, t + 0.5 * dt)
    k4 = dt * f(y + k3, t + dt)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
end

function rk4_solve(f, y0, t_span, dt, t_eval)
    nt = length(t_eval)
    y_out = zeros(length(y0), nt)
    y_out[:, 1] = y0
    y = y0
    for i in 1:(nt-1)
        t = t_eval[i]
        dt_step = t_eval[i+1] - t_eval[i]
        y = rk4_step(f, y, t, dt_step)
        y_out[:, i+1] = y
    end
    return t_eval, y_out
end

# simulate.jl
function solve_ode(T, t_eval, X0, t_ex, u_ex, params; method="RK4", kwargs...)
    rhs = make_rhs(t_ex, u_ex, params)
    if method == "RK4"
        dt = t_eval[2] - t_eval[1]
        t_out, X_out_transposed = rk4_solve(rhs, X0, T, dt, t_eval)
        return t_out, X_out_transposed'
    else
        error("Method $method not implemented. Only 'RK4' is supported.")
    end
end

# algorithms/lyapunov/benettin.jl
function benettin_algorithm(
    X_traj, t_traj, dt, fs, d0, T, lya_dt, params,
    t_ex, u_ex;
    method="RK4", rng=Random.default_rng(), kwargs...
)
    if lya_dt <= 0; error("lya_dt must be positive scalar"); end
    deci_lya = round(Int, lya_dt * fs)
    if deci_lya < 1; error("lya_dt * fs must be >= 1 sample"); end
    tau_lya = dt * deci_lya
    t_lya = t_traj[1:deci_lya:end]
    if !isempty(t_lya) && (t_lya[end] + tau_lya > T[2]); t_lya = t_lya[1:end-1]; end
    nt_lya = length(t_lya)
    local_lya = zeros(nt_lya)
    finite_lya = fill(NaN, nt_lya)
    n_state = size(X_traj, 2)
    pert = randn(rng, n_state)
    pert = pert / norm(pert) * d0
    bounds = get_minmax_range(params)
    min_b, max_b = bounds[:, 1], bounds[:, 2]
    sum_log = 0.0
    rhs = make_rhs(t_ex, u_ex, params)
    for k in 1:nt_lya
        idx_start = (k - 1) * deci_lya + 1
        idx_end = idx_start + deci_lya
        X_start = X_traj[idx_start, :]
        X_end_true = X_traj[idx_end, :]
        X_pert = X_start + pert
        mask = .!isnan.(min_b)
        X_pert[mask] = max.(X_pert[mask], min_b[mask])
        mask = .!isnan.(max_b)
        X_pert[mask] = min.(X_pert[mask], max_b[mask])
        t_seg = t_lya[k]:dt:(t_lya[k] + tau_lya)
        _, X_seg_transposed = rk4_solve(rhs, X_pert, (t_seg[1], t_seg[end]), dt, t_seg)
        X_pert_end = X_seg_transposed[:, end]
        delta = X_pert_end - X_end_true
        d_k = norm(delta)
        local_lya[k] = log(d_k / d0) / tau_lya
        if !isfinite(local_lya[k])
            local_lya = local_lya[1:k-1]; finite_lya = finite_lya[1:k-1]; t_lya = t_lya[1:k-1]
            break
        end
        pert = delta / (d_k + 1e-15) * d0
        if t_lya[k] >= 0
            sum_log += log(d_k / d0)
            finite_lya[k] = sum_log / max(t_lya[k] + tau_lya, eps(Float64))
        end
    end
    finite = finite_lya[.!isnan.(finite_lya)]
    LLE = !isempty(finite) ? finite[end] : 0.0
    return LLE, local_lya, finite_lya, t_lya
end

# srnn.jl
mutable struct Trajectory
    t::Vector{Float64}
    X::Matrix{Float64}
    params::Params
    t_ex::Union{Vector{Float64}, Nothing}
    u_ex::Union{Matrix{Float64}, Nothing}
    solver_options::Dict
    _unpacked_state::Union{Tuple, Nothing}
    _dependent_vars::Union{Tuple, Nothing}
    _sfa_contrib::Union{Matrix{Float64}, Nothing}
    _std_prod::Union{Matrix{Float64}, Nothing}
    function Trajectory(t, X, params, t_ex=nothing, u_ex=nothing, solver_options=Dict())
        new(t, X, params, t_ex, u_ex, solver_options, nothing, nothing, nothing, nothing)
    end
end

function unpack_state_traj(traj::Trajectory); if traj._unpacked_state === nothing; traj._unpacked_state = unpack_trajectory(traj.X', traj.params); end; return traj._unpacked_state; end
function compute_dependent_traj(traj::Trajectory); if traj._dependent_vars === nothing; a_E_ts, a_I_ts, b_E_ts, b_I_ts, u_d_ts = unpack_state_traj(traj); traj._dependent_vars = compute_dependent(a_E_ts, a_I_ts, b_E_ts, b_I_ts, u_d_ts, traj.params); end; return traj._dependent_vars; end
get_r(traj::Trajectory) = compute_dependent_traj(traj)[1]
get_u_d_ts(traj::Trajectory) = unpack_state_traj(traj)[5]

function get_sfa_contrib(traj::Trajectory)
    if traj._sfa_contrib === nothing
        sfa = zeros(traj.params.n, length(traj.t))
        a_E_ts, a_I_ts = unpack_state_traj(traj)[1], unpack_state_traj(traj)[2]
        if traj.params.n_E > 0 && traj.params.n_a_E > 0 && a_E_ts !== nothing
            sfa[traj.params.E_indices, :] .+= traj.params.c_SFA[traj.params.E_indices] .* dropdims(sum(a_E_ts, dims=2), dims=2)
        end
        if traj.params.n_I > 0 && traj.params.n_a_I > 0 && a_I_ts !== nothing
            sfa[traj.params.I_indices, :] .+= traj.params.c_SFA[traj.params.I_indices] .* dropdims(sum(a_I_ts, dims=2), dims=2)
        end
        traj._sfa_contrib = sfa
    end
    return traj._sfa_contrib
end

function get_std_prod(traj::Trajectory)
    if traj._std_prod === nothing
        std_p = ones(traj.params.n, length(traj.t))
        _, _, b_E_ts, b_I_ts = unpack_state_traj(traj)
        if traj.params.n_E > 0 && traj.params.n_b_E > 0 && b_E_ts !== nothing
            std_p[traj.params.E_indices, :] = dropdims(prod(b_E_ts, dims=2), dims=2)
        end
        if traj.params.n_I > 0 && traj.params.n_b_I > 0 && b_I_ts !== nothing
            std_p[traj.params.I_indices, :] = dropdims(prod(b_I_ts, dims=2), dims=2)
        end
        traj._std_prod = std_p
    end
    return traj._std_prod
end

function calculate_lle(traj::Trajectory; dt, fs, d0=1e-3, lya_dt=nothing)
    lya_dt = (lya_dt === nothing) ? 0.5 * traj.params.tau_d : lya_dt
    return benettin_algorithm(traj.X, traj.t, dt, fs, d0, (traj.t[1], traj.t[end]), lya_dt, traj.params, traj.t_ex, traj.u_ex; traj.solver_options...)
end

function plot_trajectory(traj::Trajectory; lle_results=nothing)
    plots_list = []
    if traj.u_ex !== nothing && traj.t_ex !== nothing
        push!(plots_list, plot(traj.t_ex, traj.u_ex', legend=false, ylabel="u_ex"))
    end
    push!(plots_list, plot(traj.t, get_r(traj)', legend=false, ylabel="r (Hz)"))
    push!(plots_list, plot(traj.t, get_u_d_ts(traj)', legend=false, ylabel="u_d"))
    push!(plots_list, plot(traj.t, get_sfa_contrib(traj)', legend=false, ylabel="SFA c*sum(a)"))
    push!(plots_list, plot(traj.t, get_std_prod(traj)', legend=false, ylabel="STD prod(b)", ylims=(0, 1.05)))
    if lle_results !== nothing
        _, local_lya, finite_lya, t_lya = lle_results
        push!(plots_list, plot(t_lya, [local_lya finite_lya], label=["local" "finite"], legend=:best, ylabel="Lyapunov"))
    end
    plot(plots_list..., layout=(length(plots_list), 1), size=(800, 200*length(plots_list)), sharex=true, xlabel="t (s)")
end

struct SRNNModel; params::Params; end
function get_initial_state(model::SRNNModel)
    p = model.params
    a0_E = (p.n_E > 0 && p.n_a_E > 0) ? zeros(p.n_E * p.n_a_E) : Float64[]
    a0_I = (p.n_I > 0 && p.n_a_I > 0) ? zeros(p.n_I * p.n_a_I) : Float64[]
    b0_E = (p.n_E > 0 && p.n_b_E > 0) ? ones(p.n_E * p.n_b_E) : Float64[]
    b0_I = (p.n_I > 0 && p.n_b_I > 0) ? ones(p.n_I * p.n_b_I) : Float64[]
    u_d0 = zeros(p.n)
    return vcat(a0_E, a0_I, b0_E, b0_I, u_d0)
end

function solve(model::SRNNModel, T, t_eval, X0, t_ex, u_ex; kwargs...)
    t_out, X_out = solve_ode(T, t_eval, X0, t_ex, u_ex, model.params; kwargs...)
    return Trajectory(t_out, X_out, model.params, t_ex, u_ex, Dict(kwargs))
end

export SRNNModel, Trajectory, Params, package_params, generate_M_no_iso, get_EI_indices, solve, calculate_lle, plot_trajectory, get_initial_state

end
