using Pkg
Pkg.activate(".")

using SRNN
using Random

function main()
    rng = MersenneTwister(1)

    # Network
    n = 10
    mean_in_out_degree = 5
    density = mean_in_out_degree / (n - 1)
    sparsity = 1 - density
    EI = 0.7
    scale = 0.5 / 0.79782
    w = Dict("EE" => scale * 1, "EI" => scale * 1, "IE" => scale * 1, "II" => scale * 0.5, "selfE" => 0.0, "selfI" => 0.0)

    M, EI_vec = generate_M_no_iso(n, w, sparsity, EI, rng=rng)
    E_idx, I_idx, n_E, n_I = get_EI_indices(EI_vec)

    # Time
    fs = 1000.0
    dt = 1.0 / fs
    T = (-10.0, 10.0)
    t = range(T[1], T[2], step=dt)
    nt = length(t)

    # External input u_ex (n, nt)
    u_ex = zeros(n, nt)
    stim_b0 = 0.5
    amp = 0.5
    dur = 3
    n_dur = round(Int, fs * dur)
    off = round(Int, -T[1] * fs)
    
    t_sin = t[1:n_dur]
    f_sin = ones(size(t_sin)) * 1.0
    
    idx1 = off + round(Int, fs * 6)
    u_ex[1, idx1 : idx1 + n_dur - 1] = stim_b0 .+ amp * sign.(sin.(2 * pi * f_sin .* t_sin))
    idx2 = off + round(Int, fs * 1)
    u_ex[1, idx2 : idx2 + n_dur - 1] = stim_b0 .+ amp * (-cos.(2 * pi * f_sin .* t_sin))

    # DC ramp then constant
    DC = 0.1
    ramp_duration = 5.0
    ramp_mask = t .<= (T[1] + ramp_duration)
    ramp_profile = range(0.0, stop=DC, length=sum(ramp_mask))
    u_dc = fill(DC, nt)
    u_dc[ramp_mask] = ramp_profile
    u_ex = u_ex .+ u_dc'

    # Params
    tau_STD = 0.5
    n_a_E = 3
    n_a_I = 0
    n_b_E = 1
    n_b_I = 0

    tau_a_E = (n_a_E > 0) ? 10 .^ range(log10(0.3), stop=log10(15.0), length=n_a_E) : nothing
    tau_a_I = nothing
    tau_b_E = (n_b_E == 1) ? (4 * tau_STD) : 10 .^ range(log10(0.6), stop=log10(9.0), length=n_b_E)
    tau_b_I = nothing
    tau_d = 0.025

    c_SFA = (n_a_E > 0) ? ( (EI_vec .== 1) .* (1.0 / n_a_E) ) : zeros(n)
    F_STD = Float64.(EI_vec .== 1)

    params = package_params(
        n_E, n_I, E_idx, I_idx,
        n_a_E, n_a_I, n_b_E, n_b_I,
        tau_a_E, tau_a_I,
        (n_b_E > 0) ? (n_b_E == 1 ? [tau_b_E] : tau_b_E) : nothing,
        tau_b_I,
        tau_d, n, M, c_SFA, F_STD, tau_STD, EI_vec
    )

    model = SRNNModel(params)
    X0 = get_initial_state(model)

    trajectory = SRNN.solve(model, T, collect(t), X0, collect(t), u_ex, method="RK4")

    lle_results = calculate_lle(trajectory, dt=dt, fs=fs, d0=1e-3)
    LLE, _, finite_lya, _ = lle_results
    last_finite = filter(!isnan, finite_lya)
    println(Dict("LLE" => LLE, "last_finite" => isempty(last_finite) ? nothing : last_finite[end]))

    try
        display(plot_trajectory(trajectory, lle_results=lle_results))
        println("Plot displayed. Press Enter to exit...")
        readline()
    catch e
        println("Plotting failed: $e")
    end
end

main()
