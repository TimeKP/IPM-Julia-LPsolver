function mehrotra_predictor_corrector_step(As, xs, s, r_p, r_d, r_g; reg_e=1e-7)
    n = length(xs)

    # Predictor step (affine-scaling direction)
    dx_aff, dl_aff, ds_aff = compute_newton_direction(As, xs, s, r_p, r_d, r_g; reg_e=reg_e)

    # Compute affine step size
    mask_dx = dx_aff .< 0
    mask_ds = ds_aff .< 0
    alpha_aff_primal = isempty(dx_aff[mask_dx]) ? 1.0 : min(1.0, minimum(-xs[mask_dx] ./ dx_aff[mask_dx]))
    alpha_aff_dual  = isempty(ds_aff[mask_ds]) ? 1.0 : min(1.0, minimum(-s[mask_ds] ./ ds_aff[mask_ds]))

    # Compute mu and mu_aff
    mu = dot(xs, s) / n
    x_aff = xs + alpha_aff_primal * dx_aff
    s_aff = s + alpha_aff_dual * ds_aff
    mu_aff = dot(x_aff, s_aff) / n

    # Centering parameter
    sigma = (mu_aff / mu)^3

    # Corrected complementarity residual
    r_g_corr = xs .* s + dx_aff .* ds_aff - sigma * mu * ones(n)

    # Corrector step
    dx, dl, ds = compute_newton_direction(As, xs, s, r_p, r_d, r_g_corr; reg_e=reg_e)

    return dx, dl, ds
end

function get_search_direction(As, xs, s, r_p, r_d, r_g; use_mehrotra=false, reg_e=1e-7)
    if use_mehrotra
        return mehrotra_predictor_corrector_step(As, xs, s, r_p, r_d, r_g; reg_e=reg_e)
    else
        return compute_newton_direction(As, xs, s, r_p, r_d, r_g; reg_e=reg_e)
    end
end
