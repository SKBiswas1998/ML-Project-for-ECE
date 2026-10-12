
function [G, countG] = lead_eps_optimized(h1, u, eta, st, endcount1, endcount2)

    countG = 0;

    step = 100; % large so that only one energy

    countk = 1000; % no of iterations

    err = 1e-9; % convergence error

    ud = ctranspose(u);

    

    E1_values = st:step:st;

    E2_values = E1_values + 1i * eta;

    E3_values = bsxfun(@times, E2_values, eye(endcount1)); % Avoid loop for efficiency

    %{

    E3_values = zeros(endcount1, endcount1, numel(E1_values)); % Preallocate E3_values

    for idxE = 1:numel(E1_values)

        E3_values(:,:,idxE) = E2_values(idxE) * eye(endcount1);

    end

    %}

 

 

    G = zeros(endcount1, endcount1, numel(E1_values)); % Preallocate G

    for idxE = 1:numel(E1_values)

        countG = countG + 1;

        E3 = E3_values(:,:,idxE);

        

        alpha_prev = u;

        beta_prev = ud;

        epss_prev = h1;

        eps_prev = h1;

        

        for k = 1:countk

            inv_diff = inv(E3 - eps_prev);

            

            epss_new = epss_prev + alpha_prev * inv_diff * beta_prev;

            eps_new = eps_prev + beta_prev * inv_diff * alpha_prev + alpha_prev * inv_diff * beta_prev;

            alpha_new = alpha_prev * inv_diff * alpha_prev;

            beta_new = beta_prev * inv_diff * beta_prev;

            

            if max(max(abs(alpha_new))) < err && max(max(abs(beta_new))) < err

                aa = epss_new;

                break;

            end

            

            epss_prev = epss_new;

            eps_prev = eps_new;

            alpha_prev = alpha_new;

            beta_prev = beta_new;

        end

        

        G(:,:,countG) = inv(E3 - aa);

    end

end