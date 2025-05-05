function convert_to_standard(A, b, c, hi, lo)
    INFINITY::Float64 = 1.0e308 # Threshold for detecting unbounded variables
    m = size(A,1)
    n = length(c)

    # Initialize standard-form blocks
    As = sparse(A)
    bs = copy(b)
    cs = copy(c)
    
    # Lower bounds
    if (any(lo .!=0.0))
        bs = b - A*lo
        hi = hi - lo
        As = sparse(A)
    end

    # Upper bounds
    # Convert finite upper bounds into inequality constraints using slack variables
    if (any(hi .!= INFINITY))
        loc_hi = findall(hi .!= INFINITY);
        val_hi = hi[loc_hi];
        count_hi = length(loc_hi);

        # Construct augmented matrix system for inequality: x <= hi --> x + s = hi
        Aug_1 = zeros(m,count_hi);
        Aug_2 = Matrix{Float64}(I, count_hi, count_hi)
        Aug_3 = zeros(count_hi,n);
        Aug_3[:,loc_hi] = Aug_2;

        # Final augmented matrix: [A 0; I_select I] and vector updates
        As = [A Aug_1;Aug_3 Aug_2];
        As = sparse(As);
        bs = vec([b;val_hi]);
        cs = vec([c; zeros(count_hi,1)]);
    end

    return As, bs, cs
end
