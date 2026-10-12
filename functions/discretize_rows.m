function r = discretize_rows(y, rows)
[~, r] = min(abs(y(:) - rows(:).'), [], 2);
end
