gpuls
=====

A PyTorch implementation of the conjugate gradient method for least-squares
regression, with a Slurm-based shell script wrapper for complex use cases.

Dependencies
------------

* [Slurm](https://slurm.schedmd.com/)
* Python 3 (accessible using `python3`)
* R v3.4 or higher (accessible using `R` and `Rscript`)
* PyTorch v1.1 or higher
* `rhdf5`

Installation
------------

Place the shell scripts in the user's path. Place `gpuls.py` somewhere
accessible to the user, and set the variable `GPULS_PY_PATH` in `gpuls`
accordingly. Set the `NODENAME` variable to the names of the Slurm compute
nodes to send `gpuls` jobs to.

Usage
-----

    $ gpuls
     gpuls v0.2.0
     Usage: gpuls -x X_MATRIX -y Y_VECTOR [-e E_MATRIX] [FLAGS] -o OUTPUT
     Compute the GPU least squares solution for the regression y ~ X, where y is
     a continuous phenotype and X is a (potentially large) matrix.
     -x  X_MATRIX        Path to the X matrix. Valid formats are .txt, where
                         samples are in the rows, .RData/.Rdata where it is
                         stored as a matrix or as the only element in a named
                         list, or in .h5 where it is the only data element.
     -y  Y_MATRIX        Path to the Y matrix. Valid formats are .txt, where
                         samples are in the rows, .RData/.Rdata where it is
                         stored as a vector or as the only element in a named
                         list, or in .h5 where it is the only data element.
    [-e  E_MATRIX]       Optional: a path to the matrix containing Q
                         environmental factors coded as a 0/1 or continuous
                         matrix.
    [-g]                 Optional: if switched, will run SLOWLY but will return
                         the pvalues of significance tests for individual
                         coefficients in the output key "t_p_values"
    [-k]                 Optional: if switched, will not delete the temporary
                         files (A and b) generated.
    [-n]                 Optional: if switched, will include a G x E interaction
                         term in the model, for a total of (Q + 1) * P terms.
    [-t]                 Optional: if switched, will include the E term in the
                         model.
    [-i]                 Optional: if switched, will include an intercept term
                         in the regression model.
    [-q]                 Optional: if switched, quantile normalize the genotype
                         term.
    [-r]                 Optional: if switched, quantile normalize the environment
                         term.
    [-s]                 Optional: if switched, quantile normalize the GxE
                         interaction term.
    [-p]                 Optional: if switched, precondition the inputs (can be slow).
     -o  OUTPUT          Path to the output file. Valid formats are .h5, which
                         will have the keys: "betas" (the vector of coefficients),
                         "ypred" (the predicted y values), "R2", the computed
                         coefficient of determination, "adj_R2", the computed
                         adjusted coefficient of determination, "f_statistic",
                         the F-statistic (where the null model is just an
                         intercept when one is included in the model, or where
                         the null model is one with no terms in the case when
                         there is no intercept in the model), and "f_p_value" for
                         the corresponding p-value. Alternatively, .Rdata where
                         the result will be a named list with the same keys as in
                         .h5 but as names.
     -h                  Print this help.
