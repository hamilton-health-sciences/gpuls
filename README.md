gpuls
=====

A PyTorch implementation of the conjugate gradient method for least-squares
regression, with a Slurm-based shell script wrapper for complex use cases.

Dependencies
------------

* [Slurm](https://slurm.schedmd.com/)
* Python 3 (accessible using `python3`)
* R v3.4 or higher (accessible using `R` and `Rscript`)
* Python packages: PyTorch v1.1 or higher, `rpy2`
* R packages: `rhdf5`

Installation
------------

Place the shell scripts in the user's path. Place `gpuls.py` somewhere
accessible to the user, and set the variable `GPULS_PY_PATH` in `gpuls`
accordingly. Set the `NODENAME` variable to the names of the Slurm compute
nodes to send `gpuls` jobs to.

Example
-------

The following is an example GPULS command using the provided simulated data:

    $ gpuls -i \
      -x Simulated_Data/sample_data_single_block.RData \
      -e Simulated_Data/sample_data_environmental_factor.RData \
      -y Simulated_Data/sample_data_phenotype.RData \
      -o Simulated_Data/gpuls_results.RData

This will produce regression results in the RData file
`Simulated_Data/gpuls_results.RData`, equivalent to a call like
`lm(y ~ X * E)` in R.

Usage
-----

### Data formats

Input files can be formatted as `.txt` (no column names permitted),
`.Rdata`/`.RData` where the the matrix is stored as the only element in a named
list, or as an `.h5` file where the matrix is the only data element.

Missing values are not allowed.

### Execution procedure

First, the script computes `A = Xhat^T . Xhat`, where `Xhat` is the predictor
matrix generated from the inputs `X` (and optionally `E`). The script then uses
the size of `A` to estimate the number of GPUs required to complete the
regression. The script then calls the conjugate gradient descent routine,
splitting `A` across multiple GPUs if necessary. Once converged, the solution
is copied back to the CPU and written to the filesystem.

Intermediate files are stored in `.h5` format, and can be preserved by passing
the `-k` flag to GPULS.

### Memory considerations

All matrix entries are stored as 64-bit (8 byte) floats.

Matrix size limits depend generally on the total amount of CPU and GPU RAM
available on your system. In general, sufficient CPU RAM is required to compute
`Xhat^T . Xhat` and sufficient GPU RAM is required to store `Xhat^T . Xhat` in
block form, possibly split across multiple GPUs. Thus, memory usage scales
approximately quadratically with the number of predictors (i.e., `P * (E + 1)`
where `P` is the number of columns in `X` and `E` is the number of environmental
factors in the regression or 0 if not estimating any interaction terms).

### Full documentation string

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
