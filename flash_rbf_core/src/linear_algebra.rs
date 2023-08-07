/// Linear solving given design matrix and right hand side.
///
/// # Arguments
/// * `mat`: Design matrix.
/// * `rhs`: Right hand side.
///
/// # Returns
/// Solution of solved system.
pub fn naive_linear_solver(mat: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, String> {
    let n = mat.len();
    let m = rhs.len();
    if n != m {
        return Err(String::from(
            "Incompatible design matrix and right hand side sizes!",
        ));
    }

    let mut lu = mat.to_vec();
    let mut p = (0..n).collect::<Vec<usize>>();

    // Perform LU decomposition
    for k in 0..n - 1 {
        let mut max_row = k;
        for i in k + 1..n {
            if lu[i][k].abs() > lu[max_row][k].abs() {
                max_row = i;
            }
        }
        if lu[max_row][k] == 0.0 {
            return Err(String::from("Zero obtained in LU[max_row][k]!"));
        }
        if max_row != k {
            lu.swap(k, max_row);
            p.swap(k, max_row);
        }
        for i in k + 1..n {
            let factor = lu[i][k] / lu[k][k];
            lu[i][k] = factor;
            for j in k + 1..n {
                lu[i][j] -= factor * lu[k][j];
            }
        }
    }

    // Solve Ly = Pb
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for (j, yy) in y.iter().enumerate().take(i) {
            sum += lu[i][j] * yy;
        }
        y[i] = rhs[p[i]] - sum;
    }

    // Solve Ux = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for (j, xx) in x.iter().enumerate().take(n).skip(i + 1) {
            sum += lu[i][j] * xx;
        }
        x[i] = (y[i] - sum) / lu[i][i];
    }

    Ok(x)
}
