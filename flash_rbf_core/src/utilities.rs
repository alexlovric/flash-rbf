/// Computes the squared Euclidean distance between two vectors of equal length.
///
/// # Arguments
/// * `x0`: A floating point vector.
/// * `x1`: A second floating point vector.
///
/// # Returns
/// The squared Euclidean distance.
pub fn squared_euclidean_distance(x0: &[f64], x1: &[f64]) -> f64 {
    x0.iter()
        .zip(x1.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum()
}

/// Calculates the root-mean-square error (RMSE) between two vectors.
///
/// # Arguments
/// * `y_pred`: A vector of predicted output values.
/// * `y_actual`: A vector of actual output values.
///
/// # Returns
/// The RMSE between the two vectors of output values.
pub fn calculate_rmse(y_pred: &[f64], y_actual: &[f64]) -> f64 {
    squared_euclidean_distance(y_pred, y_actual).sqrt() / y_actual.len() as f64
}

/// Merges the first two pairs of point arrays with the second, if they are unique.
///
/// # Arguments
/// * `x`: The first component array of the first pair.
/// * `y`: The second component array of the first pair.
/// * `x_new`: The first component array of the second pair.
/// * `y_new`: The second component array of the second pair.
pub fn merge_unique_points(
    x: &mut Vec<Vec<f64>>,
    y: &mut Vec<f64>,
    x_new: &[Vec<f64>],
    y_new: &[f64],
) {
    for (i, point) in x_new.iter().enumerate() {
        let is_duplicate = x.iter().any(|unique_point| {
            point.iter().zip(unique_point.iter()).all(|(p, u)| p == u)
        });
        if !is_duplicate {
            x.push(point.clone());
            y.push(y_new[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_rmse() {
        let y_pred = [2.0, 4.0, 6.0, 8.0];
        let y_actual = [1.5, 4.5, 6.5, 8.5];
        let rmse = calculate_rmse(&y_pred, &y_actual);

        assert_eq!(rmse, 0.25);
    }
}
