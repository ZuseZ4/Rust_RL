use ndarray::{par_azip, s, Array, Array4, ArrayD, Axis};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};

    #[test]
    fn test_padding1() {
        let input = arr2(&[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]);
        let input = input.into_shape((1, 1, 4, 4)).unwrap();
        let output = add_padding(input, vec![0, 0, 1, 1]);
        let target = arr2(&[
            [0., 0., 0., 0., 0., 0.],
            [0., 1., 2., 3., 4., 0.],
            [0., 5., 6., 7., 8., 0.],
            [0., 9., 10., 11., 12., 0.],
            [0., 13., 14., 15., 16., 0.],
            [0., 0., 0., 0., 0., 0.],
        ]);
        assert_eq!(
            output,
            target.into_shape((1, 1, 6, 6)).unwrap(),
            "{:?}",
            output.shape()
        );
    }
    #[test]
    fn test_padding2() {
        let input = arr3(&[
            [[1., 2., 3.], [5., 6., 7.]],
            [[9., 10., 11.], [13., 14., 15.]],
        ]);
        let input = input.into_shape((1, 2, 2, 3)).unwrap();
        let output = add_padding(input, vec![0, 0, 1, 1]);
        let target = arr3(&[
            [
                [0., 0., 0., 0., 0.],
                [0., 1., 2., 3., 0.],
                [0., 5., 6., 7., 0.],
                [0., 0., 0., 0., 0.],
            ],
            [
                [0., 0., 0., 0., 0.],
                [0., 9., 10., 11., 0.],
                [0., 13., 14., 15., 0.],
                [0., 0., 0., 0., 0.],
            ],
        ]);
        assert_eq!(output, target.into_shape((1, 2, 4, 5)).unwrap());
    }
}

/// We create a new Array of zeros with the size of the original input+padding.
/// Afterwards we copy the original image over to the center of the new image.
/// TODO change to full/normal/...
pub fn add_padding(input: Array4<f32>, padding: Vec<usize>) -> Array4<f32> {
    let n = input.ndim();
    debug_assert!(n == padding.len());
    let shape: &[usize] = input.shape();
    let x = shape[n - 2] + 2 * padding[n - 2]; // calculate the new dim with padding
    let y = shape[n - 1] + 2 * padding[n - 1]; // calculate the new dim with padding

    let batch_size: usize = shape[0];
    let channels: usize = shape[1];

    let mut out: Array4<f32> = Array::zeros((batch_size, channels, x, y));
    let p0 = padding[0] as i32;
    let p1 = padding[1] as i32;
    let p2 = padding[2] as i32;
    let p3 = padding[3] as i32;
    let d0 = shape[0] as i32 - p0;
    let d1 = shape[1] as i32 - p1;
    let d2 = x as i32 - p2;
    let d3 = y as i32 - p3;
    let slice = s![p0..d0, p1..d1, p2..d2, p3..d3];
    out.slice_mut(slice).assign(&input); // we can surely create the slice nicer
    out
}

/// TODO implement this
pub fn remove_padding(padding: usize, input: ArrayD<f32>) -> ArrayD<f32> {
    input
}
