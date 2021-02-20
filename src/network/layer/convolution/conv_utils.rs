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
        ])
        .into_dyn();
        let output = add_padding(1, input);
        assert_eq!(
            output,
            arr2(&[
                [0., 0., 0., 0., 0., 0.],
                [0., 1., 2., 3., 4., 0.],
                [0., 5., 6., 7., 8., 0.],
                [0., 9., 10., 11., 12., 0.],
                [0., 13., 14., 15., 16., 0.],
                [0., 0., 0., 0., 0., 0.]
            ])
            .into_dyn()
        );
    }
    #[test]
    fn test_padding2() {
        let input = arr3(&[
            [[1., 2., 3.], [5., 6., 7.]],
            [[9., 10., 11.], [13., 14., 15.]],
        ])
        .into_dyn();
        let output = add_padding(1, input);
        assert_eq!(
            output,
            arr3(&[
                [
                    [0., 0., 0., 0., 0.],
                    [0., 1., 2., 3., 0.],
                    [0., 5., 6., 7., 0.],
                    [0., 0., 0., 0., 0.]
                ],
                [
                    [0., 0., 0., 0., 0.],
                    [0., 9., 10., 11., 0.],
                    [0., 13., 14., 15., 0.],
                    [0., 0., 0., 0., 0.]
                ]
            ])
            .into_dyn()
        );
    }
}

/// We create a new Array of zeros with the size of the original input+padding.
/// Afterwards we copy the original image over to the center of the new image.
/// TODO change to full/normal/...
pub fn add_padding(input: Array4<f32>, padding: Vec<usize>) -> Array4<f32> {
    let n = input.ndim();
    let shape: &[usize] = input.shape();
    let x = shape[n - 2] + 2 * padding[n - 2]; // calculate the new dim with padding
    let y = shape[n - 1] + 2 * padding[n - 1]; // calculate the new dim with padding

    let batch_size: usize = shape[0];
    let channels: usize = shape[1];

    let mut out: Array4<f32> = Array::zeros((batch_size, channels, x, y));
    let p2 = padding[2] as i32;
    let p3 = padding[3] as i32;
    let slice = s![0..-0, 0..-0, p2..-p2, p3..-p3];
    out.slice_mut(slice).assign(&input); // we can surely create the slice nicer
    out
}

/// TODO implement this
pub fn remove_padding(padding: usize, input: ArrayD<f32>) -> ArrayD<f32> {
    input
}
