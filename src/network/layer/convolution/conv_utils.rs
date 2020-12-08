use ndarray::{par_azip, s, Array, Array1, Array2, Array3, ArrayD, Ix3};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, arr3};

    #[test]
    fn test_unfold1() {
        let input = arr3(&[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]])
        .into_dyn();
        let output = unfold_3d_matrix(1, input, 3, true);
        assert_eq!(
            output,
            arr2(&[
                [1., 2., 3., 5., 6., 7., 9., 10., 11.],
                [2., 3., 4., 6., 7., 8., 10., 11., 12.],
                [5., 6., 7., 9., 10., 11., 13., 14., 15.],
                [6., 7., 8., 10., 11., 12., 14., 15., 16.]
            ])
        );
    }
    #[test]
    fn test_unfold2() {
        let input = arr3(&[[
            [1., 2., 3., 4.],
            [5., 6., 7., 8.],
            [9., 10., 11., 12.],
            [13., 14., 15., 16.],
        ]])
        .into_dyn();
        let output = unfold_3d_matrix(1, input, 2, true);
        assert_eq!(
            output,
            arr2(&[
                [1., 2., 5., 6.],
                [2., 3., 6., 7.],
                [3., 4., 7., 8.],
                [5., 6., 9., 10.],
                [6., 7., 10., 11.],
                [7., 8., 11., 12.],
                [9., 10., 13., 14.],
                [10., 11., 14., 15.],
                [11., 12., 15., 16.]
            ])
        );
    }

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
pub fn add_padding(padding: usize, input: ArrayD<f32>) -> ArrayD<f32> {
    let n = input.ndim(); // 2d or 3d input?
    let shape: &[usize] = input.shape();
    let x = shape[n - 2] + 2 * padding; // calculate the new dim with padding
    let y = shape[n - 1] + 2 * padding; // calculate the new dim with padding

    if n > 4 {
        unimplemented!();
    }

    if n == 4 {
        let batch_size: usize = shape[0];
        let channels: usize = shape[1];
        let mut out: ArrayD<f32> = Array::zeros((batch_size, channels, x, y)).into_dyn();
        par_azip!((mut out_entry in out.outer_iter_mut(), input_entry in input.outer_iter()) {
          let tmp = add_padding(padding, input_entry.into_owned());
          out_entry.assign(&tmp);
        });
        return out;
    }
    let start: isize = padding as isize;
    let x_stop: isize = padding as isize + shape[n - 2] as isize;
    let y_stop: isize = padding as isize + shape[n - 1] as isize;
    let mut out: ArrayD<f32>;
    if n == 2 {
        out = Array::zeros((x, y)).into_dyn();
        out.slice_mut(s![start..x_stop, start..y_stop])
            .assign(&input);
    } else {
        let z = shape[n - 3];
        out = Array::zeros((z, x, y)).into_dyn();
        out.slice_mut(s![.., start..x_stop, start..y_stop])
            .assign(&input);
    }
    out
}

/// We shape the input into a 2d array, so we can apply our vector of (kernel)vectors with a matrix-matrix multiplication.
pub fn shape_into_kernel(x: Array3<f32>) -> Array2<f32> {
    let (shape_0, shape_1, shape_2) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    x.into_shape((shape_0, shape_1 * shape_2)).unwrap()
}

/// For efficiency reasons we handle kernels and images in 2d, here we revert that in order to receive the expected output.
pub fn fold_output(x: Array2<f32>, (num_kernels, n, m): (usize, usize, usize)) -> Array3<f32> {
    // add self.batch_size as additional dim later > for batch processing?
    let (shape_x, shape_y) = (x.shape()[0], x.shape()[1]);
    let output = x.into_shape((num_kernels, n, m));
    match output {
        Ok(v) => v,
        Err(_) => panic!(
            "Got array with shape [{},{}], but expected {}*{}*{} elements.",
            shape_x, shape_y, num_kernels, n, m
        ),
    }
}

/// checked with Rust Playground. Works for quadratic filter and arbitrary 2d/3d images
/// We receive either 2d or 3d input. In order to have one function handling both cases we use ArrayD.
/// We unfold each input image once into a vector of vector (say 2d Array).
/// Each inner vector corresponds to a 2d or 3d subsection of the input image of the same size as a single kernel.
pub fn unfold_3d_matrix(
    in_channels: usize,
    input: ArrayD<f32>,
    k: usize,
    forward: bool,
) -> Array2<f32> {
    assert_eq!(input.ndim(), 3);
    let (len_y, len_x) = (input.shape()[1], input.shape()[2]);

    let mut xx = if forward {
        Array::zeros(((len_y - k + 1) * (len_x - k + 1), k * k * in_channels))
    } else {
        Array::zeros(((len_y - k + 1) * (len_x - k + 1) * in_channels, k * k))
    };

    let mut row_num = 0;

    // windows is not implemented on ArrayD. Here we already know the input dimension, so we just declare it as 3d. No reshaping occurs!
    let x_3d: Array3<f32> = input.into_dimensionality::<Ix3>().unwrap();
    if forward {
        let windows = x_3d.windows([in_channels, k, k]);
        for window in windows {
            let unrolled: Array1<f32> =
                window.into_owned().into_shape(k * k * in_channels).unwrap();
            xx.row_mut(row_num).assign(&unrolled);
            row_num += 1;
        }
    } else {
        // During the backprop part we have to do some reshaping, since we store our kernels as 2d matrix but receive 3d feedback from the next layer.
        // In the 2d case the dimensions match, so we skipped it.
        let windows = x_3d.windows([1, k, k]);
        for window in windows {
            let unrolled: Array1<f32> = window.into_owned().into_shape(k * k).unwrap();
            xx.row_mut(row_num).assign(&unrolled);
            row_num += 1;
        }
    }
    xx
}
