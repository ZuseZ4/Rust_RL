use crate::network::layer::Layer;
use crate::network::optimizer::Optimizer;
use ndarray::{s, Array, Array1, Array2, Array3, ArrayD, Axis, Ix2, Ix3};
use ndarray_rand::rand_distr::Normal; //{StandardNormal,Normal}; //not getting Standardnormal to work. should be cleaner & faster
use ndarray_rand::RandomExt;

pub struct ConvolutionLayer {
    learning_rate: f32,
    kernels: Array2<f32>,
    filter_shape: (usize, usize),
    filter_depth: usize,
    padding: usize,
    bias: Array1<f32>,
    last_input: ArrayD<f32>,
    kernel_updates: Array2<f32>,
    bias_updates: Array1<f32>,
    batch_size: usize,
    num_in_batch: usize,
    weight_optimizer: Box<dyn Optimizer>,
    bias_optimizer: Box<dyn Optimizer>,
}

impl ConvolutionLayer {
    pub fn print_kernel(&self) {
        let n = self.kernels.nrows();
        println!("printing kernels: \n");
        for i in 0..n {
            let arr = self.kernels.index_axis(Axis(0), i);
            println!(
                "{}\n",
                arr.into_shape((self.filter_depth, self.filter_shape.0, self.filter_shape.1))
                    .unwrap()
            );
        }
    }

    pub fn set_kernels(&mut self, kernels: Array2<f32>) {
        self.kernels = kernels;
    }

    pub fn new(
        filter_shape: (usize, usize),
        filter_depth: usize,
        filter_number: usize,
        padding: usize,
        batch_size: usize,
        learning_rate: f32,
        optimizer: Box<dyn Optimizer>,
    ) -> Self {
        assert_eq!(
            filter_shape.0, filter_shape.1,
            "currently only supporting quadratic filter!"
        );
        assert!(filter_shape.0 >= 1, "filter_shape has to be one or greater");
        assert!(
            filter_depth >= 1,
            "filter depth has to be at least one (and equal to img_channels)"
        );
        let kernel_elements = filter_shape.0 * filter_shape.1 * filter_depth;
        let kernels: Array2<f32> = Array::random(
            (filter_number, kernel_elements),
            Normal::new(0.0, 1.0 as f32).unwrap(),
        );
        let mut weight_optimizer = optimizer.clone();
        let mut bias_optimizer = optimizer;
        weight_optimizer.set_input_shape(vec![filter_number, kernel_elements]);
        bias_optimizer.set_input_shape(vec![filter_number]);
        ConvolutionLayer {
            filter_shape,
            learning_rate,
            kernels,
            filter_depth,
            padding,
            bias: Array::zeros(filter_number), //http://cs231n.github.io/neural-networks-2/
            last_input: Array::zeros(0).into_dyn(),
            kernel_updates: Array::zeros((filter_number, kernel_elements)),
            bias_updates: Array::zeros(filter_number),
            batch_size,
            num_in_batch: 0,
            weight_optimizer,
            bias_optimizer,
        }
    }

    // checked with Rust Playground. Quadratic filter and arbitrary 2d/3d images
    fn unfold_matrix(&self, input: ArrayD<f32>, k: usize, forward: bool) -> Array2<f32> {
        let n_dim = input.ndim();
        let (len_y, len_x) = (input.shape()[n_dim - 2], input.shape()[n_dim - 1]);

        let mut xx = if input.ndim() == 3 && !forward {
            Array::zeros(((len_y - k + 1) * (len_x - k + 1) * self.filter_depth, k * k))
        } else {
            Array::zeros(((len_y - k + 1) * (len_x - k + 1), k * k * self.filter_depth))
        };

        let mut row_num = 0;

        if input.ndim() == 2 {
            let x_2d: Array2<f32> = input.into_dimensionality::<Ix2>().unwrap();
            let windows = x_2d.windows([k, k]);
            for window in windows {
                let unrolled: Array1<f32> = window
                    .into_owned()
                    .into_shape(k * k * self.filter_depth)
                    .unwrap();
                xx.row_mut(row_num).assign(&unrolled);
                row_num += 1;
            }
        } else {
            let x_3d: Array3<f32> = input.into_dimensionality::<Ix3>().unwrap();
            if forward {
                let windows = x_3d.windows([self.filter_depth, k, k]);
                for window in windows {
                    let unrolled: Array1<f32> = window
                        .into_owned()
                        .into_shape(k * k * self.filter_depth)
                        .unwrap();
                    xx.row_mut(row_num).assign(&unrolled);
                    row_num += 1;
                }
            } else {
                let windows = x_3d.windows([1, k, k]);
                for window in windows {
                    let unrolled: Array1<f32> = window.into_owned().into_shape(k * k).unwrap();
                    xx.row_mut(row_num).assign(&unrolled);
                    row_num += 1;
                }
            }
        }
        xx
    }

    fn add_padding(&self, input: ArrayD<f32>) -> ArrayD<f32> {
        let shape: &[usize] = input.shape().clone();
        let n = input.ndim();
        let x = shape[n - 2] + 2 * self.padding;
        let y = shape[n - 1] + 2 * self.padding;
        let start: isize = self.padding as isize;
        let x_stop: isize = self.padding as isize + shape[n - 2] as isize;
        let y_stop: isize = self.padding as isize + shape[n - 1] as isize;
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

    fn fold_output(
        &self,
        x: Array2<f32>,
        (num_kernels, n, m): (usize, usize, usize),
    ) -> Array3<f32> {
        x.into_shape((num_kernels, n, m)).unwrap() // add self.batch_size as additional dim later
    }

    fn shape_into_kernel(&self, x: Array3<f32>) -> Array2<f32> {
        let (shape_0, shape_1, shape_2) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        x.into_shape((shape_0, shape_1 * shape_2)).unwrap()
    }
}

impl Layer for ConvolutionLayer {
    fn get_type(&self) -> String {
        format!("Conv")
        //self.print_kernel();
    }

    fn get_num_parameter(&self) -> usize {
        self.kernels.nrows() * self.kernels.ncols() + self.kernels.nrows() // num_kernels * size_kernels + bias
    }

    fn get_output_shape(&self, input_shape: Vec<usize>) -> Vec<usize> {
        let mut res = vec![self.kernels.nrows(), 0, 0];
        let num_dim = input_shape.len();
        res[1] = input_shape[num_dim - 2] - self.filter_shape.0 + 1 + 2 * self.padding;
        res[2] = input_shape[num_dim - 1] - self.filter_shape.1 + 1 + 2 * self.padding;
        res
    }

    fn predict(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        let tmp = self.get_output_shape(input.shape().to_vec());
        let (output_shape_x, output_shape_y) = (tmp[1], tmp[2]);
        let input = self.add_padding(input);

        // prepare input matrix
        let x_unfolded = self.unfold_matrix(input, self.filter_shape.0, true);

        // calculate convolution (=output for next layer)
        let n = self.kernels.nrows();
        let mut prod = Array::zeros((n, output_shape_x * output_shape_y));
        for i in 0..n {
            let kernel = &self.kernels.row(i);
            prod.row_mut(i)
                .assign(&(x_unfolded.dot(kernel) + self.bias[i])); // add bias? -> Done :)
        }

        // reshape product for next layer
        let res = self.fold_output(prod, (self.kernels.nrows(), output_shape_x, output_shape_y));
        res.into_dyn()
    }

    fn forward(&mut self, input: ArrayD<f32>) -> ArrayD<f32> {
        self.last_input = self.add_padding(input.clone());
        self.predict(input)
    }

    fn backward(&mut self, feedback: ArrayD<f32>) -> ArrayD<f32> {
        // precalculate the weight updates for this batch_element

        // prepare input matrix
        let x: Array3<f32> = feedback.into_dimensionality::<Ix3>().unwrap();
        let feedback_as_kernel = self.shape_into_kernel(x.clone());

        //prepare feedback
        let k: usize = (feedback_as_kernel.shape()[1] as f64).sqrt() as usize;
        let input_unfolded = self.unfold_matrix(self.last_input.clone(), k, false);

        //calculate kernel updates
        let prod = input_unfolded.dot(&feedback_as_kernel.t()).t().into_owned();
        let sum: Array1<f32> = x.sum_axis(Axis(1)).sum_axis(Axis(1));

        if self.num_in_batch == 0 {
            self.kernel_updates = prod;
            self.bias_updates = sum;
        } else {
            self.kernel_updates += &prod;
            self.bias_updates += &sum;
        }
        self.num_in_batch += 1;

        if self.num_in_batch == self.batch_size {
            self.num_in_batch = 0;
            let weight_delta = 1. / (self.batch_size as f32) * self.kernel_updates.clone();
            let bias_delta = 1. / (self.batch_size as f32) * self.bias_updates.clone();
            self.kernels -= &(self.weight_optimizer.optimize2d(weight_delta) * self.learning_rate);
            self.bias -= &(self.bias_optimizer.optimize1d(bias_delta) * self.learning_rate);
            //println!("{:}", self.kernels);
            //println!("{:}", self.kernels.index_axis(Axis(0),3));
            //println!("bias {:}", self.bias);
        }

        // calc feedback for the previous layer:
        // use fliped kernel vectors here
        // https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
        // return that

        Array::zeros(self.last_input.shape()).into_dyn()
    }
}
