use nalgebra::DMatrix;
use rand::distributions::Standard;
use rand::Rng;
use serde::Deserialize;
use std::env;
use std::{error::Error, fs::File, println, process};

#[derive(Debug, Deserialize)]
struct Record {
    x: f32,
    y: f32,
}

const NCOLS: usize = 2;
const N_SAMPLES: usize = 60;

fn read_csv() -> Result<Vec<f32>, Box<dyn Error>> {
    let current_dir = env::current_dir()?;
    let file = File::open(current_dir.join("data/simple_regression.csv"))?;
    let mut rdr = csv::ReaderBuilder::new().delimiter(b',').from_reader(file);
    let records: Vec<Record> = rdr.deserialize().collect::<Result<Vec<_>, _>>()?;
    let data: Vec<f32> = records
        .into_iter()
        .flat_map(|record| vec![record.x, record.y])
        .collect();
    Ok(data)
}

fn linear_regression(data: Vec<f32>) -> Result<(), Box<dyn Error>> {
    let matrix: DMatrix<f32> = DMatrix::from_row_slice(data.len() / NCOLS, NCOLS, &data);
    let nrows = matrix.nrows();

    let mut m_train = matrix.view((0, 0), (nrows - N_SAMPLES, NCOLS)).into_owned();
    let mut m_test = matrix
        .view((nrows - N_SAMPLES, 0), (N_SAMPLES, NCOLS))
        .into_owned();

    // Split into train and test
    let (x_train, y_train) = m_train.columns_range_pair_mut(..NCOLS - 1, NCOLS - 1..);
    let (x_test, y_test) = m_test.columns_range_pair_mut(..NCOLS - 1, NCOLS - 1..);
    let x_train = x_train.into_owned();
    let y_train = y_train.into_owned();
    let _x_test = x_test.into_owned();
    let _y_test = y_test.into_owned();

    // Bias
    let mut bias = DMatrix::from_vec(1, 1, vec![1.0; 1]);

    // Random weights
    let mut weights: DMatrix<f32> = DMatrix::from_vec(
        NCOLS - 1,
        1,
        rand::thread_rng()
            .sample_iter(Standard)
            .take(NCOLS - 1)
            .collect(),
    ) * (2.0 / (NCOLS - 1) as f32).sqrt();

    // Learning Rate
    let lr = 0.0001;

    // Number of epochs
    for _n in 0..100 {
        // Predict y
        let mut y_pred = x_train.clone_owned() * weights.clone_owned();
        // Apply bias
        for i in 0..y_pred.nrows() {
            let row = y_pred.row(i).into_owned();
            y_pred.row_mut(i).copy_from(&(row + bias.clone_owned()));
        }
        // Calculate error
        let error = y_pred - y_train.clone_owned();
        // Calculate gradient
        let gradient = x_train.clone_owned().transpose() * error.clone();
        // Adjust weights
        weights -= lr * gradient.clone();
        // Adjust bias
        let sum = (lr * error).sum();
        let sum_matrix = DMatrix::from_vec(1, 1, vec![sum; 1]);
        bias -= &sum_matrix;
    }

    // By default, using this dataset the result should be: 43.436935 * x + -1.2420014
    println!("{:?} * x + {:?}", weights[0], bias[0]);

    Ok(())
}

fn main() {
    match read_csv() {
        Ok(data) => {
            if let Err(err) = linear_regression(data) {
                println!("error running example: {}", err);
                process::exit(1);
            }
        }
        Err(err) => {
            println!("error reading csv: {}", err);
            process::exit(1);
        }
    }
}
