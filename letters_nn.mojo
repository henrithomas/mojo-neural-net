from tensor import Tensor, TensorSpec, TensorShape
from algorithm import parallelize, vectorize
from utils.index import Index
from random import randn, random_si64
from pathlib import path
from math import exp
from python import Python

alias type = DType.float64
alias simdwidth = simdwidthof[type]()
alias mu: Float64 = 0.01
alias mini_batch_size: Int = 50
alias epochs: Int = 1
alias error_target: Float32 = .1
alias input_layer_size: Int = 16
alias hidden_layer_size: Int = 52
alias output_layer_size: Int = 26
alias data_width: Int = 17
alias data_size: Int = 20000
alias training_size: Int = 16000
alias validation_size: Int = 4000

fn matmul_simple(t1: Tensor[type], t2: Tensor[type]) -> Tensor[type]:
    var t_mul: Tensor[type] = Tensor[type](TensorShape(t1.shape()[0],t2.shape()[1]))

    for i in range(t_mul.shape()[0]):
        for j in range(t1.shape()[1]):
            for k in range(t_mul.shape()[1]):
                t_mul[Index(i, k)] += t1[Index(i,j)] * t2[Index(j,k)]
                
    return t_mul   

fn transpose_simple(t: Tensor[type]) -> Tensor[type]:
    var t_transpose: Tensor[type] = Tensor[type](TensorShape(t.shape()[1], t.shape()[0]))

    return t_transpose

fn matmul(t1: Tensor[type], t2: Tensor[type]) -> Tensor[type]:
    var t_mul: Tensor[type] = Tensor[type](TensorShape(t1.shape()[0],t2.shape()[1]))

    @parameter
    fn calc_row(i: Int):
        for j in range(t1.shape()[1]):
            @parameter
            fn dot[simd_width: Int](k: Int):
                t_mul.simd_store[simd_width](
                    i * t_mul.shape()[1] + k, 
                    t_mul.simd_load[simd_width](i * t_mul.shape()[1] + k) + t1[Index(i,j)] * t2.simd_load[simd_width](j * t_mul.shape()[1] + k)
                )
            vectorize[simdwidth, dot](t_mul.shape()[1])

    parallelize[calc_row](t_mul.shape()[0], t_mul.shape()[0])

    return t_mul 


fn dot(t1: Tensor[type], t2: Tensor[type]) -> Float64:
    var vec_dot: Float64 = 0.0
    var temp_vec: Tensor[type] = Tensor[type](t1.shape())
    var sum_vec: Tensor[type] = Tensor[type](simdwidth)
    var sum_simd = SIMD[type, simdwidth](0.0)

    @parameter
    fn compute_mul[simd_width: Int](idx: Int):
        temp_vec.simd_store[simd_width](idx, t1.simd_load[simd_width](idx) * t2.simd_load[simd_width](idx))

    vectorize[simdwidth, compute_mul](t1.shape()[1])

    for i in range(temp_vec.shape()[1]):
        vec_dot += temp_vec[i]

    return vec_dot

# 1.0/(1.0 + exp(-x))
fn sigmoid(z: Tensor[type]) -> Tensor[type]:
    var activations: Tensor[type] = Tensor[type](z.shape())

    @parameter
    fn compute_exp[simd_width: Int](idx: Int):
        activations.simd_store[simd_width](idx, (1 / (1 + exp[type, simd_width](-1 * z.simd_load[simd_width](idx)))))
    
    vectorize[simdwidth, compute_exp](activations.num_elements())
    
    return activations

# sigmoid(z) * (1 - sigmoid(z))
fn sigmoid_prime(a: Tensor[type]) raises -> Tensor[type]:
    var sigma_prime: Tensor[type] = Tensor[type](a.shape()) 

    sigma_prime = a * (1 - a)

    return sigma_prime

fn sigmoid_prime_full(z: Tensor[type]) raises -> Tensor[type]:
    var sigma_prime: Tensor[type] = Tensor[type](z.shape()) 
    let sigmoid_z = sigmoid(z)

    sigma_prime = sigmoid_z * (1 - sigmoid_z)

    return sigma_prime

fn feed_forward():
    return

fn output_error(a_L: Tensor[type], expected: Tensor[type], a_L_prime: Tensor[type]) raises -> Tensor[type]:
    var error_L: Tensor[type] = Tensor[type](a_L.shape())

    error_L = (a_L - expected) * a_L_prime

    return error_L

fn backpropagation(w: Tensor[type], error: Tensor[type], a_prime: Tensor[type]) raises -> Tensor[type]:
    var error_l: Tensor[type] = Tensor[type](TensorShape(mini_batch_size, hidden_layer_size))
    var fake_w_transpose: Tensor[type] = randn[type](TensorSpec(type, w.shape()[1], w.shape()[0]), 0, 1)

    error_l = a_prime * matmul(error, fake_w_transpose)

    return error_l

fn update_weights(inout w: Tensor[type], error: Tensor[type], a_prev: Tensor[type]) raises:
    let fake_a_transpose: Tensor[type] = randn[type](TensorSpec(type, a_prev.shape()[1], a_prev.shape()[0]))
    
    w = w - mu * matmul(fake_a_transpose, error)

fn update_biases(inout b: Tensor[type], error: Tensor[type]) raises:
    b = b - mu * error

fn get_inidices(inout indices: Tensor[DType.int64]):
    for i in range(mini_batch_size):
        indices[Index(0,i)] = random_si64(0,1000)

def get_data():
    var data_input = Tensor[type](TensorSpec(type, data_size, data_width))
    let np = Python.import_module("numpy")
    
    test = np.genfromtxt("letters-data-normalized.txt", np.float64)
    print(test.shape, test[2])

    for i in range(data_size):
        for j in range(data_width):
            data_input[Index(i,j)] = test[i][j].to_float64()
    print(str(data_input))


# input activations
# feed forward
# output error
# backpropagation of error
# gradient descent to update weights
fn main() raises:
    print("learning rate: ", mu, " error target: ", error_target)
    print("mini-batch size: ", mini_batch_size, " number of epochs: ", epochs)
    print("input size: ", input_layer_size," hidden layer size: ", hidden_layer_size, " output size: ", output_layer_size)

    _ = get_data()

    var output_check = Tensor[type](TensorSpec(type, output_layer_size, output_layer_size))

    for i in range(output_check.shape()[0]):
        for j in range(output_check.shape()[1]):
            if(i == j):
                output_check[Index(i,j)] = 0.999
            else:
                output_check[Index(i,j)] = 0.001

    let X_specs = TensorSpec(type, mini_batch_size, input_layer_size)

    let W_l_specs = TensorSpec(type, input_layer_size, hidden_layer_size)
    let W_L_specs = TensorSpec(type, hidden_layer_size, output_layer_size)

    let a_l_specs = TensorSpec(type, mini_batch_size, hidden_layer_size)
    let a_L_specs = TensorSpec(type, mini_batch_size, output_layer_size)

    let B_l_specs = TensorSpec(type, mini_batch_size, hidden_layer_size)
    let B_L_specs = TensorSpec(type, mini_batch_size, output_layer_size)

    var X: Tensor[type] = randn[type](X_specs, 0, 1) # Tensor[type](TensorShape(mini_batch_size, input_layer_size)

    var W_l: Tensor[type] = randn[type](W_l_specs, 0, 1)
    var W_L: Tensor[type] = randn[type](W_L_specs, 0, 1)
    var B_l: Tensor[type] = randn[type](B_l_specs, 0, 1)
    var B_L: Tensor[type] = randn[type](B_L_specs, 0, 1)

    var test_indices: Tensor[DType.int64] = Tensor[DType.int64](TensorShape(1, mini_batch_size)) 
    get_inidices(test_indices)
    print(test_indices)

    # TESTING ONLY
    var fake_expected: Tensor[type] = randn[type](a_L_specs, 1,1) # Tensor[type](TensorShape(mini_batch_size, output_layer_size))# 
    # for i in range(fake_expected.num_elements()):
    #     fake_expected[i] = output_check[Index(19,i)]
 
    @unroll(mini_batch_size)
    for i in range(epochs):
        print("epoch ", (i + 1))

        var z_l = matmul(X, W_l) + B_l
        var a_l = sigmoid(z_l)
        var a_l_prime = sigmoid_prime(a_l)

        var z_L = matmul(a_l, W_L) + B_L
        var a_L = sigmoid(z_L)
        var a_L_prime = sigmoid_prime(a_L)
        var d_L = output_error(a_L, fake_expected, a_L_prime)
        var d_l = backpropagation(W_L, d_L, a_l_prime)

        print(str(a_L))
        print(str(d_L))

        update_weights(W_L, d_L, a_l)
        update_weights(W_l, d_l, X)
        update_biases(B_L, d_L)
        update_biases(B_l, d_l)

        # calculate batch error

    print("done")
