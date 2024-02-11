from tensor import Tensor, TensorSpec, TensorShape
from algorithm import parallelize, vectorize
from utils.index import Index
from random import randn
from pathlib import path
from math import exp

alias type = DType.float32
alias simdwidth = simdwidthof[type]()
alias mu: Float32 = 0.01
alias mini_batch_size: Int = 50
alias error_target: Float32 = .1
alias input_layer_size: Int = 16
alias hidden_layer_size: Int = 52
alias output_layer_size: Int = 26

fn matmul_simple(t1: Tensor[type], t2: Tensor[type]) -> Tensor[type]:
    var t_mul: Tensor[type] = Tensor[type](TensorShape(t1.shape()[0],t2.shape()[1]))

    for i in range(t_mul.shape()[0]):
        for j in range(t1.shape()[1]):
            for k in range(t_mul.shape()[1]):
                t_mul[Index(i, k)] += t1[Index(i,j)] * t2[Index(j,k)]
                
    return t_mul   

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


fn dot(t1: Tensor[type], t2: Tensor[type]) -> Float32:
    var vec_dot: Float32 = 0.0
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

fn sigmoid_prime(z: Tensor[type]) -> Tensor[type]:
    var sigma_prime: Tensor[type] = Tensor[type](z.shape()) 
    return sigma_prime

fn feed_forward():
    return

# input activations
# feed forward
# output error
# backpropagation of error
# gradient descent to update weights
fn main() raises:
    print("learning rate:", mu, "error target", error_target)
    print("mini-batch size", mini_batch_size, "hidden layer size:", hidden_layer_size)

    let X_specs = TensorSpec(type, mini_batch_size, input_layer_size)

    let W_l_specs = TensorSpec(type, input_layer_size, hidden_layer_size)
    let W_L_specs = TensorSpec(type, hidden_layer_size, output_layer_size)

    let a_l_specs = TensorSpec(type, mini_batch_size, hidden_layer_size)
    let a_L_specs = TensorSpec(type, mini_batch_size, output_layer_size)

    let B_l_specs = TensorSpec(type, mini_batch_size, hidden_layer_size)
    let B_L_specs = TensorSpec(type, mini_batch_size, output_layer_size)
    
    # var X: Tensor[type] = Tensor[type](X_specs)
    var X: Tensor[type] = randn[type](X_specs, 0, 1)

    var W_l: Tensor[type] = randn[type](W_l_specs, 0, 1)
    var W_L: Tensor[type] = randn[type](W_L_specs, 0, 1)
    var B_l: Tensor[type] = randn[type](B_l_specs, 0, 1)
    var B_L: Tensor[type] = randn[type](B_L_specs, 0, 1)

    # print(str(X))
    # print(str(W_l))

    for i in range(1):
        var z_l = matmul(X, W_l) + B_l
        var a_l = sigmoid(z_l)
        var z_L = matmul(a_l, W_L) + B_L
        var a_L = sigmoid(z_L)

    print("done")
