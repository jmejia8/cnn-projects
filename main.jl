using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
import Random: randperm, seed!

include("architecture.jl")



# Function to convert the RGB image to Float64 Arrays

getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))
accuracy(m, x, y) = mean(onecold(cpu(m(x)), 1:10) .== onecold(cpu(y), 1:10))

function mytrainer(train_set, X_test, Y_test; parm = 0.9)
    max_epochs = 30
    m = testCNN()

    loss(x, y) = crossentropy(m(x), y)


    # Defining the callback and the optimizer

    evalcb = throttle(() -> @show(accuracy(m, X_test, Y_test)), 10)
    # evalcb = throttle(() -> println("training..."), 10)

    opt = ADAMW()

    # Starting to train models

    accuracies = Float64[]
    for t = 1:max_epochs
        Flux.train!(loss, params(m), train_set, opt, cb = evalcb)
        push!(accuracies, accuracy(m, X_test, Y_test))
    end
    m
end

function main()

    seed!(1)

    # Fetching the train and validation data and getting them into proper shape
    N_train = 49000
    N_test = 10000
    I = 1:50000 #randperm(50000)
    train_set_idx = I[1:N_train]
    test_set_idx = reverse(I)[1:N_test]

    X = trainimgs(CIFAR10)
    
    imgs = [getarray(x.img) for x in X]
    labels = onehotbatch([x.ground_truth.class for x in X], 1:10)
    

    X_train = cat(imgs[train_set_idx]..., dims = 4) |> gpu
    Y_train = labels[:, train_set_idx] |> gpu 
    
    X_test = cat(imgs[test_set_idx]..., dims = 4) |> gpu
    Y_test = labels[:, test_set_idx] |> gpu

    train_set = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(train_set_idx, 10)])



    test = valimgs(CIFAR10)

    testimgs = [getarray(test[i].img) for i in 1:100]
    Y_eval = onehotbatch([test[i].ground_truth.class for i in 1:100], 1:10) |> gpu
    X_eval = cat(testimgs..., dims = 4) |> gpu

    # Print the final accuracy

    m = mytrainer(train_set, X_test, Y_test)
    @show(accuracy(m, X_eval, Y_eval))
    # return imshow(imgs[2])
end

main()