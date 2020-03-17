using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
import Random: randperm

include("architecture.jl")



# Function to convert the RGB image to Float64 Arrays

getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))

function mytrainer!(train_set, X_test, Y_test)
    m = testCNN()

    loss(x, y) = crossentropy(m(x), y)

    accuracy(x, y) = mean(onecold(cpu(m(x)), 1:10) .== onecold(cpu(y), 1:10))

    # Defining the callback and the optimizer

    evalcb = throttle(() -> @show(accuracy(X_test, Y_test)), 1)

    opt = ADAGrad()

    # Starting to train models

    Flux.train!(loss, params(m), train_set, opt, cb = evalcb)

end


function main()
    # Fetching the train and validation data and getting them into proper shape
    N_train = 5000
    N_test = 1000
    I = randperm(50000)
    train_set_idx = I[1:N_train]
    test_set_idx = reverse(I)[1:N_test]

    X = trainimgs(CIFAR10)
    
    imgs = [getarray(x.img) for x in X]
    labels = onehotbatch([x.ground_truth.class for x in X], 1:10)
    

    train_set = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(train_set_idx, 4)])
    
    X_test = cat(imgs[test_set_idx]..., dims = 4) |> gpu
    Y_test = labels[:, test_set_idx] |> gpu



    mytrainer!(train_set, X_test, Y_test)

    # Fetch the test data from Metalhead and get it into proper shape.
    # CIFAR-10 does not specify a validation set so valimgs fetch the testdata instead of testimgs

    test = valimgs(CIFAR10)

    testimgs = [getarray(test[i].img) for i in 1:100]
    testY = onehotbatch([test[i].ground_truth.class for i in 1:100], 1:10) |> gpu
    testX = cat(testimgs..., dims = 4) |> gpu

    # Print the final accuracy

    @show(accuracy(testX, testY))
end