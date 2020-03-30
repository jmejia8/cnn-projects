using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Metalhead: trainimgs
using Images: channelview
using Statistics: mean
using Base.Iterators: partition
import Random: randperm, seed!
using Plots
using BSON: @save, @load

pyplot(legend=false)

include("architecture.jl")



# Function to convert the RGB image to Float64 Arrays

getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))
accuracy(m, x, y) = mean(onecold(cpu(m(x)), 1:10) .== onecold(cpu(y), 1:10))

function mytrainer(train_set, X_test, Y_test; parm = 0.9)
    max_epochs = 30
    m = testCNN()

    if isfile("mymodel.bson")
        @load "mymodel.bson" weights
        Flux.loadparams!(m, weights)
    end

    loss(x, y) = crossentropy(m(x), y)


    # Defining the callback and the optimizer

    evalcb = throttle(() -> @show(accuracy(m, X_test, Y_test)), 10)
    # evalcb = throttle(() -> println("training..."), 10)

    opt = ADAMW()

    # Starting to train models

    accuracies = Float64[]
    for t = 1:max_epochs
        Flux.train!(loss, params(m), train_set, opt)
        a = accuracy(m, X_test, Y_test)
        println("Epoch ", t, " accuracy: ", a)

        push!(accuracies, a)
    end

    weights = params(cpu(m))
    @save "mymodel.bson" weights
    m, accuracies
end

function getData()
	X = trainimgs(CIFAR10)
	imgs = [getarray(X[i].img) for i in 1:50000]
	labels = onehotbatch([X[i].ground_truth.class for i in 1:50000],1:10)
	train = gpu.([(cat(imgs[i]..., dims = 4), labels[:,i]) for i in partition(1:49000, 100)])
	valset = collect(49001:50000)
	valX = cat(imgs[valset]..., dims = 4) |> gpu
	valY = labels[:, valset] |> gpu
	return train, valX, valY	

end

function main()

    seed!(1)
	println("Loading images")
	train_set, X_test, Y_test = getData()

    println("Loading validation images")
    test = valimgs(CIFAR10)

    testimgs = [getarray(test[i].img) for i in 1:100]
    Y_eval = onehotbatch([test[i].ground_truth.class for i in 1:100], 1:10)  |> gpu 
    X_eval = cat(testimgs..., dims = 4)  |> gpu 

    # Print the final accuracy

    println("Training...")
    m, accuracies = mytrainer(train_set, X_test, Y_test)
    
    println("Testing...")
    @show(accuracy(m, X_eval, Y_eval))
    plot(1:length(accuracies), accuracies, markershape = :o)
    # return imshow(imgs[2])
end


main()



