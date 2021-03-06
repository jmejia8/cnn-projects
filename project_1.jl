using Flux, Metalhead, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Metalhead: trainimgs
using Images: channelview, RGB, load
using Statistics: mean
using Base.Iterators: partition
import Random: randperm, seed!
using Plots
using BSON: @save, @load
using Printf: @printf, @sprintf

pyplot(legend=false, reuse=true)

include("architecture.jl")



# Function to convert the RGB image to Float64 Arrays

getarray(X) = Float32.(permutedims(channelview(X), (2, 3, 1)))
accuracy(m, x, y) = mean(onecold(cpu(m(x)), 1:10) .== onecold(cpu(y), 1:10))

function mytrainer(train_set, X_test, Y_test; parm = 0.9)
    max_epochs = 30
    m = testCNN()

    if isfile("tmp_model.bson")
        @load "tmp_model.bson" model
        Flux.loadparams!(m, params(model))
    end

    loss(x, y) = crossentropy(m(x), y)


    # Defining the callback and the optimizer

    evalcb = throttle(() -> @show(accuracy(m, X_test, Y_test)), 10)
    # evalcb = throttle(() -> println("training..."), 10)

    opt = ADAMW()

    # Starting to train models

    accuracies = Float64[]
    the_loss = Float64[]

    #all_loss(x, y) = 

    for t = 1:max_epochs
        Flux.train!(loss, params(m), train_set, opt)
    
        bson_file = "tmp_model_epoch$(t).bson"
        model = m
        @save bson_file model
        
        a = accuracy(m, X_test, Y_test)
        lss = mean(loss(X_test, Y_test))
        println("Epoch ", t, " accuracy: ", a, " loss: ", lss)
    
        push!(accuracies, 1 - a)
        push!(the_loss, lss)
        l = @layout [a; b]
        p = plot(1:t, accuracies, reuse=true, markershape=:o,
                      layout=l,
                      xlabel="Epoch", ylabel="Error",
                      xlim=[1,  max_epochs],
                      #ylim=[0, 1]
                      )
        plot!(p[2], 1:t, the_loss, reuse=true,
              markershape=:o,
              #ylim=[0, 3],
              xlim=[1,  max_epochs],
              xlabel="Epoch",
              ylabel="Loss")
        gui()
    end

    model = m
    @save "tmp_model.bson" model
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

function getSavedModel(file_name)
    
    !isfile(file_name) && error("No existing file.") 
    @load file_name model

    m = testCNN()
    Flux.loadparams!(m, params(model))

    return m

end

function evalModel(file_name)
    m = getSavedModel(file_name)

    println("Loading validation images")
    test = valimgs(CIFAR10)
    println("perro")
    testimgs = [getarray(test[i].img) for i in 1:1000]
    
    println("gato")
    Y_eval = onehotbatch([test[i].ground_truth.class for i in 1:1000], 1:10)  |> gpu 
    X_eval = cat(testimgs..., dims = 4)  |> gpu 

    println("Testing...")
    @show(accuracy(m, X_eval, Y_eval))
    # return imshow(imgs[2])
end

function classify(model_file, image)
    
    m = getSavedModel(model_file)

    x = getarray(image)
    
    xx = cat(x, dims=4)
    vals = m(xx)
    
    
    l = @layout [ a  b ]
    p = plot( image, layout=l, title=CIFAR10.C10Labels[argmax(vals)])
    bar!(p[2], CIFAR10.C10Labels, vals, orientation = :h)


end

function main()


    l = @layout [a;b]
    p = plot( layout=l, reuse=true)
    gui()
    
    println("Loading images")
    train_set, X_test, Y_test = getData()



    # Print the final accuracy

    println("Training...")
    m, accuracies = mytrainer(train_set, X_test, Y_test)
    
end


function createVideo()

    for i = 1:157
        img_path = @sprintf("/home/jesus/Downloads/video/jpg/image-%03d.jpg", i)
        p = classify("cnntest.bson", load(img_path))
        
        fname = @sprintf("img/image-%03d.png", i)
        @show fname
        savefig(p, fname)   
    end


end
#main()

#@time evalModel("tmp_model.bson")
function classifyDummy()

    for fname in ["titanic.jpg","cyber.jpg","delorean.jpg","bird.jpg" ]
        p = classify("bson/cnntest.bson", load(joinpath("input_img", fname)))
        savefig(p, joinpath("outout_img",fname))
        println(fname)
    end

end

main()


