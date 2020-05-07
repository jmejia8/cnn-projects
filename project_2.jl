## Classification of MNIST dataset
## with the convolutional neural network know as LeNet5.
## This script also combines various
## packages from the Julia ecosystem  with Flux.
using Flux
using Flux.Data: DataLoader
using Flux.Optimise: Optimiser, WeightDecay
using Flux: onehotbatch, onecold, logitcrossentropy
using Statistics, Random
using Parameters: @with_kw
using Logging: with_logger, global_logger
using TensorBoardLogger: TBLogger, tb_overwrite, set_step!, set_step_increment!
import ProgressMeter
import MLDatasets
import DrWatson: savename, struct2dict
import BSON
using CUDAapi
using Plots

gr(legend=false)
imshow(x; kargs...) = plot(Gray.(x);kargs...)
imshow!(p, x; kargs...) = plot!(p, Gray.(x);kargs...)
# LeNet5 "constructor".
# The model can be adapted to any image size
# and number of output classes.
function LeNet5(; imgsize=(28,28,1), nclasses=10)
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)

    return Chain(
            x -> reshape(x, imgsize..., :),
            Conv((5, 5), imgsize[end]=>6, relu),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            MaxPool((2, 2)),
            x -> reshape(x, :, size(x, 4)),
            Dense(prod(out_conv_size), 120, relu),
            Dense(120, 84, relu),
            Dense(84, nclasses)
          )
end

function get_data(args)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32, dir=args.datapath)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32, dir=args.datapath)

    # MLDatasets uses HWCN format, Flux works with WHCN
    xtrain = permutedims(reshape(xtrain, 28, 28, 1, :), (2, 1, 3, 4))
    xtest = permutedims(reshape(xtest, 28, 28, 1, :), (2, 1, 3, 4))

    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    train_loader = DataLoader(xtrain, ytrain, batchsize=args.batchsize, shuffle=true)
    test_loader = DataLoader(xtest, ytest,  batchsize=args.batchsize)

    return train_loader, test_loader
end

loss(ŷ, y) = logitcrossentropy(ŷ, y)

function eval_loss_accuracy(loader, model, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end

## utility functions

num_params(model) = sum(length, Flux.params(model))

round4(x) = round(x, digits=4)


# arguments for the `train` function
@with_kw mutable struct Args
    η = 3e-4             # learning rate
    λ = 0                # L2 regularizer param, implemented as weight decay
    batchsize = 128      # batch size
    epochs = 20          # number of epochs
    seed = 0             # set seed > 0 for reproducibility
    cuda = true          # if true use cuda (if available)
    infotime = 1 	     # report every `infotime` epochs
    checktime = 5        # Save the model every `checktime` epochs. Set to 0 for no checkpoints.
    tblogger = false      # log training with tensorboard
    savepath = nothing    # results path. If nothing, construct a default path from Args. If existing, may overwrite
    datapath = joinpath(homedir(), "Datasets", "MNIST") # data path: change to your data directory
end

function plotdata()

    args = Args()
    args.seed > 0 && Random.seed!(args.seed)

    ## DATA


    m = LeNet5()

    BSON.@load "runs/lenet_batchsize=128_seed=0_η=0.0003_λ=0/model.bson" model
    Flux.loadparams!(m, params(model))
    l = @layout [a b c ; d e f; g h i]
    p = plot(title="MNIST", layout=l)

    for i = 1:9
        xtest, ytest = MLDatasets.MNIST.testdata(i)
        @show ytest
        x = xtest'
        #x = test_loader.data[1][:,:,1,i]
        v = argmax(m(x))[1] -1
        display(v)
        imshow!(p[i], x, title = string(v))
    end
    p
end

function plotconv()
    the_loss = [2.30320,0.16470,0.10030,0.0770,0.06770,0.05780,0.0510,0.04730,0.04370,0.05120,0.04380,0.04020,0.04110,0.03730,0.0350,0.0382,0.03360,0.03530,0.03840,0.02990,]
	the_acc = [95.21,97.06,97.68,97.86,98.23,98.42,98.58,98.62,98.24,98.62,98.79,98.64,98.83,98.84,98.85,98.89,98.83,98.79,99.0,98.8]

	t = 1:20
	l = @layout [a; b]
	p = plot(layout = l)
	plot!(p[1], t, log.(the_loss), marker = :o, ylabel="Log Loss", xlabel="Epoch")
	plot!(p[2], t, 100 .- the_acc, marker = :o, ylabel="% Error", xlabel="Epoch")

end

function train(; kws...)
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)
    use_cuda = args.cuda && CUDAapi.has_cuda_gpu()
    if use_cuda
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ## DATA
    train_loader, test_loader = get_data(args)
    @info "Dataset MNIST: $(train_loader.nobs) train and $(test_loader.nobs) test examples"

    ## MODEL AND OPTIMIZER
    model = LeNet5() |> device
    @info "LeNet5 model: $(num_params(model)) trainable params"

    ps = Flux.params(model)

    opt = ADAM(args.η)
    if args.λ > 0
        opt = Optimiser(opt, WeightDecay(args.λ))
    end

    ## LOGGING UTILITIES
    if args.savepath == nothing
        experiment_folder = savename("lenet", args, scientific=4,
                    accesses=[:batchsize, :η, :seed, :λ]) # construct path from these fields
        args.savepath = joinpath("runs", experiment_folder)
    end
    if args.tblogger
        tblogger = TBLogger(args.savepath, tb_overwrite)
        set_step_increment!(tblogger, 0) # 0 auto increment since we manually set_step!
        @info "TensorBoard logging at \"$(args.savepath)\""
    end

    function report(epoch)
        train = eval_loss_accuracy(train_loader, model, device)
        test = eval_loss_accuracy(test_loader, model, device)
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        if args.tblogger
            set_step!(tblogger, epoch)
            with_logger(tblogger) do
                @info "train" loss=train.loss  acc=train.acc
                @info "test"  loss=test.loss   acc=test.acc
            end
        end
    end

    ## TRAINING
    @info "Start Training"
    report(0)
    for epoch in 1:args.epochs
        p = ProgressMeter.Progress(length(train_loader))

        for (x, y) in train_loader
            x, y = x |> device, y |> device
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
            ProgressMeter.next!(p)   # comment out for no progress bar
        end

        epoch % args.infotime == 0 && report(epoch)
        if args.checktime > 0 && epoch % args.checktime == 0
            !ispath(args.savepath) && mkpath(args.savepath)
            modelpath = joinpath(args.savepath, "model.bson")
            let model=cpu(model), args=struct2dict(args)
                BSON.@save modelpath model epoch args
            end
            @info "Model saved in \"$(modelpath)\""
        end
    end
end


#plotdata()
plotconv()
