module MyAutoEncoder
# Encode MNIST images as compressed vectors that can later be decoded back into
# images.

using Flux, Flux.Data.MNIST
using Flux, Flux.Data.FashionMNIST
using Flux: @epochs, onehotbatch, mse, throttle
using Base.Iterators: partition
using Parameters: @with_kw
using CUDAapi, BSON
using DelimitedFiles
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

@with_kw mutable struct Args
    lr::Float64 = 1e-3		# Learning rate
    epochs::Int = 30		# Number of epochs
    N::Int = 2			# Size of the encoding
    batchsize::Int = 1000	# Batch size for training
    sample_len::Int = 20 	# Number of random digits in the sample image
    throttle::Int = 5		# Throttle timeout
    p_noise::Float64 = 0.0
    savepath::String = "./"
end

function add_noise(c_image, p)
        if p == 0.0
            return c_image
        end
        c_image_noise = copy(c_image)
        I = rand(prod(size(c_image))) .< p
        c_image_noise[I] = rand(sum(I))
        return c_image_noise
end

function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in 1:length(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
        Y_batch[:, :, :, i] = Float32.(Y[idxs[i]])
    end
    return (X_batch, Y_batch)
end


function get_processed_data(args)
    # Loading Images
    imgs = FashionMNIST.images()
    #Converting image of type RGB to float 
    imgs = channelview.(imgs)
    # Partition into batches of size 1000 
    original_images = Array[]
    noisy_images = Array[]
    
    for imgs in partition(imgs, args.batchsize)
        c_image = float(hcat(vec.(imgs)...)) 
        push!(original_images, c_image)
        
        c_image_noise = add_noise(c_image, args.p_noise)
        push!(noisy_images, c_image_noise)

    end

    # original_images = gpu.(original_images)
    train_data = zip(noisy_images, original_images)
    train_data = gpu.(train_data)
    return train_data
end

function get_model(args)
    encoder = Chain(Dense(28^2, 12^2, relu), Dense(12^2, args.N, sigmoid)) |> gpu
    decoder = Chain(Dense(args.N, 12^2, relu), Dense(12^2, 28^2, leakyrelu)) |> gpu 
    m = Chain(encoder, decoder)
    m
end

function total_loss(loss, XY)
    s = 0.0
    for i in length(XY)
        s += loss(XY[i][1], XY[i][2])
    end

    s / length(XY)
end

function train(; kws...)
    args = Args(; kws...)	

    bson_name = "fashion_mnist_ae_N"* string(args.N)* "_p"* string(args.p_noise)* ".bson"
    
    if isfile(bson_name)
        @info "Trained NET..."
        model = get_model(args)
    
        # Loading the saved parameters
        BSON.@load joinpath(args.savepath, bson_name) params
    
        # Loading parameters onto the model
        Flux.loadparams!(model, params)
        return model, args
    end


    train_data = get_processed_data(args)
    @info("Constructing model......")
    # You can try to make the encoder/decoder network larger
    # Also, the output of encoder is a coding of the given input.
    # In this case, the input dimension is 28^2 and the output dimension of
    # encoder is 32. This implies that the coding is a compressed representation.
    # We can make lossy compression via this `encoder`.

    # Defining main model as a Chain of encoder and decoder models
    m = get_model(args)
    @info("Training model.....")
    loss(x, y) = mse(m(x), y)
    ## Training
    evalcb = throttle(() -> @show( total_loss(loss, train_data)  ), args.throttle)
    opt = ADAM(args.lr)
	
    err = Float64[]
    for current_epoch in 1:args.epochs 
        Flux.train!(loss, Flux.params(m), (train_data), opt, cb = evalcb)
        @info "Epoch $current_epoch"
        push!(err, total_loss(loss, train_data))
        @show err[current_epoch]
    end

    BSON.@save joinpath(args.savepath, bson_name) params=cpu.(Flux.params(m)) 
    
    
    writedlm(bson_name * ".csv", err, ',')
    return m, args
end

using Images

img(x::Vector) = Gray.(reshape(clamp.(x, 0, 1), 28, 28))

function sample(m, args)
    imgs = FashionMNIST.images()
    #Converting image of type RGB to float 
    imgs = channelview.(imgs)
    # `args.sample_len` random digits
    I = rand(1:length(imgs), args.sample_len)
    original = [(imgs[i]) for i in I]
    before = [add_noise(imgs[i], args.p_noise) for i in I]
    # Before and after images
    after = img.(map(x -> cpu(m)(float(vec(x))), before))
    # Stack them all together
    hcat(vcat.(original, before, after)...)
end


function main(;kws...)
    cd(@__DIR__)
    m, args= train(;kws...)
    # Sample output
    @info("Saving image sample as sample_ae.png")
    
    fpath = "ae_N"* string(args.N) *"_p"* string(args.p_noise) *".png"
    save(fpath, sample(m, args))
    run(`display $fpath`)
end


function test(; kws...)
    args = Args(; kws...)
    
    # Loading the test data
    
    train_data = get_processed_data(args)
    @info("Constructing model......")
    
    # Re-constructing the model with random initial weights
    model = get_model(args)
    
    # Loading the saved parameters
    BSON.@load joinpath(args.savepath, "fashion_mnist_ae_N2_p0.0.bson") params
    
    # Loading parameters onto the model
    Flux.loadparams!(model, params)
    x = rand(2)

    L = []
    for i in range(0, 1, length=20)
        imggs = [img(cpu(model[2])([i,j])) for j in range(0, 1, length=20) ]
        imgg = hcat(imggs...)
        if isempty(L)
            L = imgg
        else
            L = vcat(L, imgg)
        end
    end
    fpath = "algo_N"* string(args.N) *"_p"* string(args.p_noise) *".png"
    save(fpath, L)
    run(`display $fpath`)
    
end

end
