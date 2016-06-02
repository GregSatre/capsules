----------------------------------------------------------------------
-- Import necessary packages
require 'optim'
require 'xlua'
require 'nn'
require 'stn'
require 'IntensityScale'
require 'load_dataset'
disp = require 'display'
----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Training/Optimization')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-type', 'cuda', 'CPU or GPU training: double | cuda')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS | RMSProp')
   cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-nesterov', false, 'use nesterov momentum (SGD only)')
   cmd:option('-alpha', 0.99, 'Decay rate (RMSProp only)')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------
-- Seed the RNG
torch.manualSeed(12345)
----------------------------------------------------------------------
-- Define dataset constants
numTrainSamples = 60000
validRatio = 59936/60000
-- validRatio = 1/6
numTestSamples = 10000
geometry = 32
sampleSize = 32*32
-- Define network hyperparameters
hiddenSize1 = 500
hiddenSize2 = 500
hiddenSize3 = 500
numCapsules = 10
use_rot = true
use_sca = true
use_tra = true
numTransformParams = 0
if use_rot then
    numTransformParams = numTransformParams + 1
end
if use_sca then
    numTransformParams = numTransformParams + 1
end
if use_tra then
    numTransformParams = numTransformParams + 2
end

-- Define training variables
lr = opt.learningRate or 1e-2
template_lr = 1e-2
if opt.type == 'cuda' then
    print('==> Requiring CUDA')
    require 'cutorch'
    require 'cunn'
    print('==> Required CUDA successfully')
    -- torch.setdefaulttensortype('torch.CudaTensor')
end
----------------------------------------------------------------------
-- Load train and test sets
trainData = mnist.loadTrainSet(numTrainSamples, {32,32})
testData = mnist.loadTestSet(numTestSamples, {32,32})

-- Split into training and validation sets. Data is already shuffled
numValidSamples = numTrainSamples*validRatio
validationSet = {
                  data = trainData.data[{{numTrainSamples-numValidSamples+1, numTrainSamples}, {}, {}, {}}]:clone(),
                  size = function() return numValidSamples end
                }
trainingSet = {
                data = trainData.data[{{1, numTrainSamples-numValidSamples}, {}, {}, {}}]:clone(),
                size = function() return numTrainSamples-numValidSamples end
              }

function normalize(data)
  mean = data:mean()
  std = data:std()
  data:add(-mean)
  data:div(std)
end

-- Normalize datasets
testData:normalizeGlobal()
normalize(validationSet.data)
normalize(trainingSet.data)
----------------------------------------------------------------------
-- Define the model

-- First define the encoder
encoder = nn.Sequential()
encoder:add(nn.Reshape(sampleSize))
encoder:add(nn.Linear(sampleSize, hiddenSize1))
encoder:add(nn.BatchNormalization(hiddenSize1))
encoder:add(nn.ReLU())
encoder:add(nn.Linear(hiddenSize1, hiddenSize2))
encoder:add(nn.BatchNormalization(hiddenSize2))
encoder:add(nn.ReLU())
encoder:add(nn.Linear(hiddenSize2, hiddenSize3))
encoder:add(nn.BatchNormalization(hiddenSize3))
encoder:add(nn.ReLU())
encoder:add(nn.Linear(hiddenSize3, numCapsules*(numTransformParams+1))) -- +1 for intensity
encoder:add(nn.BatchNormalization(numCapsules*(numTransformParams+1))) -- Probably shouldn't have BN after encoder output
concat = nn.ConcatTable()
-- 1st concat branch: scale templates
seq1 = nn.Sequential()
seq1:add(nn.Narrow(2,1,numCapsules)) -- Split the obtained representation into a (BatchSize,numCapsules) Tensor of intensities
seq1:add(nn.View(opt.batchSize, numCapsules, 1, 1))
seq1:add(nn.Sigmoid()) -- Intensities should be between 0 and 1
seq1:add(nn.IntensityScale(numCapsules, 32, 32))
seq1:add(nn.View(opt.batchSize*numCapsules, 32, 32, 1))
-- 2nd concat branch: generate sampling grid
seq2 = nn.Sequential()
seq2:add(nn.Narrow(2,numCapsules+1,numCapsules*numTransformParams)) -- Split the obtained representation into a (BatchSize,numCapsules*numTransformParams) Tensor of parameters
-- seq2:add(nn.Tanh())
seq2:add(nn.View(opt.batchSize*numCapsules, numTransformParams))
seq2:add(nn.AffineTransformMatrixGenerator(use_rot, use_sca, use_tra))
seq2:add(nn.AffineGridGeneratorBHWD(geometry, geometry))
concat:add(seq1)
concat:add(seq2)
-- Put the two together with the sampler to form the decoder
decoder = nn.Sequential()
decoder:add(concat)
decoder:add(nn.BilinearSamplerBHWD()) -- This outputs a BatchSize,NumCapsules,32,32 Tensor of all transformed templates in the batch
decoder:add(nn.View(opt.batchSize, numCapsules, 32, 32))
decoder:add(nn.SpatialBatchNormalization(numCapsules))
templateAdder = nn.Sequential()
-- templateAdder:add(nn.Max(1, 3)) -- Dimension, nInputDims
-- templateAdder:add(nn.MulConstant(20, true)) -- boolean indicates in place multiplication
templateAdder:add(nn.Exp())
templateAdder:add(nn.Sum(1, 3)) -- Dimension, nInputDims. Outputs a BatchSize,32,32 Tensor
templateAdder:add(nn.Log())
-- templateAdder:add(nn.MulConstant(1/20, true)) -- boolean indicates in place multiplication
-- Add the templates together
decoder:add(templateAdder)

-- -- Put it all together in an autoencoder model
autoencoder = nn.Sequential()
autoencoder:add(encoder)
autoencoder:add(decoder)

-- The loss will be the classic Mean-Squared Error
criterion = nn.MSECriterion()
----------------------------------------------------------------------
-- CUDA?
if opt.type == 'cuda' then
    autoencoder:cuda()
    criterion:cuda()
    -- torch.setdefaulttensortype('torch.CudaTensor')
end
----------------------------------------------------------------------
-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
-- validLogger = optim.Logger(paths.concat(opt.save, 'valid.log'))
-- testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
----------------------------------------------------------------------
-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if autoencoder then
    parameters,gradParameters = autoencoder:getParameters()
end
----------------------------------------------------------------------
state = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    alpha = opt.alpha,
}

-- feval = function(x)
--     -- get new parameters
--     if x ~= parameters then
--        parameters:copy(x)
--     end

--     -- reset gradients. IMPORTANT
--     gradParameters:zero()
--     -- evaluate function for complete mini batch
--     local batchSize = inputs:size(1)
--     -- local outputs = autoencoder:forward(inputs)
--     local predictions = encoder:forward(inputs)
--     local outputs = decoder:forward(predictions)
--     local loss = criterion:forward(outputs, inputs)
--     -- estimate dloss/dW
--     local dloss_do = criterion:backward(outputs, inputs)
--     local dt_dloss = autoencoder:backward(predictions, dloss_do)
--     numBatches = numBatches + 1
--     meanError = meanError + loss
--     -- Normalize gradient and error
--     -- gradParameters:div(batchSize)
--     -- pred_gradParameters:div(batchSize)
--     -- template_gradParameters:div(batchSize)
--     -- loss = loss/batchSize

--     -- return loss and dloss/dX
--     return loss,gradParameters
-- end


-- Training function
function train()
    meanError = 0
    numBatches = 0

    -- epoch tracker
    epoch = epoch or 1

    -- local vars
    local time = sys.clock()

    -- shuffle at each epoch
    -- torch.setdefaulttensortype('torch.DoubleTensor')
    shuffle = torch.randperm(trainingSet:size())
    -- torch.setdefaulttensortype('torch.CudaTensor')
    -- do one epoch
    print('==> doing epoch on training data:')
    print('==> online epoch # ' .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    for t = 1,trainingSet:size(),opt.batchSize do
        -- disp progress
        xlua.progress(t, trainingSet:size())

        -- create mini batch
        local inputs = torch.Tensor(opt.batchSize, 32, 32)
        if opt.type == 'cuda' then inputs = inputs:cuda() end
        local k = 1
        for i = t,math.min(t+opt.batchSize-1,trainingSet:size()) do
            -- load new sample
            local input = torch.Tensor(1,32,32)
            if opt.type == 'cuda' then input = trainingSet.data[shuffle[i]]:cuda()
            elseif opt.type == 'double' then input = trainingSet.data[shuffle[i]]:double() end
            inputs[k] = input
            k = k+1
        end

        local feval = function(x)
                -- get new parameters
                if x ~= parameters then
                   parameters:copy(x)
                end

                -- reset gradients. IMPORTANT
                gradParameters:zero()
                -- evaluate function for complete mini batch
                local batchSize = inputs:size(1)
                -- local outputs = autoencoder:forward(inputs)
                local predictions = encoder:forward(inputs)
                local outputs = decoder:forward(predictions)
                local loss = criterion:forward(outputs, inputs)
                -- estimate dloss/dW
                local dloss_do = criterion:backward(outputs, inputs)
                local dt_dloss = autoencoder:backward(predictions, dloss_do)
                numBatches = numBatches + 1
                meanError = meanError + loss
                -- Normalize gradient and error
                -- gradParameters:div(batchSize)
                -- pred_gradParameters:div(batchSize)
                -- template_gradParameters:div(batchSize)
                -- loss = loss/batchSize

                -- return loss and dloss/dX
                return loss,gradParameters
            end

        if opt.optimization == 'SGD' then -- SGD
            optim.sgd(feval, parameters, state)
        elseif opt.optimization == 'RMSProp' then -- RMSProp
            optim.rmsprop(feval, parameters, state)
        end
    end

    -- time taken
    time = sys.clock() - time
    time = time / trainingSet:size()
    print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- -- get error on training set
    -- local inputs = trainingSet.data:cuda()
    -- local reconstructions = autoencoder:forward(inputs)
    -- trainingErr = criterion:forward(reconstructions, inputs)

    -- get error on validation set
    -- local inputs = validationSet.data:view(10000,32,32):cuda()
    -- local reconstructions = autoencoder:forward(inputs)
    -- print("ok")
    -- validErr = criterion:forward(reconstructions, inputs)

    -- update logger/plot
    meanError = meanError/numBatches
    trainLogger:add{['mean batch error'] = meanError}
                    -- ['mean validation error'] = validErr}

    -- next epoch
    epoch = epoch + 1
end

epoch = 1
while epoch<20 do
    train()
    if opt.plot then
        trainLogger:style{['mean batch error'] = '-'}
                          -- ['mean validation error'] = '-'}
        -- torch.setdefaulttensortype('torch.DoubleTensor')
        trainLogger:plot()
        -- torch.setdefaulttensortype('torch.CudaTensor')
        -- validLogger:plot()
    end

    -- save/log current net every 50 epochs
    if opt.save then
        if epoch % 20 == 0 and epoch <= 100 then
          local filename = paths.concat(opt.save, 'autoencoderSPN'..tostring(epoch)..'.net')
          os.execute('mkdir -p ' .. sys.dirname(filename))
          print('==> saving autoencoder to '..filename)
          torch.save(filename, autoencoder)
        end
    end

    -- if pcall(train) then
    --     if opt.plot then
    --        trainLogger:style{['mean training error'] = '-',
    --                          ['mean validation error'] = '-'}
    --        trainLogger:plot()
    --        -- validLogger:plot()
    --     end

    --     -- save/log current net every 50 epochs
    --     if epoch % 50 == 0 then
    --       local filename = paths.concat(opt.save, 'autoencoderSPN'..tostring(epoch)..'.net')
    --       os.execute('mkdir -p ' .. sys.dirname(filename))
    --       print('==> saving autoencoder to '..filename)
    --       torch.save(filename, autoencoder)
    --     end
    -- else
    --     -- save/log current net in case of unexpected exit
    --     local filename = paths.concat(opt.save, 'autoencoderSPN'..tostring(epoch)..'.net')
    --     os.execute('mkdir -p ' .. sys.dirname(filename))
    --     print('==> saving autoencoder to '..filename)
    --     torch.save(filename, autoencoder)
    -- end
end

model = torch.load('results/SPN/exp24/autoencoderSPN100.net')
intscale = model:findModules('nn.IntensityScale')
templates = intscale[1].template
views = model:findModules('nn.View')
trans_templates = views[5].output

inputs = torch.Tensor(64,32,32)
for i=1,64 do
    inputs[i]=trainingSet.data[i]
end

outputs = model:forward(inputs)

-- torch.setdefaulttensortype('torch.DoubleTensor')

-- disp.image(inputs)
-- disp.image(outputs)
-- disp.image(templates)

-- for i=23,25 do
--     disp.image(trans_templates[i])
--     disp.image(trans_templates[i]:sum(1))
-- end

disp.image(trans_templates[21])
disp.image(trans_templates[21]:sum(1))
