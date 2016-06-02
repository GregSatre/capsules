require 'nn'
require 'cutorch'
require 'cunn'
require 'load_dataset'
require 'IntensityScale'
require 'stn'
require 'optim'
require 'xlua'
require 'provider'
require 'image'
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
   cmd:option('-save', 'subdirectory to save/log experiments in')
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
n_train_samples = 50000
n_valid_samples = 10000
n_test_samples = 10000
geometry = {32, 32}
sampleSize = 32*32
-- Define network hyperparameters
depth = 1
height = 32
width = 32
t_height = 32
t_width = 32
hiddenSize1 = 500
hiddenSize2 = 1000
hiddenSize3 = 500
numCapsules = 12
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
-- Optimization hyperparameters
momentum = 0.9
coefL1 = 0
coefL2 = 0

----------------------------------------------------------------------
-- Load train and test sets
trainData = mnist.loadTrainSet(n_train_samples+n_valid_samples, geometry)
testData = mnist.loadTestSet(n_test_samples, geometry)
-- Normalize train and test sets
trainData:normalizeGlobal()
testData:normalizeGlobal()
-- Separate training and validation sets
-- print('==> loading yuv normalized cifar-10 data')
-- provider = torch.load 'provider.t7'
-- provider.trainData.data = provider.trainData.data:cuda()
-- provider.testData.data = provider.testData.data:cuda()
-- trainData = provider.trainData
-- testData = provider.testData
validData = {
                  data = trainData.data[{{n_train_samples+1, n_train_samples+n_valid_samples}, {}, {}, {}}]:clone(),
                  size = function() return n_valid_samples end
                }
trainData = {
                data = trainData.data[{{1, n_train_samples}, {}, {}, {}}]:clone(),
                size = function() return n_train_samples end
              }
----------------------------------------------------------------------
-- Define the model
-- Feedforward encoder
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
-- Convolutional encoder
-- encoder = nn.Sequential()
-- encoder:add(nn.SpatialConvolution(1,128,5,5,1,1))
-- encoder:add(nn.SpatialBatchNormalization(128))
-- encoder:add(nn.ReLU())
-- encoder:add(nn.SpatialConvolution(128,256,5,5,1,1))
-- encoder:add(nn.SpatialBatchNormalization(256))
-- encoder:add(nn.ReLU())
-- encoder:add(nn.SpatialConvolution(256,256,4,4,1,1))
-- encoder:add(nn.SpatialBatchNormalization(256))
-- encoder:add(nn.ReLU())
-- encoder:add(nn.SpatialConvolution(256,256,3,3,1,1))
-- encoder:add(nn.SpatialBatchNormalization(256))
-- encoder:add(nn.ReLU())
-- encoder:add(nn.SpatialConvolution(256,numCapsules*(numTransformParams+1),19,19,1,1))
-- encoder:add(nn.View(opt.batchSize,numCapsules*(numTransformParams+1),1,1))
-- Define the decoder
concat = nn.ConcatTable()
-- 1st concat branch: scale templates
seq1 = nn.Sequential()
seq1:add(nn.Narrow(2,1,numCapsules)) -- Split the obtained representation into a (BatchSize,numCapsules) Tensor of intensities
seq1:add(nn.View(opt.batchSize, numCapsules,1,1))
seq1:add(nn.Sigmoid()) -- Intensities should be between 0 and 1
seq1:add(nn.IntensityScale(numCapsules, t_height, t_width))
seq1:add(nn.View(opt.batchSize*numCapsules, t_height, t_width, 1))
-- 2nd concat branch: generate sampling grid
seq2 = nn.Sequential()
seq2:add(nn.Narrow(2,numCapsules+1,numCapsules*numTransformParams)) -- Split the obtained representation into a (BatchSize,numCapsules*numTransformParams) Tensor of parameters
-- seq2:add(nn.Tanh())
seq2:add(nn.View(opt.batchSize*numCapsules, numTransformParams))
seq2:add(nn.AffineTransformMatrixGenerator(use_rot, use_sca, use_tra))
seq2:add(nn.AffineGridGeneratorBHWD(height, width))
concat:add(seq1)
concat:add(seq2)
-- Put the two together with the sampler to form the decoder
decoder = nn.Sequential()
decoder:add(concat)
decoder:add(nn.BilinearSamplerBHWD())
decoder:add(nn.View(opt.batchSize, numCapsules, height, width)) -- This outputs a BatchSize,NumCapsules,depth,height,width Tensor of all transformed templates in the batch
templateAdder = nn.Sequential()
-- templateAdder:add(nn.Max(1, 3)) -- Dimension, nInputDims
-- templateAdder:add(nn.MulConstant(10)) -- boolean indicates in place multiplication
-- templateAdder:add(nn.Exp())
templateAdder:add(nn.Sum(1, 3)) -- Dimension, nInputDims. Outputs a BatchSize,height,width,depth Tensor
-- templateAdder:add(nn.Log())
templateAdder:add(nn.MulConstant(1/numCapsules)) -- boolean indicates in place multiplication
-- Add the templates together
decoder:add(templateAdder)

-- -- Put it all together in an autoencoder model
model = nn.Sequential()
model:add(encoder)
model:add(decoder)

criterion = nn.MSECriterion()

-- use CUDA
model:cuda()
criterion:cuda()

parameters, gradParameters = model:getParameters()

function getNextBatch()
    local inputs = torch.Tensor(opt.batchSize,height,width)
    local inputs = inputs:cuda()
    for j = 1,opt.batchSize do
        inputs[j] = trainData.data[shuffle[batch_index*opt.batchSize+j]]
    end
    return inputs, inputs
end

function feval(x)
    if x ~= parameters then
        parameters:copy(x)
    end
    -- get batch
    local inputs, targets = getNextBatch()
    -- zero gradients
    gradParameters:zero()
    -- evaluate prediction, loss and backpropagate gradients
    local prediction = model:forward(inputs)
    local loss = criterion:forward(prediction, targets)
    model:backward(inputs, criterion:backward(prediction, targets))
    return loss, gradParameters
end

batch_index = 0
shuffle = torch.randperm(trainData:size())
function train()
    epoch = epoch or 1
    shuffle = torch.randperm(trainData:size())
    batch_index = 0
    local mean_batch_loss = 0
    local max_batches = torch.floor(trainData:size()/opt.batchSize)
    -- Progress update
    print('==> doing epoch on training data:')
    print('==> online epoch # ' .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    local time = sys.clock()
    -- Optimize weights
    while batch_index < max_batches do
        xlua.progress(batch_index, max_batches)
        _, loss = optim.rmsprop(feval, parameters, optim_params)
        batch_index = batch_index + 1
        mean_batch_loss = mean_batch_loss + loss[1]
    end
    -- Progress info
    time = sys.clock() - time
    time = time / trainData:size()
    print("==> time to learn 1 sample = " .. (time*1000) .. 'ms')
    ----
    epoch = epoch + 1
    mean_batch_loss = mean_batch_loss/max_batches
    return mean_batch_loss
end

function validate()
    valid_inputs = torch.Tensor(opt.batchSize,height,width):cuda()
    local mean_valid_loss = 0
    local num_batches = 0
    for i=1,validData:size(),opt.batchSize do
        valid_inputs:zero()
        for j=1,opt.batchSize do
            valid_inputs[j] = validData.data[i+j-1]
        end
        mean_valid_loss = mean_valid_loss + criterion:forward(model:forward(valid_inputs),valid_inputs)
        num_batches = num_batches + 1
    end
    mean_valid_loss = mean_valid_loss/num_batches
    return mean_valid_loss
end

function test()
    local test_inputs = torch.Tensor(opt.batchSize,height,width):cuda()
    local mean_test_loss = 0
    local num_batches = 0
    for i=1,testData:size(),opt.batchSize do
        test_inputs:zero()
        for j=1,opt.batchSize do
            test_inputs[j] = testData.data[i+j-1]
        end
        mean_test_loss = mean_test_loss + criterion:forward(model:forward(test_inputs),test_inputs)
        num_batches = num_batches + 1
    end
    mean_test_loss = mean_test_loss/num_batches
    return mean_test_loss
end

function displaySamples(show_inputs, show_outputs, show_templates)
    inputs = torch.Tensor(opt.batchSize,height,width):cuda()
    for i=1,opt.batchSize do
        inputs[i]=testData.data[i]
    end

    outputs = model:forward(inputs)

    intscale = model:findModules('nn.IntensityScale')
    inputs = inputs:double()
    outputs = outputs:double()
    templates = intscale[1].template:double():view(numCapsules,t_height,t_width)

    if show_inputs then
        disp.image(inputs)
    end
    if show_outputs then
        disp.image(outputs)
    end
    if show_templates then
        disp.image(templates)
    end
end

model:cuda()
logger = optim.Logger(paths.concat(opt.save, 'train.log'))
inputs, _ = getNextBatch()

optim_params = {
    learningRate = opt.learningRate
}

epoch = 1
batch_index = 0
shuffle = torch.randperm(trainData:size())
while epoch<50 do
    local train_loss = train()
    local valid_loss = validate()
    local test_loss = test()
    if epoch%50 == 0 then
        displaySamples(false, false, true)
    end
    logger:add{['mean train error'] = train_loss,
               ['mean valid error'] = valid_loss,
               ['mean test error'] = test_loss}
    logger:style{['mean train error'] = '-',
                 ['mean valid error'] = '-',
                 ['mean test error'] = '-'}
    -- logger:plot()
end

displaySamples(true, true, true)
-- save current net
if opt.save then
    local filename = paths.concat(opt.save, 'autoencoderSPN'..tostring(epoch)..'.net')
    os.execute('mkdir -p ' .. opt.save)
    print('==> saving autoencoder to '..filename)
    torch.save(filename, model)
end
-- save hyperparameters
params_file = io.open(paths.concat(opt.save, 'hyperparameters.txt'), 'w')
params_file:write('Translation '..tostring(use_tra)..'/n')
params_file:write('Rotation '..tostring(use_rot)..'/n')
params_file:write('Scaling '..tostring(use_sca)..'/n')
params_file:write('Capsules '..tostring(numCapsules)..'/n')
params_file:close()