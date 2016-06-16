require 'nn'
require 'cutorch'
require 'cunn'
require 'load_dataset'
require 'IntensityScale'
require 'IntensityScaleTable'
require 'TemplateLayer'
require 'TemplateConstant'
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
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-load', '', 'file to load a previsouly trained encoder from')
   cmd:option('-type', 'cuda', 'CPU or GPU training: double | cuda')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS | RMSProp')
   cmd:option('-translation', true)
   cmd:option('-rotation', true)
   cmd:option('-scaling', true)
   cmd:option('-hidden1', 1000)
   cmd:option('-hidden2', 1000)
   cmd:option('-hidden3', 1000)
   cmd:option('-templateSize', 12, 'size in pixels of square template')
   cmd:option('-templateNumber', 10, 'number of templates')
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
preTrainedLoad = not(opt.load == '')
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
t_height = opt.templateSize
t_width = opt.templateSize
hiddenSize1 = opt.hidden1
hiddenSize2 = opt.hidden2
hiddenSize3 = opt.hidden3
numCapsules = opt.templateNumber
capSize1 = 10
capSize2 = 10
capSize3 = 10
capSize4 = 1
capSize5 = 1
numTransformParams = 0
if not(opt.rotation or opt.scaling or opt.translation) then
    numTransformParams = 6
else
    if opt.rotation then
        numTransformParams = numTransformParams + 1
    end
    if opt.scaling then
        numTransformParams = numTransformParams + 1
    end
    if opt.translation then
        numTransformParams = numTransformParams + 2
    end
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
if preTrainedLoad then
    preTrainedEncoder = torch.load(opt.load)
    linearLayers = preTrainedEncoder:findModules('nn.Linear')
    topLayerSize = linearLayers[#linearLayers].weight:size()[1]
    trainableEncoder = nn.Sequential()
    trainableEncoder:add(nn.BatchNormalization(topLayerSize))
    trainableEncoder:add(nn.ReLU())
    trainableEncoder:add(nn.Linear(topLayerSize, numCapsules*(numTransformParams+1)))
    encoder = nn.Sequential()
    encoder:add(preTrainedEncoder)
    encoder:add(trainableEncoder)
else
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
    encoder:add(nn.Linear(hiddenSize3, capSize1*capSize2*(numTransformParams+1))) -- +1 for intensity
end
-- Define the decoder
decoder = nn.Sequential()
-- Create a {templates, template parameters} table
concat = nn.ConcatTable()
concat:add(nn.TemplateConstant(capSize1,12,12)) -- Number of templates, height, width
concat:add(nn.Identity())
decoder:add(concat)
decoder:add(nn.TemplateLayer(capSize1,capSize2,capSize3,numTransformParams,12,12,20,20, opt.batchSize, 'max'))
decoder:add(nn.TemplateLayer(capSize2,capSize3,capSize4,numTransformParams,20,20,20,20, opt.batchSize, 'max'))
decoder:add(nn.TemplateLayer(capSize3,capSize4,capSize5,numTransformParams,20,20,32,32, opt.batchSize, 'max'))
decoder:add(nn.SelectTable(1))
decoder:add(nn.View(32,32))

-- -- Put it all together in an autoencoder model
model = nn.Sequential()
model:add(encoder)
model:add(decoder)

criterion = nn.MSECriterion()

-- use CUDA
model:cuda()
criterion:cuda()

-- Fetch trainable parameters
if preTrainedLoad then
    preTrainedParams, preTrainedGradParams = preTrainedEncoder:getParameters()
    modelParams, modelGradParams = model:getParameters()
    parameters = modelParams:narrow(1,preTrainedParams:size(1)+1,modelParams:size(1)-preTrainedParams:size(1))
    gradParameters = modelGradParams:narrow(1,preTrainedGradParams:size(1)+1,modelGradParams:size(1)-preTrainedGradParams:size(1))
else
    parameters, gradParameters = model:getParameters()
end

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
    -- print(prediction:size())
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

    -- intscale = model:findModules('nn.IntensityScale')
    inputs = inputs:double()
    outputs = outputs:double()
    -- templates = intscale[1].template:double():view(numCapsules,t_height,t_width)

    if show_inputs then
        disp.image(inputs)
    end
    if show_outputs then
        disp.image(outputs)
    end
    -- if show_templates then
        -- disp.image(templates)
    -- end
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
while epoch<11 do
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
    logger:plot()
end

displaySamples(true, true, true)
-- save current net
if opt.save then
    local model_file = paths.concat(opt.save, 'autoencoderSPN'..tostring(epoch)..'.net')
    os.execute('mkdir -p ' .. opt.save)
    print('==> saving autoencoder to '..model_file)
    torch.save(model_file, model)
    local encoder_file = paths.concat(opt.save, 'encoderSPN'..tostring(epoch)..'.net')
    print('==> saving encoder to '..encoder_file)
    torch.save(encoder_file, encoder)
end