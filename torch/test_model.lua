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
   cmd:option('-capSize1', 4)
   cmd:option('-capSize2', 4)
   cmd:option('-capSize3', 4)
   cmd:option('-sumType', '', 'Type of sum to use for combining templates')
   cmd:option('-templateSize', 12, 'size in pixels of square template')
   cmd:option('-templateNumber', 10, 'number of templates')
   cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'Weight decay (corresponds to L2 regularization')
   cmd:option('-clamp', false, 'Clamp the parameters to constrain their range')
   cmd:option('-renorm', false, 'Renormalize the template parameters')
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
capSize1 = opt.capSize1
capSize2 = opt.capSize2
capSize3 = opt.capSize3
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
-- Separate training and validation sets
validData = {
                  data = trainData.data[{{n_train_samples+1, n_train_samples+n_valid_samples}, {}, {}, {}}]:clone(),
                  size = function() return n_valid_samples end
                }
trainData = {
                data = trainData.data[{{1, n_train_samples}, {}, {}, {}}]:clone(),
                size = function() return n_train_samples end
              }
-- Normalize train/valid/test sets
local std = trainData.data:std()
local mean = trainData.data:mean()
trainData.data:add(-mean):mul(1/std)
validData.data:add(-mean):mul(1/std)
testData.data:add(-mean):mul(1/std)
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
concat:add(nn.TemplateConstant(capSize1,opt.templateSize,opt.templateSize)) -- Number of templates, height, width
concat:add(nn.Identity())
decoder:add(concat)
decoder:add(nn.TemplateLayer(capSize1,capSize2,capSize3,numTransformParams,opt.templateSize,opt.templateSize,opt.templateSize,opt.templateSize, opt.batchSize, opt.sumType))
decoder:add(nn.TemplateLayer(capSize2,capSize3,capSize4,numTransformParams,opt.templateSize,opt.templateSize,opt.templateSize,opt.templateSize, opt.batchSize, opt.sumType))
decoder:add(nn.TemplateLayer(capSize3,capSize4,capSize5,numTransformParams,opt.templateSize,opt.templateSize,32,32, opt.batchSize, opt.sumType))
-- decoder:add(nn.TemplateLayer(capSize4,capSize5,1,numTransformParams,12,12,32,32, opt.batchSize, ''))
-- decoder:add(nn.TemplateLayer(capSize5,1,1,numTransformParams,12,12,32,32, opt.batchSize, ''))
decoder:add(nn.SelectTable(1))
decoder:add(nn.View(32,32))

-- -- Put it all together in an autoencoder model
model = nn.Sequential()
model:add(encoder)
model:add(decoder)
model:reset()

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
constant = model:findModules('nn.TemplateConstant')
templates = constant[1].templates:view(capSize1,t_height,t_width)

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
        -- Constrain weights to have max absolute value of 3
        if opt.clamp then
            parameters:clamp(-3,3)
        end
        -- Normalize the template parameters so that their norm does not exceed 35
        if opt.renorm then
            templates:renorm(2,1,35)
        end
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

function displaySamples(model, show_inputs, show_outputs, show_templates)
    local inputs = torch.Tensor(opt.batchSize,height,width):cuda()
    for i=1,opt.batchSize do
        inputs[i]=testData.data[i]
    end

    local outputs = model:forward(inputs)

    local model_constant = model:findModules('nn.TemplateConstant')
    inputs = inputs:double()
    outputs = outputs:double()
    local model_templates = constant[1].templates:double():view(capSize1,t_height,t_width)

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
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay
}

epoch = 1
batch_index = 0
shuffle = torch.randperm(trainData:size())
function trainUntilUserExit()
    -- local constant = model:findModules('nn.TemplateConstant')
    -- local templates = constant[1].templates
    -- while true do
    --     io.write(string.format("Currently on epoch %d. Continue (y/n)? ",epoch))
    --     local continue = io.read()
    --     if continue == "y" then
    --         io.write("How many more epochs? ")
    --         local epoch_lim = epoch + tonumber(io.read())
    --         while epoch<epoch_lim and continue do
    --             local train_loss = train()
    --             local valid_loss = validate()
    --             local test_loss = test()
    --             print("Parameters L2 norm: "..tostring(parameters:norm()))
    --             print("Parameters max: "..tostring(parameters:max()))
    --             print("Templates L2 norm: "..tostring(templates:norm()))
    --             print("Templates max: "..tostring(templates:max()))
    --             print("Templates min: "..tostring(templates:min()))
    --             logger:add{['mean train error'] = train_loss,
    --                        ['mean valid error'] = valid_loss,
    --                        ['mean test error'] = test_loss}
    --             logger:style{['mean train error'] = '-',
    --                          ['mean valid error'] = '-',
    --                          ['mean test error'] = '-'}
    --             logger:plot()
    --         end
    --         displaySamples(false, true, true)
    --     elseif continue == "n" then
    --         break
    --     end
    -- end
    while epoch<=500 do
        local train_loss = train()
        local valid_loss = validate()
        local test_loss = test()
        print("Parameters L2 norm: "..tostring(parameters:norm()))
        print("Parameters max: "..tostring(parameters:max()))
        print("Templates L2 norm: "..tostring(templates:norm()))
        print("Templates max: "..tostring(templates:max()))
        print("Templates min: "..tostring(templates:min()))
        logger:add{['mean train error'] = train_loss,
                   ['mean valid error'] = valid_loss,
                   ['mean test error'] = test_loss}
        logger:style{['mean train error'] = '-',
                     ['mean valid error'] = '-',
                     ['mean test error'] = '-'}
        logger:plot()
    end
    -- displaySamples(model, false, true, true)
end

function saveParams()
    -- save current net
    if opt.save then
        local model_file = paths.concat(opt.save, 'autoencoderSPN'..tostring(epoch)..'.net')
        os.execute('mkdir -p ' .. opt.save)
        print('==> saving autoencoder to '..model_file)
        torch.save(model_file, model)
        local encoder_file = paths.concat(opt.save, 'encoderSPN'..tostring(epoch)..'.net')
        print('==> saving encoder to '..encoder_file)
        torch.save(encoder_file, encoder)
        -- save the training parameters
        cmd:log(opt.save..'/log.txt', opt)
    end
end

displaySamples(model, true, false, false)

-- local status, err = pcall(trainUntilUserExit)
-- if not status then
--     print(err)
-- end

-- saveParams()
