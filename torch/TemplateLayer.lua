require 'nn'
require 'stn'
require 'IntensityScaleTable'

local TemplateLayer, Parent = torch.class('nn.TemplateLayer', 'nn.Module')

function TemplateLayer:__init(sizeIn, sizeInter, sizeOut, numTransformParams, height, width, sumType)
    Parent.__init(self)
    self.sizeIn = sizeIn
    self.sizeInter = sizeInter
    self.sizeOut = sizeOut
    self.numTransformParams
    self.height = height
    self.width = width
    self.sumType = sumType

    self.parallel = nn.ParallelTable()
    local templateTransformer = initTemplateTransformer(self.sizeIn, self.sizeInter, self.numTransformParams, self.height, self.width, self.sumType)
    local paramPredictor = initParamPredictor(self.sizeIn, self.sizeInter, self.sizeOut, self.numTransformParams)
    parallel:add(templateTransformer)
    parallel:add(paramPredictor)

    self:reset()
end

-- input is a table containing two tensors:
--    the transformed templates from the previous layer, which will serve as templates for this layer
--          batch,sizeIn,heightIn,widthIn
--    the parameters to transform those new templates
--          batch,sizeIn*sizeInter*(numTransformParams+1)
-- output is a table containing two tensors:
--    the new transformed templates, which will serve as templates for the next layer
--          batch,sizeInter,height,width
--    the predicted parameters to transform those templates
--          batch,sizeInter*sizeOut2*(numTransformParams+1)
function TemplateLayer:updateOutput(input)
    local templates = input[1]
    local transformParams = input[2]
    self.output = self.parallel:forward({{templates, transformParams}, transformParams})
    return self.output
end

function TemplateLayer:updateGradInput(input, gradOuput)
    local templates = input[1]
    local transformParams = input[2]
    self.gradInput = self.parallel:backward({{templates, transformParams}, transformParams}, gradOuput)
    return self.gradInput
end

function initTemplateTransformer(sizeIn, sizeInter, numTransformParams, height, width, sumType)
    local templateTransformer = nn.Sequential()
    local concat = nn.ConcatTable()
    -- Branch that copies templates and scales their values according to the predicted intensities
    local intensifyTemplates = nn.Sequential()
    local narrowIntensities = nn.ParallelTable()
    narrowIntensities:add(nn.Identity())
    local seq = nn.Sequential()
    seq:add(nn.Narrow(2,1,sizeIn*sizeInter))
    seq:add(nn.View(batchSize,sizeIn,sizeInter,1,1))
    seq:add(nn.Sigmoid()) -- Intensities should be between 0 and 1
    narrowIntensities:add(seq)
    intensifyTemplates:add(narrowIntensities)
    intensifyTemplates:add(nn.IntensityScaleTable())
    concat:add(intensifyTemplates)
    -- Branch that generates transformation grids according to the predicted parameters
    local narrowParams = nn.Sequential()
    narrowParams:add(nn.SelectTable(2))
    narrowParams:add(nn.Narrow(2,sizeIn*sizeInter+1,sizeIn*sizeInter*numTransformParams))
    narrowParams:add(nn.View(batchSize*sizeIn*sizeInter, numTransformParams))
    narrowParams:add(nn.AffineTransformMatrixGenerator(true, true, true))
    narrowParams:add(nn.AffineGridGeneratorBHWD(height, width))
    concat:add(narrowParams)
    -- Combine both branches to form
    templateTransformer:add(concat)
    templateTransformer:add(nn.BilinearSamplerBHWD())
    templateTransformer:add(nn.View(batchSize, sizeIn, sizeInter, height, width))
    templateTransformer:add(templateSum(sumType))

    return templateTransformer
end

function initParamPredictor(self.sizeIn, self.sizeInter, self.sizeOut, self.numTransformParams)
    local paramPredictor = nn.Sequential()
    -- Insert batch norm here?
    paramPredictor:add(nn.ReLU())
    paramPredictor:add(nn.Linear(sizeIn*sizeInter(numTransformParams+1)), sizeInter*sizeOut*(numTransformParams+1))
    return paramPredictor
end

function templateSum(sumType)
    local templateSum = nn.Sequential()
    if sumType == 'max' then
        templateSum:add(nn.Max(1,4))
    elseif sumType == 'mean' then
        templateSum:add(nn.Mean(1,4))
    else
        templateAdder:add(nn.MulConstant(10)) -- boolean indicates in place multiplication
        templateAdder:add(nn.Exp())
        templateAdder:add(nn.Sum(1, 4)) -- Dimension, nInputDims. Outputs a BatchSize,sizeInter,height,width tensor
        templateAdder:add(nn.Log())
        templateAdder:add(nn.MulConstant(1/10)) -- boolean indicates in place multiplication
    end
    return templateSum
end

local parallel = nn.ParallelTable()

local templateTransformer = nn.Sequential()
local concat = nn.ConcatTable()
    local intensifyTemplates = nn.Sequential()
        local narrowIntensities = nn.ParallelTable()
        narrowIntensities:add(nn.Identity())
            local seq = nn.Sequential()
            seq:add(nn.Narrow(2,1,sizeIn*sizeInter))
            seq:add(nn.View(batchSize,sizeIn,sizeInter,1,1))
            seq:add(nn.Sigmoid()) -- Intensities should be between 0 and 1
        narrowIntensities:add(seq)
    intensifyTemplates:add(narrowIntensities)
    intensifyTemplates:add(nn.IntensityScaleTable())
concat:add(intensifyTemplates)
    local narrowParams = nn.Sequential()
    narrowParams:add(nn.SelectTable(2))
    narrowParams:add(nn.Narrow(2,sizeIn*sizeInter+1,sizeIn*sizeInter*numTransformParams))
    narrowParams:add(nn.View(batchSize*sizeIn*sizeInter, numTransformParams))
    narrowParams:add(nn.AffineTransformMatrixGenerator(true, true, true))
    narrowParams:add(nn.AffineGridGeneratorBHWD(height, width))
concat:add(narrowParams)
templateTransformer:add(concat)
templateTransformer:add(nn.BilinearSamplerBHWD())
templateTransformer:add(nn.View(batchSize, sizeIn, sizeInter, height, width))
templateTransformer:add(nn.Sum(2))


local paramPredictor = nn.Sequential()
-- Insert batch norm here?
paramPredictor:add(nn.ReLU())
paramPredictor:add(nn.Linear(sizeIn*sizeInter(numTransformParams+1)), sizeInter*sizeOut*(numTransformParams+1))

parallel:add(templateTransformer)
parallel:add(paramPredictor)