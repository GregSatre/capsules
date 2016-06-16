require 'nn'
require 'stn'
require 'IntensityScaleTable'

local TemplateLayer, Parent = torch.class('nn.TemplateLayer', 'nn.Module')

function TemplateLayer:__init(sizeIn, sizeInter, sizeOut, numTransformParams, heightIn, widthIn, heightInter, widthInter, batchSize, sumType)
    Parent.__init(self)
    self.sizeIn = sizeIn
    self.sizeInter = sizeInter
    self.sizeOut = sizeOut
    self.numTransformParams = numTransformParams
    self.heightIn = heightIn
    self.widthIn = widthIn
    self.heightInter = heightInter
    self.widthInter = widthInter
    self.batchSize = batchSize
    self.sumType = sumType

    self.parallel = nn.ConcatTable()
    local templateTransformer = initTemplateTransformer(self.sizeIn, self.sizeInter, self.numTransformParams, self.heightIn, self.widthIn, self.heightInter, self.widthInter, self.batchSize, self.sumType)
    local paramPredictor = initParamPredictor(self.sizeIn, self.sizeInter, self.sizeOut, self.numTransformParams)
    self.parallel:add(templateTransformer)
    self.parallel:add(paramPredictor)

    self:reset()
end

-- input is a table containing two tensors:
--    the transformed templates from the previous layer, which will serve as templates for this layer
--          batch,sizeIn,heightIn,widthIn
--    the parameters to transform those new templates
--          batch,sizeIn*sizeInter*(numTransformParams+1)
-- output is a table containing two tensors:
--    the new transformed templates, which will serve as templates for the next layer
--          batch,sizeInter,heightInter,widthInter
--    the predicted parameters to transform those templates
--          batch,sizeInter*sizeOut2*(numTransformParams+1)
function TemplateLayer:updateOutput(input)
    self.output = self.parallel:forward(input)
    return self.output
end

function TemplateLayer:updateGradInput(input, gradOuput)
    self.gradInput = self.parallel:backward(input, gradOuput)
    return self.gradInput
end

function initTemplateTransformer(sizeIn, sizeInter, numTransformParams, heightIn, widthIn, heightInter, widthInter, batchSize, sumType)
    local templateTransformer = nn.Sequential()
    local concat = nn.ConcatTable()
    -- Branch that copies templates and scales their values according to the predicted intensities
    local intensifyTemplates = nn.Sequential()
    local narrowIntensities = nn.ParallelTable()
    narrowIntensities:add(nn.Identity())
    local seq = nn.Sequential()
    seq:add(nn.Narrow(2,1,sizeIn*sizeInter))
    seq:add(nn.View(batchSize,sizeIn,sizeInter,1,1))
    seq:add(nn.SoftSign()) -- Intensities should be between 0 and 1
    narrowIntensities:add(seq)
    intensifyTemplates:add(narrowIntensities)
    intensifyTemplates:add(nn.IntensityScaleTable())
    intensifyTemplates:add(nn.View(batchSize*sizeIn*sizeInter, heightIn, widthIn, 1))
    concat:add(intensifyTemplates)
    -- Branch that generates transformation grids according to the predicted parameters
    local narrowParams = nn.Sequential()
    narrowParams:add(nn.SelectTable(2))
    narrowParams:add(nn.Narrow(2,sizeIn*sizeInter+1,sizeIn*sizeInter*numTransformParams))
    narrowParams:add(nn.View(batchSize*sizeIn*sizeInter, numTransformParams))
    narrowParams:add(nn.AffineTransformMatrixGenerator(true, true, true))
    narrowParams:add(nn.AffineGridGeneratorBHWD(heightInter, widthInter))
    concat:add(narrowParams)
    -- Combine both branches to form
    templateTransformer:add(concat)
    templateTransformer:add(nn.BilinearSamplerBHWD())
    templateTransformer:add(nn.View(batchSize, sizeIn, sizeInter, heightInter, widthInter))
    templateTransformer:add(templateSum(sumType))
    templateTransformer:add(nn.View(batchSize, sizeInter, 1, heightInter, widthInter))

    return templateTransformer
end

function initParamPredictor(sizeIn, sizeInter, sizeOut, numTransformParams)
    local paramPredictor = nn.Sequential()
    paramPredictor:add(nn.SelectTable(2))
    -- Insert batch norm here?
    paramPredictor:add(nn.ELU())
    paramPredictor:add(nn.Linear(sizeIn*sizeInter*(numTransformParams+1), sizeInter*sizeOut*(numTransformParams+1)))
    return paramPredictor
end

function templateSum(sumType)
    local templateSum = nn.Sequential()
    if sumType == 'max' then
        templateSum:add(nn.Max(1,4))
    elseif sumType == 'mean' then
        templateSum:add(nn.Mean(1,4))
    else
        templateSum:add(nn.MulConstant(10)) -- boolean indicates in place multiplication
        templateSum:add(nn.Exp())
        templateSum:add(nn.Sum(1, 4)) -- Dimension, nInputDims. Outputs a BatchSize,sizeInter,heightInter,widthInter tensor
        templateSum:add(nn.Log())
        templateSum:add(nn.MulConstant(1/10)) -- boolean indicates in place multiplication
    end
    return templateSum
end

-- local parallel = nn.ParallelTable()

-- local templateTransformer = nn.Sequential()
-- local concat = nn.ConcatTable()
--     local intensifyTemplates = nn.Sequential()
--         local narrowIntensities = nn.ParallelTable()
--         narrowIntensities:add(nn.Identity())
--             local seq = nn.Sequential()
--             seq:add(nn.Narrow(2,1,sizeIn*sizeInter))
--             seq:add(nn.View(batchSize,sizeIn,sizeInter,1,1))
--             seq:add(nn.Sigmoid()) -- Intensities should be between 0 and 1
--         narrowIntensities:add(seq)
--     intensifyTemplates:add(narrowIntensities)
--     intensifyTemplates:add(nn.IntensityScaleTable())
-- concat:add(intensifyTemplates)
--     local narrowParams = nn.Sequential()
--     narrowParams:add(nn.SelectTable(2))
--     narrowParams:add(nn.Narrow(2,sizeIn*sizeInter+1,sizeIn*sizeInter*numTransformParams))
--     narrowParams:add(nn.View(batchSize*sizeIn*sizeInter, numTransformParams))
--     narrowParams:add(nn.AffineTransformMatrixGenerator(true, true, true))
--     narrowParams:add(nn.AffineGridGeneratorBHWD(heightInter, widthInter))
-- concat:add(narrowParams)
-- templateTransformer:add(concat)
-- templateTransformer:add(nn.BilinearSamplerBHWD())
-- templateTransformer:add(nn.View(batchSize, sizeIn, sizeInter, heightInter, widthInter))
-- templateTransformer:add(nn.Sum(2))


-- local paramPredictor = nn.Sequential()
-- -- Insert batch norm here?
-- paramPredictor:add(nn.ReLU())
-- paramPredictor:add(nn.Linear(sizeIn*sizeInter(numTransformParams+1)), sizeInter*sizeOut*(numTransformParams+1))

-- parallel:add(templateTransformer)
-- parallel:add(paramPredictor)