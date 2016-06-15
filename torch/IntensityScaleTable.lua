require 'nn'

local IntensityScaleTable, Parent = torch.class('nn.IntensityScaleTable', 'nn.Module')

function IntensityScaleTable:__init()
    Parent.__init(self)
    self.gradInput = {}
end

-- input format : { (batchSize, sizeIn, 1, height, width) templates Tensor,
--                  (batchSize, sizeIn, sizeOut, 1,1) intensities Tensor}
-- output format : (batchSize, sizeIn, sizeOut, height, width) Tensor
function IntensityScaleTable:updateOutput(input)
    local bsize = input[2]:size()[1]
    local sizeIn = input[2]:size()[2]
    local sizeOut = input[2]:size()[3]
    local templates = input[1]:repeatTensor(1,1,sizeOut,1,1)
    local intensities = input[2]:expand(templates:size())

    self.output = templates
    -- Element-wise multiplication between intensities and output
    self.output:cmul(intensities)
    return self.output
end

function IntensityScaleTable:updateGradInput(input, gradOutput)
    local bsize = input[2]:size()[1]
    local sizeIn = input[2]:size()[2]
    local sizeOut = input[2]:size()[3]
    local templates = input[1]:repeatTensor(1,1,sizeOut,1,1)
    local intensities = input[2]:expand(templates:size()):clone()
    -- Gradients of templates w/r to ouput
    intensities:cmul(gradOutput)
    self.gradInput[1] = intensities:sum(3) -- sum over sizeOut-repeated dimension
    -- Gradients of intensities w/r to output
    local gradOutput_temp = gradOutput:clone()
    -- Element-wise multiplication between templates and gradOutput_temp
    gradOutput_temp:cmul(templates)
    self.gradInput[2] = gradOutput_temp:sum(4):sum(5)
    return self.gradInput
end
