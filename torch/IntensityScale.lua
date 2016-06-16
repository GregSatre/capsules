require 'nn'

local IntensityScale, Parent = torch.class('nn.IntensityScale', 'nn.Module')

function IntensityScale:__init(numCapsules, height, width)
    Parent.__init(self)
    self.template = torch.Tensor(1, numCapsules, height, width)
    self.gradTemplate = torch.Tensor(self.template:size())
    -- self.bsize = bsize
    self.numCapsules = numCapsules
    self.height = height
    self.width = width
    self:reset()
end

-- input format : (BatchSize, numCapsules, 1,1) Tensor
function IntensityScale:updateOutput(input)
    -- checkDimension(input, self.bsize, self.numCapsules)
    local bsize = input:size()[1]
    -- Expand first, then clone
    -- self.output = self.template:expand(bsize, self.numCapsules, self.height, self.width):clone()
    self.output = self.template:repeatTensor(bsize,1,1,1)
    local intensities = input:expand(self.output:size())
    -- Element-wise multiplication between intensities and output
    self.output:cmul(intensities)
    return self.output
end

function IntensityScale:updateGradInput(input, gradOutput)
    -- checkDimension(input, self.bsize, self.numCapsules)
    local bsize = input:size()[1]
    local gradOutput_temp = gradOutput:clone()
    -- Element-wise multiplication between template and gradOutput_temp
    gradOutput_temp:cmul(self.template:expandAs(gradOutput))
    self.gradInput:resizeAs(input):zero()
    self.gradInput = gradOutput_temp:sum(3):sum(4)
    return self.gradInput
end

function IntensityScale:accGradParameters(input, gradOutput, scale)
    -- checkDimension(input, self.bsize, self.numCapsules)
    local scale = scale or 1
    local bsize = input:size()[1]
    local height = gradOutput:size()[3]
    local width = gradOutput:size()[4]
    local intensities = input:repeatTensor(1,1,height,width)
    -- Element-wise multiplication between intensities and gradOutput
    intensities:cmul(gradOutput)
    -- Sum gradients over batch dimension
    self.gradTemplate:add(scale, intensities:sum(1))
end

function IntensityScale:cuda()
    Parent.cuda(self)
    self.template:cuda()
    self.gradTemplate:cuda()
end

function IntensityScale:reset()
    local sqrt = torch.sqrt
    local numWeights = self.numCapsules*self.height*self.width
    self.template = torch.randn(self.template:size())*0.02
    -- self.template = torch.randn(self.template:size()):div(torch.sqrt(self.height*self.width))
    self.gradTemplate:zero()
end

function IntensityScale:zeroGradParameters()
    self.gradTemplate:zero()
end

function IntensityScale:parameters()
    return {self.template}, {self.gradTemplate}
end

function checkDimension(input, bsize, numCapsules)
    if input:size()[1] ~= bsize or input:size()[2] ~= numCapsules then
        -- print("incoherent sizes in IntensityScale module")
    end
end