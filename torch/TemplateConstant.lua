require 'nn'

local TemplateConstant, Parent = torch.class('nn.TemplateConstant', 'nn.Module')

function TemplateConstant:__init(size, height, width)
    Parent.__init(self)
    self.templates = torch.Tensor(1, size, 1, height, width)
    self.gradTemplates = torch.Tensor(1, size, 1, height, width)
end

-- input format : { (batchSize, sizeIn, 1, height, width) templates Tensor,
--                  (batchSize, sizeIn, sizeOut, 1,1) intensities Tensor}
-- output format : (batchSize, sizeIn, sizeOut, height, width) Tensor
function TemplateConstant:updateOutput(input)
    local batchSize = input:size()[1]
    self.output = self.templates:repeatTensor(batchSize,1,1,1,1)
    return self.output
end

function TemplateConstant:updateGradInput(input, gradOutput)
    self.gradInput = input:clone():zero()
    return self.gradInput
end

function TemplateConstant:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    local batchSize = input:size()[1]
    self.gradTemplates:add(scale, gradOutput:sum(1)) -- sum over batch dimension
end

function TemplateConstant:zeroGradParameters()
    self.gradTemplates:zero()
end

function TemplateConstant:parameters()
    return {self.templates}, {self.gradTemplates}
end

function TemplateConstant:reset()
    self.templates = torch.randn(self.templates:size())
    self.gradTemplates:zero()
end
