require 'nn'

local TemplateCapsule, Parent = torch.class('nn.TemplateCapsule', 'nn.Module')

-- TemplateCapsule constructor
-- inputs:
--          geometry : {int, int}, the geometry of the template
function TemplateCapsule:__init(geometry, output_width)
   Parent.__init(self)
   self.template = torch.Tensor(geometry[1], geometry[2])
   self.gradTemplate = torch.Tensor(self.template:size())
   self.gradInput = torch.Tensor(3)
   self.output = torch.zeros(output_width, output_width)

   self:reset()
end

-- input: tensor of size 3
--        input[1] and input[2] are translation parameters
--        input[3] is the probability coefficient
-- output:
--        32*32 translated and probability-weighted template
function TemplateCapsule:updateOutput(input)
    local trans_x = input[1]
    local trans_y = input[2]
    local prob = input[3]

    local out_y = 1
    local out_x = 0
    local d1 = self.output:size()[1] -- number of rows
    local d2 = self.output:size()[2] -- number of columns

    local timer = torch.Timer()
    -- iterate over the columns (x and d2), then the rows (y and d1)
    self.output:apply(function()
            out_x = out_x + 1
            if out_x > d2 then
                out_x = 1
                out_y = out_y +1
            end
            return self:getInterpolatedTemplateValue(out_x - trans_x, out_y - trans_y)
        end)
    self.output:mul(prob)
    print('Time for UpdateOutput: '..timer:time().real..' seconds')
    return self.output
end

-- (column, row)
function TemplateCapsule:getInterpolatedTemplateValue(x, y)
    -- Get four surrounding pixel coordinates
    -- print(x)
    local low_x = math.floor(x)
    local high_x = low_x + 1
    local low_y = math.floor(y)
    local high_y = low_y + 1

    -- First is row number (y, height), second is column (x, width)
    local temp11 = getTensorValue(self.template, low_y, low_x)
    local temp12 = getTensorValue(self.template, low_y, high_x)
    local temp21 = getTensorValue(self.template, high_y, low_x)
    local temp22 = getTensorValue(self.template, high_y, high_x)

    local x_remainder = x - low_x
    local y_remainder = y - low_y

    local temp1X = x_remainder*temp12 + (1 - x_remainder)*temp11
    local temp2X = x_remainder*temp22 + (1 - x_remainder)*temp21
    local tempYX = y_remainder*temp2X + (1 - y_remainder)*temp1X
    -- Normally divide by (x2 - x1)(y2 - y1) but here this is equal to 1
    return tempYX
end

function getTensorValue(tensor, row, column)
    local temp_height = tensor:size()[1]
    local temp_width = tensor:size()[2]
    if column < 1 or column > temp_width or row < 1 or row > temp_height then
        return 0
    else
        return tensor[row][column]
    end
end

function TemplateCapsule:updateGradInput(input, gradOutput)
    -- input is the input to the module
    -- gradOutput is the gradient of the error with respect to the output of the module
    -- self.gradInput is the gradient of the error with respect to the input of the module

    local timer = torch.Timer()

    local trans_x = input[1]
    local trans_y = input[2]
    local prob = input[3]

    local grad_out_height = gradOutput:size()[1]
    local grad_out_width = gradOutput:size()[2]
    if gradOutput:type() == 'torch.CudaTensor' then
        grad_t_x = torch.CudaTensor(grad_out_height, grad_out_width):fill(0)
        grad_t_y = torch.CudaTensor(grad_out_height, grad_out_width):fill(0)
        grad_prob = torch.CudaTensor(grad_out_height, grad_out_width):fill(0)
    else
        grad_t_x = torch.Tensor(grad_out_height, grad_out_width):fill(0)
        grad_t_y = torch.Tensor(grad_out_height, grad_out_width):fill(0)
        grad_prob = torch.Tensor(grad_out_height, grad_out_width):fill(0)
    end

    self.gradInput:zero()
    -- calculate the gradients
    for row = 1, grad_out_height do
        for column = 1, grad_out_width do
            -- pixel location before translation
            local in_x = column - trans_x
            local in_y = row - trans_y

            -- Get four surrounding pixel coordinates
            local low_x = math.floor(in_x)
            local high_x = low_x + 1
            local low_y = math.floor(in_y)
            local high_y = low_y + 1

            -- First is row number (y, height), second is column (x, width)
            local temp11 = getTensorValue(self.template, low_y, low_x)
            local temp12 = getTensorValue(self.template, low_y, high_x)
            local temp21 = getTensorValue(self.template, high_y, low_x)
            local temp22 = getTensorValue(self.template, high_y, high_x)

            local x_remainder = in_x - low_x
            local y_remainder = in_y - low_y

            local temp1X = x_remainder*temp12 + (1 - x_remainder)*temp11
            local temp2X = x_remainder*temp22 + (1 - x_remainder)*temp21
            local tempYX = y_remainder*temp2X + (1 - y_remainder)*temp1X
            -- Normally divide by (x2 - x1)(y2 - y1) but here this is equal to 1

            -- First component is y-axis, then x-axis (first row, then column)
            grad_t_x[{row, column}] = y_remainder*(temp21 - temp22) + (1 - y_remainder)*(temp11 - temp12)
            grad_t_y[{row, column}] = x_remainder*(temp12 - temp22) + (1 - x_remainder)*(temp11 - temp21)
            grad_prob[{row, column}] = tempYX
        end
    end

    -- local i = 1
    -- local j = 0

    -- grad_t_x:apply(function(element)
    --         j = j + 1
    --         -- print(temp_width)
    --         if j > grad_out_width then
    --             j = 1
    --             i = i + 1
    --         end
    --         -- accumulate the calculated gradient and the current cell's value
    --         return calculateInputGradient(i, j, input, self.template, 'x')
    --     end)

    -- local i = 1
    -- local j = 0

    -- grad_t_y:apply(function(element)
    --         j = j + 1
    --         -- print(temp_width)
    --         if j > grad_out_width then
    --             j = 1
    --             i = i + 1
    --         end
    --         -- accumulate the calculated gradient and the current cell's value
    --         return calculateInputGradient(i, j, input, self.template, 'y')
    --     end)

    -- local i = 1
    -- local j = 0

    -- grad_prob:apply(function(element)
    --         j = j + 1
    --         -- print(temp_width)
    --         if j > grad_out_width then
    --             j = 1
    --             i = i + 1
    --         end
    --         -- accumulate the calculated gradient and the current cell's value
    --         return calculateInputGradient(i, j, input, self.template, 'prob')
    --     end)

    -- scale by (-prob) to complete the gradients
    grad_t_x:mul(prob)
    grad_t_y:mul(prob)
    -- print(grad_t_x:type())
    -- print(grad_t_x:size())
    -- print(gradOutput:type())
    -- print(gradOutput:size())
    -- print(self.gradInput:type())
    -- print(self.gradInput:size())
    print('Time for UpdateGradInput: '..timer:time().real..' seconds')
    self.gradInput[1] = torch.dot(grad_t_x, gradOutput)
    self.gradInput[2] = torch.dot(grad_t_y, gradOutput)
    self.gradInput[3] = torch.dot(grad_prob, gradOutput)

    return self.gradInput
end

function TemplateCapsule:accGradParameters(input, gradOutput, scale)
    -- input is the input to the module
    -- gradOutput is the gradient of the error with respect to the output of the module
    -- self.gradTemplate is the gradient of the error with respect to the parameters of the module (i.e. the template values)
    local timer = torch.Timer()

    local scale = scale or 1

    local temp_height = self.gradTemplate:size()[1]
    local temp_width = self.gradTemplate:size()[2]

    local i = 1
    local j = 0

    self.gradTemplate:apply(function(element)
            j = j + 1
            -- print(temp_width)
            if j > temp_width then
                j = 1
                i = i + 1
            end
            -- accumulate the calculated gradient and the current cell's value
            return element + calculateTemplateGradient(i, j, input, gradOutput, scale)
        end)
    print('Time for AccGradParameters: '..timer:time().real..' seconds')
    return self.gradTemplate
end

function calculateInputGradient(i, j, input, template, type)
    local trans_x = input[1]
    local trans_y = input[2]
    local prob = input[3]

    local in_x = j - trans_x
    local in_y = i - trans_y

    -- Get four surrounding pixel coordinates
    local low_x = math.floor(in_x)
    local high_x = low_x + 1
    local low_y = math.floor(in_y)
    local high_y = low_y + 1

    -- First is row number (y, height), second is column (x, width)
    local temp11 = getTensorValue(template, low_y, low_x)
    local temp12 = getTensorValue(template, low_y, high_x)
    local temp21 = getTensorValue(template, high_y, low_x)
    local temp22 = getTensorValue(template, high_y, high_x)

    local x_remainder = in_x - low_x
    local y_remainder = in_y - low_y

    -- Normally divide by (x2 - x1)(y2 - y1) but here this is equal to 1

    -- First component is y-axis, then x-axis (first row, then column)
    if type == 'x' then
        return y_remainder*(temp21 - temp22) + (1 - y_remainder)*(temp11 - temp12) * prob
    elseif type == 'y' then
        return x_remainder*(temp12 - temp22) + (1 - x_remainder)*(temp11 - temp21) * prob
    else
        local temp1X = x_remainder*temp12 + (1 - x_remainder)*temp11
        local temp2X = x_remainder*temp22 + (1 - x_remainder)*temp21
        local tempYX = y_remainder*temp2X + (1 - y_remainder)*temp1X
        return tempYX
    end
end

-- Derivative of the error with regards to template element (i,j)
function calculateTemplateGradient(i, j, input, gradOutput, scale)
    local trans_x = input[1]
    local trans_y = input[2]
    local prob = input[3]

    local row_low = math.floor(i + trans_y)
    local col_low = math.floor(j + trans_x)
    local row_high = row_low + 1
    local col_high = col_low + 1

    local low_x = math.floor(trans_x)
    local low_y = math.floor(trans_y)
    local high_x = low_x + 1
    local high_y = low_y + 1

    local result = 0
    result = result + getTensorValue(gradOutput, row_low, col_low)*(1 - math.abs(low_y - trans_y))*(1 - math.abs(low_x - trans_x))
    result = result + getTensorValue(gradOutput, row_low, col_high)*(1 - math.abs(low_y - trans_y))*(1 - math.abs(high_x - trans_x))
    result = result + getTensorValue(gradOutput, row_high, col_low)*(1 - math.abs(high_y - trans_y))*(1 - math.abs(low_x - trans_x))
    result = result + getTensorValue(gradOutput, row_high, col_high)*(1 - math.abs(high_y - trans_y))*(1 - math.abs(high_x - trans_x))
    result = result * prob * scale
    return result
end

function TemplateCapsule:zeroGradParameters()
    self.gradTemplate:zero()
end

function TemplateCapsule:updateParameters(learningRate)
    self.template:add(-learningRate, self.gradTemplate)
end

function TemplateCapsule:parameters()
    return {self.template}, {self.gradTemplate}
end

function TemplateCapsule:reset()
    self.template = torch.randn(self.template:size())
end