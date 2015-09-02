require 'nn'

require 'TemplateCapsule'

caps = nn.TemplateCapsule({2,2}, 5)
criterion = nn.MSECriterion()

caps.template = torch.eye(2)
input = torch.Tensor({0, 0, 1})
output = caps:forward(input)

x = torch.Tensor(output:size()):copy(output)

input =  torch.Tensor({0.5, 0, 1})
output = caps:forward(input)

err = criterion:forward(output, x)

df_do = criterion:backward(output, x)
caps:backward(input, df_do)