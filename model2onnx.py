import torch
import onnx
import torch.nn.parallel
from ConvNextModel.model import GazeLSTM
from onnxoptimizer import optimize

model = GazeLSTM()
model = torch.nn.DataParallel(model)
checkpoint = torch.load('ConvNextModel/model_best_Gaze360.pth.tar', map_location=torch.device('cuda'))
model.load_state_dict(checkpoint['state_dict'])

if isinstance(model, torch.nn.DataParallel):
    model = model.module
model.eval()
input_image = torch.zeros(1,7,3,224,224)
input_shape = (1,7,3,224,224)
dummy_input = torch.randn(input_shape).cuda()
input_names = ["input"]  # Name of the input to your model
output_names = ["output"]  # Name of the output from your model
dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
# dynamic_axes is used to specify the variable batch_size
# you can set it to None if the model doesn't have variable batch size

# convert pytorch to onnx
torch.onnx.export(model, dummy_input, 'model.onnx', input_names=input_names,
                  output_names=output_names, dynamic_axes=dynamic_axes)

model = onnx.load('model.onnx')

# Optimize the model
optimized_model = optimize(model)

# Save the optimized model
onnx.save(optimized_model, 'optimized_model.onnx')