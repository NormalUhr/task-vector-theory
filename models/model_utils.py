import torch


def ptv(parameters, grad=False):
    vec = []
    grad_vec = []
    for param in parameters:
        vec.append(param.view(-1))
        if grad:
            grad_vec.append(param.grad.view(-1))
    if grad:
        return torch.cat(vec), torch.cat(grad_vec)
    return torch.cat(vec)


def vtp(vec, parameters, grad=False, append=False):
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        if grad:
            if append:
                param.grad.copy_(param.grad + vec[pointer:pointer + num_param].view_as(param).data)
            else:
                param.grad.copy_(vec[pointer:pointer + num_param].view_as(param).data)
        else:
            param.data.copy_(vec[pointer:pointer + num_param].view_as(param).data)

        # Increment the pointer
        pointer += num_param