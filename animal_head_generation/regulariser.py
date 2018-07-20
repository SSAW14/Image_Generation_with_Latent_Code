import numpy as np

def tv_norm(x, beta=2.0, verbose=False, operator='naive'):
  """
  Compute the total variation norm and its gradient.
  
  The total variation norm is the sum of the image gradient
  raised to the power of beta, summed over the image.
  We approximate the image gradient using finite differences.
  We use the total variation norm as a regularizer to encourage
  smoother images.

  Inputs:
  - x: numpy array of shape (N, C, H, W)

  Returns a tuple of:
  - loss: Scalar giving the value of the norm
  - dx: numpy array of shape (N, C, H, W) giving gradient of the loss
        with respect to the input x.
  """
  if operator == 'naive':
    x_diff = x[:, :, :-1, :-1] - x[:, :, :-1, 1:]
    y_diff = x[:, :, :-1, :-1] - x[:, :, 1:, :-1]
  elif operator == 'sobel':
    x_diff  =  x[:, :, :-2, 2:]  + 2 * x[:, :, 1:-1, 2:]  + x[:, :, 2:, 2:]
    x_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, 1:-1, :-2] + x[:, :, 2:, :-2]
    y_diff  =  x[:, :, 2:, :-2]  + 2 * x[:, :, 2:, 1:-1]  + x[:, :, 2:, 2:]
    y_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, :-2, 1:-1] + x[:, :, :-2, 2:]
  elif operator == 'sobel_squish':
    x_diff  =  x[:, :, :-2, 1:-1]  + 2 * x[:, :, 1:-1, 1:-1]  + x[:, :, 2:, 1:-1]
    x_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, 1:-1, :-2] + x[:, :, 2:, :-2]
    y_diff  =  x[:, :, 1:-1, :-2]  + 2 * x[:, :, 1:-1, 1:-1]  + x[:, :, 1:-1, 2:]
    y_diff -= x[:, :, :-2, :-2] + 2 * x[:, :, :-2, 1:-1] + x[:, :, :-2, 2:]
  else:
    assert False, 'Unrecognized operator %s' % operator
  grad_norm2 = x_diff ** 2.0 + y_diff ** 2.0
  grad_norm2[grad_norm2 < 1e-3] = 1e-3
  grad_norm_beta = grad_norm2 ** (beta / 2.0)
  loss = np.sum(grad_norm_beta)
  dgrad_norm2 = (beta / 2.0) * grad_norm2 ** (beta / 2.0 - 1.0)
  dx_diff = 2.0 * x_diff * dgrad_norm2
  dy_diff = 2.0 * y_diff * dgrad_norm2
  dx = np.zeros_like(x)
  if operator == 'naive':
    dx[:, :, :-1, :-1] += dx_diff + dy_diff
    dx[:, :, :-1, 1:] -= dx_diff
    dx[:, :, 1:, :-1] -= dy_diff
  elif operator == 'sobel':
    dx[:, :, :-2, :-2] += -dx_diff - dy_diff
    dx[:, :, :-2, 1:-1] += -2 * dy_diff
    dx[:, :, :-2, 2:] += dx_diff - dy_diff
    dx[:, :, 1:-1, :-2] += -2 * dx_diff
    dx[:, :, 1:-1, 2:] += 2 * dx_diff
    dx[:, :, 2:, :-2] += dy_diff - dx_diff
    dx[:, :, 2:, 1:-1] += 2 * dy_diff
    dx[:, :, 2:, 2:] += dx_diff + dy_diff
  elif operator == 'sobel_squish':
    dx[:, :, :-2, :-2] += -dx_diff - dy_diff
    dx[:, :, :-2, 1:-1] += dx_diff -2 * dy_diff
    dx[:, :, :-2, 2:] += -dy_diff
    dx[:, :, 1:-1, :-2] += -2 * dx_diff + dy_diff
    dx[:, :, 1:-1, 1:-1] += 2 * dx_diff + 2 * dy_diff
    dx[:, :, 1:-1, 2:] += dy_diff
    dx[:, :, 2:, :-2] += -dx_diff
    dx[:, :, 2:, 1:-1] += dx_diff

  
  def helper(name, x):
    num_nan = np.isnan(x).sum()
    num_inf = np.isinf(x).sum()
    num_zero = (x == 0).sum()
    print '%s: NaNs: %d infs: %d zeros: %d' % (name, num_nan, num_inf, num_zero)
  
  if verbose:
    print '-' * 40
    print 'tv_norm debug output'
    helper('x', x)
    helper('x_diff', x_diff)
    helper('y_diff', y_diff)
    helper('grad_norm2', grad_norm2)
    helper('grad_norm_beta', grad_norm_beta)
    helper('dgrad_norm2', dgrad_norm2)
    helper('dx_diff', dx_diff)
    helper('dy_diff', dy_diff)
    helper('dx', dx)
    print
  
  return loss, dx

def grad_l1_norm(x):
  grad_x = np.zeros_like(x)
  grad_x[x > 0] = 1
  grad_x[x < 0] = -1
  return grad_x

def gdl_norm(x, xt, alpha = 1.0):
  """
  Compute the gradient difference loss and its gradient.
  
  Please refer to the paper "Deep multi-scale video prediction beyond mean square error" for details.
  We use the gradient difference loss as a regularizer to encourage smoother images and similar edge to target images.

  Inputs:
  - x: numpy array of shape (N, C, H, W)
  - xt: numpy array of shape (N, C, H, W)
  - alpha: a parameter bigger than 1

  Returns a tuple of:
  - loss: Scalar giving the value of the norm
  - dx: numpy array of shape (N, C, H, W) giving gradient of the loss
        with respect to the input x.
  """
  grad_x_xt = xt[:, :, :, 1:] - xt[:, :, :, :-1]
  grad_y_xt = xt[:, :, 1:, :] - xt[:, :, :-1, :]

  grad_x_x = x[:, :, :, 1:] - x[:, :, :, :-1]
  grad_y_x = x[:, :, 1:, :] - x[:, :, :-1, :]

  grad_norm_x_alpha = abs(grad_x_x - grad_x_xt) ** alpha
  grad_norm_y_alpha = abs(grad_y_x - grad_y_xt) ** alpha
  loss = np.sum(grad_norm_x_alpha) + np.sum(grad_norm_y_alpha)

  diff_grad_x = alpha * (abs(grad_x_x - grad_x_xt) ** (alpha-1)) * grad_l1_norm(grad_x_x - grad_x_xt)
  diff_grad_y = alpha * (abs(grad_y_x - grad_y_xt) ** (alpha-1)) * grad_l1_norm(grad_y_x - grad_y_xt)

  dx = np.zeros_like(x)

  dx[:, :, :, 1:] += diff_grad_x
  dx[:, :, :, :-1] -= diff_grad_x
  dx[:, :, 1:, :] += diff_grad_y
  dx[:, :, :-1, :] -= diff_grad_y

  return loss, dx










