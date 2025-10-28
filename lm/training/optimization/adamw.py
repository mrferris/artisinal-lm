import torch.optim as optim
import torch
import math

class AdamW(optim.Optimizer):
    """
    AdamW Optimization implementation.
    """

    def __init__(self, params, lr, weight_decay, betas: tuple[float], eps) -> None:

        self.learning_rate = lr
        self.beta_1 = betas[0]
        self.beta_2 = betas[1]
        self.weight_decay = weight_decay
        self.epsilon = eps
        self.step_count = 1

        super().__init__(params, defaults={
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps
        })        

    def step(self):

        for group in self.param_groups:
            for param in group["params"]:

                # Optimizer class stores a state for each param            
                state = self.state[param]
                if len(state) == 0:
                    state["moment"] = torch.zeros_like(param, requires_grad=False)
                    state["variance"] = torch.zeros_like(param, requires_grad=False)

                moment: torch.Tensor = state["moment"]
                variance: torch.Tensor = state["variance"]

                grad: torch.Tensor = param.grad

                moment.mul_(self.beta_1).add_(grad, alpha=(1 - self.beta_1))
                variance.mul_(self.beta_2).addcmul_(grad, grad, value=(1 - self.beta_2))

                learning_rate_bias = math.sqrt(1 - (self.beta_2 ** self.step_count))
                learning_rate_bias /= (1 - (self.beta_1 ** self.step_count))
                adjusted_learning_rate = self.learning_rate * learning_rate_bias

                param.data.addcdiv_(moment, variance.sqrt().add_(self.epsilon), value=-adjusted_learning_rate)

                param.data.mul_(1 - (self.weight_decay * self.learning_rate))

        self.step_count += 1

    def set_learning_rate(self, lr: float) -> None:

        self.learning_rate = lr



            

            
