import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

import sys
sys.path.append('../../../src')
import least

class PseudoLabel(nn.Module):
    def __init__(self, model, optimizer, alpha=0.1, beta=0.9, gamma=1, episodic=False, mask_algo='None'):
        super().__init__()
        """
        Hparams
        -------
        alpha (float) : learning rate coefficient
        beta (float) : threshold
        gamma (int) : number of updates
        """
        self.model = model
        self.optimizer = optimizer
        self.beta = beta
        self.steps = gamma # 1 or 3
        assert self.steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.mask_algo = mask_algo
        self.iter_idx = 0
    
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, adapt=True):
        if adapt:
            if self.episodic:
                self.reset()

            for _ in range(self.steps):
                # if self.hparams['cached_loader']:
                #     outputs = self.forward_and_adapt(x, self.model.classifier, self.optimizer)
                # else:
                self.model.eval()
                outputs = self.forward_and_adapt(x, self.model, self.optimizer)
                self.model.train()
        else:
            # if self.hparams['cached_loader']:
            #     outputs = self.model.classifier(x)
            # else:
            outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        # forward
        optimizer.zero_grad()
        outputs = model(x)
        # adapt
        py, y_prime = F.softmax(outputs, dim=-1).max(1)
        flag = py > self.beta
        
        loss = F.cross_entropy(outputs[flag], y_prime[flag])
        loss.backward()
        # Call LEAST to modify optimizer and trainable layers
        optimizer = least.get_new_optimizer(self.mask_algo, model, optimizer, self.iter_idx)
        self.iter_idx += 1
        optimizer.step()
        return outputs

    # def configure_model_optimizer(self, algorithm, alpha):
    #     adapted_algorithm = copy.deepcopy(algorithm)
    #     optimizer = torch.optim.Adam(
    #         adapted_algorithm.parameters(), 
    #         lr=algorithm.hparams["lr"]  * alpha,
    #         weight_decay=algorithm.hparams['weight_decay']
    #     )
    #     return adapted_algorithm, optimizer

    # def predict(self, x, adapt=False):
    #     return self(x, adapt)

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.iter_idx = 0

def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
                    print(nm, np)
    return params, names

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = copy.deepcopy(model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # enable all trainable
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        else:
            m.requires_grad_(True)
    return model

# def check_model(model):
#     """Check model for compatability with tent."""
#     is_training = model.training
#     assert is_training, "tent needs train mode: call model.train()"
#     param_grads = [p.requires_grad for p in model.parameters()]
#     has_any_params = any(param_grads)
#     has_all_params = all(param_grads)
#     assert has_any_params, "tent needs params to update: " \
#                            "check which require grad"
#     assert not has_all_params, "tent should not update all params: " \
#                                "check which require grad"
#     has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
#     assert has_bn, "tent needs normalization for its optimization"