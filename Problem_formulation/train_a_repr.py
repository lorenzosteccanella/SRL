from collections import deque
import torch
import wandb
from Utils.Utils import *


def minibatch_deep(dataset, model, config):

    optimizer = torch.optim.AdamW(model.encoder.parameters(), weight_decay=0, lr=config["l_rate"], amsgrad=config["amsgrad"])
    print("TRAINING")
    n_steps = 0
    for epoch in range(config["epochs"]):
        if config["max_n_steps"] is not None:
            if n_steps > config["max_n_steps"]:
                break
        loss_epoch = 0
        for n_batch, batch in enumerate(dataset):
            try:
                x1_o, x1_a_o, x2_o, x2_a_o, d_obj, x1_c, x1_a_c, x2_c, x2_a_c, d, x1_f, x1_a_f, x2_f, x2_a_f = batch
                d = torch.tensor(d)
                zeros = torch.zeros_like(d)
                d_o = model.encode_state_dist(x1_o, x2_o)
                objective = (1 / d ** 2) * (d_o - d) ** 2
                cost_function = (1 / d ** 2) * torch.max(d_o - d, zeros) ** 2
                objective = objective.sum()
                constraints = cost_function.sum()

                w_anealing = config["constr_weight"]  # min(config["constr_weight"], (0.1 * pow(1+1e-4, n_steps)))
                loss = objective + w_anealing * constraints
                loss.backward()
                # plot_grad_flow(model.encoder.named_parameters())
                optimizer.step()
                optimizer.zero_grad()
                n_steps += 1
                loss_epoch += loss.detach()
                n_batch_in_epoch = (n_batch + 1)
                if config["wandb_record"]:
                    wandb.log({"train/loss": loss},
                              step=(config["max_n_steps"]+a_n_steps))

            except KeyboardInterrupt:
                return None, None

        print(epoch, w_anealing, loss_epoch / n_batch_in_epoch, end="\n\n")

    return loss_epoch/n_batch_in_epoch


def minibatch_deep_WE(dataset, model, config):

    optimizer = torch.optim.AdamW(model.encoder.parameters(), weight_decay=0, lr=config["l_rate"], amsgrad=config["amsgrad"])
    print("TRAINING")
    n_steps = 0
    for epoch in range(config["epochs"]):
        if config["max_n_steps"] is not None:
            if n_steps > config["max_n_steps"]:
                break
        loss_epoch = 0
        for n_batch, batch in enumerate(dataset):
            try:
                x1_o, x1_a_o, x2_o, x2_a_o, d_obj, x1_c, x1_a_c, x2_c, x2_a_c, d, x1_f, x1_a_f, x2_f, x2_a_f = batch
                d_o = model.encode_state_dist(x1_o, x2_o)
                #d_o = model.encode_state_action_dist(x1_o, x1_a_o, x2_o, x2_a_o)
                optimizer.zero_grad()
                w_int_anealing = config["weight_interpol"] #min(config["weight_interpol"], (0.1 * pow(1 + 1e-4, n_steps)))
                objective = (100 / d ** w_int_anealing) * (1 / d ** 2) * (d_o - d) ** 2
                loss = objective.mean()
                loss.backward()
                #plot_grad_flow(model.encoder.named_parameters())
                optimizer.step(); n_steps += 1
                n_steps += 1
                loss_epoch += loss.detach()
                n_batch_in_epoch = (n_batch + 1)
                if config["wandb_record"]:
                    wandb.log({"train/loss": loss},
                              step=n_steps)

            except KeyboardInterrupt:
                return None, None

        print(epoch, w_int_anealing, loss_epoch / n_batch_in_epoch, end="\n\n")

    return loss_epoch/n_batch_in_epoch


def minibatch_deep_PER(per, model, config):

    optimizer = torch.optim.AdamW(model.encoder.parameters(), weight_decay=0, lr=config["l_rate"], amsgrad=config["amsgrad"])
    print("TRAINING")
    objective_wa = deque(maxlen=1000)
    constraints_wa = deque(maxlen=1000)
    loss_wa = deque(maxlen=1000)
    for n_steps in range(config["max_n_steps"]):
        batch = per.dataset.sample_batch(config["batch_size"], sample_attrs=["x1_o", "x2_o", "x1_a_o", "x2_a_o", "d_obj",
                                                                         "x1_c", "x2_c", "x1_a_c", "x2_a_c", "d_c"])
        x1_o, x2_o, x1_a_o, x2_a_o, d_obj, x1_c, x2_c, x1_a_c, x2_a_c, d = batch[1]
        index = batch[2]
        is_weight = torch.tensor(batch[3])
        d = torch.tensor(d)
        zeros = torch.zeros_like(d)
        d_o = model.encode_state_dist(x1_o, x2_o)
        objective = (1/d**2) * (d_o - d)**2
        cost_function = (1/d**2) * torch.max(d_o - d, zeros)**2
        prioritization = (1/d) * torch.max(d_o - d, zeros).abs().detach()
        objective = (is_weight * objective).sum()
        constraints = cost_function.sum()

        w_anealing = config["constr_weight"]# min(config["constr_weight"], (0.1 * pow(1+1e-4, n_steps)))
        loss = objective + w_anealing * constraints
        loss.backward()
        #plot_grad_flow(model.encoder.named_parameters())
        optimizer.step()
        optimizer.zero_grad()

        error = (torch.min(prioritization, torch.ones_like(d))).detach().numpy()
        per.dataset.update_priority(error, index)

        if config["wandb_record"]:
            wandb.log({"train/loss": loss,
                       "train/objective": objective,
                       "train/constraints": constraints,
                       "train/weight_anealing": w_anealing},
                      step=n_steps)

        objective_wa.append(objective.detach().numpy())
        constraints_wa.append((constraints).detach().numpy())
        loss_wa.append(loss.detach().numpy())

        if n_steps % 1000 == 0:
            print(w_anealing, sum(objective_wa)/len(objective_wa), sum(constraints_wa)/len(constraints_wa),
                  sum(loss_wa)/len(loss_wa), end="\n\n")

    return sum(loss_wa)/len(loss_wa)
