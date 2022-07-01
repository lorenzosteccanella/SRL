from collections import deque

import numpy as np
import torch
import wandb


def minibatch_deep_A(dataset, model, config):
    model.encoder.eval()
    optimizer = torch.optim.AdamW(model.action_encoder.parameters(), weight_decay=0, lr=config["l_rate"], amsgrad=config["amsgrad"])
    print("TRAINING ACTION MODEL")
    a_n_steps = 0
    for epoch in range(config["epochs_action_mod"]):
        if config["max_n_steps_action_mod"] is not None:
            if a_n_steps > config["max_n_steps_action_mod"]:
                break
        objective2_epoch = 0
        n_batch_in_epoch = 0
        max_error = 0
        for n_batch, batch in enumerate(dataset):
            try:
                x1_o, x1_a_o, x2_o, x2_a_o, d_obj, x1_c, x1_a_c, x2_c, x2_a_c, d, x1_f, x1_a_f, x2_f, x2_a_f = batch
                d_c_next_state = model.encode_state_next_state(x1_f, x2_f, x1_a_f)
                optimizer.zero_grad()
                objective2 = ((d_c_next_state[0] - d_c_next_state[1]) ** 2).sum()   # WARNING NOT SURE IF SUM OR MEAN
                loss = objective2
                loss.backward()
                optimizer.step(); a_n_steps += 1
                objective2_epoch += objective2.detach()
                n_batch_in_epoch = (n_batch + 1)
                max_error = max(max_error, np.max((d_c_next_state[0] - d_c_next_state[1]).detach().numpy()))
                if config["wandb_record"]:
                    wandb.log({"train/objective2": objective2})
            except KeyboardInterrupt:
                return None, None

        print(epoch, objective2_epoch/n_batch_in_epoch, max_error, end="\n\n")

    return objective2_epoch/n_batch_in_epoch


def minibatch_deep_A_ER(er, model, config):
    optimizer = torch.optim.AdamW(model.action_encoder.parameters(), weight_decay=0, lr=config["l_rate"], amsgrad=config["amsgrad"])
    print("TRAINING ACTION MODEL")
    a_n_steps = config["max_n_steps"]
    objective2_wa = deque(maxlen=1000)
    max_error = 0
    for n_steps in range(config["max_n_steps_action_mod"]):
        batch = er.dataset.sample_random_unique_batch(config["batch_size_action_mod"], sample_attrs=["x1_o", "x2_o", "x1_a_o", "x2_a_o", "d_obj",
                                                                         "x1_c", "x2_c", "x1_a_c", "x2_a_c", "d_c",
                                                                         "x1_f", "x2_f", "x1_a_f", "x2_a_f"])
        x1_o, x2_o, x1_a_o, x2_a_o, d_obj, x1_c, x2_c, x1_a_c, x2_a_c, d, x1_f, x2_f, x1_a_f, x2_a_f = batch[1]
        d_c_next_state = model.encode_state_next_state(x1_f, x2_f, x1_a_f)
        optimizer.zero_grad()
        objective2 = ((d_c_next_state[0] - d_c_next_state[1]) ** 2).mean()   # WARNING NOT SURE IF SUM OR MEAN
        loss = objective2
        loss.backward()
        optimizer.step()
        if config["wandb_record"]:
            wandb.log({"train/objective2": objective2})
        objective2_wa.append(objective2.detach().numpy())
        max_error = max(max_error, np.max((d_c_next_state[0] - d_c_next_state[1]).detach().numpy()))
        if n_steps % 1000 == 0:
            print(sum(objective2_wa)/len(objective2_wa), max_error, end="\n\n")
            max_error = 0

    return sum(objective2_wa)/len(objective2_wa)
