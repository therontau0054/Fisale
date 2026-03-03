import os
import sys
import toml
import torch
import pickle

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import  CosineAnnealingLR
from data_loaders.venous_valve import DataLoader_Venous_Valve
from model.fisale import Fisale
from model.loss import relative_l2_loss, mse_l2_loss
from utils.normalizer import Normalizer

def calculate_losses(x, y, dim = 2, criterion = torch.nn.MSELoss(), root = False):
    root_factor = 0.5 if root else 1
    position_loss = criterion(x[:, :, :dim], y[:, :, :dim]).item() ** root_factor
    physics_quantities_losses = [
        criterion(x[:, :, dim + i : dim + i + 1], y[:, :, dim + i : dim + i + 1]).item() ** root_factor
        for i in range(x.shape[-1] - dim)
    ]
    return [position_loss] + physics_quantities_losses

def train(data_path, net, config, normalizers, criterion):
    torch.manual_seed(config["venous_valve"]["hyperparameter"]["seed"])

    lr = config["venous_valve"]["hyperparameter"]["lr"]
    lr_decay = config["venous_valve"]["hyperparameter"]["lr_decay"]
    epochs = config["venous_valve"]["hyperparameter"]["epochs"]
    weight_decay = config["venous_valve"]["hyperparameter"]["weight_decay"]
    batch_time_step = config["venous_valve"]["hyperparameter"]["batch_time_step"]
    noise_sigma = config["venous_valve"]["hyperparameter"]["noise_sigma"]

    optimizer = torch.optim.AdamW(
        net.parameters(), 
        lr = lr, 
        weight_decay = weight_decay
    )

    train_data_loader = DataLoader_Venous_Valve(
        data_path = data_path,
        mode = "train"
    )
    eval_data_loader = DataLoader_Venous_Valve(
        data_path = data_path,
        mode = "eval"
    )

    train_datas = DataLoader(
        train_data_loader,
        batch_size = 1,
        shuffle = True,
        num_workers = 0
    )

    eval_datas = DataLoader(
        eval_data_loader,
        batch_size = 1,
        num_workers = 0
    )

    scheduler = CosineAnnealingLR(
        optimizer, 
        epochs, 
        eta_min = lr / lr_decay
    )
    net.to(device)

    # normalization
    for train_data in train_datas:
        current_solid, current_fluid, current_interface, \
        target_solid, target_fluid, target_interface, \
        next_solid, next_fluid, next_interface = train_data
        current_solid = normalizers["current_solid"](current_solid[0].to(device), accumulate = True)
        current_fluid = normalizers["current_fluid"](current_fluid[0].to(device), accumulate = True)
        current_interface = normalizers["current_interface"](current_interface[0].to(device), accumulate = True)
        target_solid = normalizers["target_solid"](target_solid[0].to(device), accumulate = True)
        target_fluid = normalizers["target_fluid"](target_fluid[0].to(device), accumulate = True)
        target_interface = normalizers["target_interface"](target_interface[0].to(device), accumulate = True)
        next_solid = normalizers["next_solid"](next_solid[0].to(device), accumulate = True)
        next_fluid = normalizers["next_fluid"](next_fluid[0].to(device), accumulate = True)
        next_interface = normalizers["next_interface"](next_interface[0].to(device), accumulate = True)

    for key, value in normalizers.items():
        value.save_variable(f"{assets_root}/normalizers/{model_name}_{key}_normalizer_{version}.pth")

    min_eval_loss = 1e10
    for epoch in range(epochs):
        net.train()
        mean_train_losses = {
            "loss": 0.,
            "solid_loss": 0.,
            "fluid_loss": 0.,
            "interface_loss": 0.
        }
        batch_num = 0
        for i, train_data in enumerate(train_datas):
            current_solid, current_fluid, current_interface, \
            target_solid, target_fluid, target_interface, \
            _, _, _ = train_data

            total_time_steps = current_solid.shape[1]

            for j in range(0, total_time_steps, batch_time_step):
                _current_solid = normalizers["current_solid"](current_solid[0][j : j + batch_time_step].to(device))
                _current_fluid = normalizers["current_fluid"](current_fluid[0][j : j + batch_time_step].to(device))
                _current_interface = normalizers["current_interface"](current_interface[0][j : j + batch_time_step].to(device))
                _target_solid = normalizers["target_solid"](target_solid[0][j : j + batch_time_step].to(device))
                _target_fluid = normalizers["target_fluid"](target_fluid[0][j : j + batch_time_step].to(device))
                _target_interface = normalizers["target_interface"](target_interface[0][j : j + batch_time_step].to(device))

                pred_solid, pred_fluid, pred_interface = net(
                    _current_solid + torch.randn_like(_current_solid) * noise_sigma,
                    _current_fluid + torch.randn_like(_current_fluid) * noise_sigma,
                    _current_interface + torch.randn_like(_current_interface) * noise_sigma
                )

                solid_loss = criterion(pred_solid, _target_solid)
                fluid_loss = criterion(pred_fluid, _target_fluid)
                interface_loss = criterion(pred_interface, _target_interface)

                loss = solid_loss + fluid_loss + interface_loss

                print(
                    f"epoch: {epoch},",
                    f"lr: {scheduler.get_last_lr()[0]},",
                    f"sample: {i},",
                    f"batch: {j // batch_time_step},",
                    f"train_loss: {loss.item()},",
                    f"train_solid_loss: {solid_loss.item()},",
                    f"train_fluid_loss: {fluid_loss.item()},",
                    f"train_interface_loss: {interface_loss.item()}"
                )
            
                mean_train_losses["loss"] += loss.item()
                mean_train_losses["solid_loss"] += solid_loss.item()
                mean_train_losses["fluid_loss"] += fluid_loss.item()
                mean_train_losses["interface_loss"] += interface_loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_num += 1
        scheduler.step()

        mean_train_losses["loss"] /= batch_num
        mean_train_losses["solid_loss"] /= batch_num
        mean_train_losses["fluid_loss"] /= batch_num
        mean_train_losses["interface_loss"] /= batch_num

        print(
            f"epoch: {epoch},",
            f"mean_train_loss: {mean_train_losses['loss']},",
            f"mean_train_solid_loss: {mean_train_losses['solid_loss']},", 
            f"mean_train_fluid_loss: {mean_train_losses['fluid_loss']},", 
            f"mean_train_interface_loss: {mean_train_losses['interface_loss']}"
        )

        if (epoch + 1) % config["venous_valve"]["hyperparameter"]["eval_epoch_gap"] != 0:
            continue
        
        net.eval()

        mean_eval_losses = {
            "loss": 0.,
            "solid_losses": [],
            "fluid_losses": [],
            "interface_losses": []
        }

        with torch.no_grad():
            batch_num = 0
            for i, eval_data in enumerate(eval_datas):
                current_solid, current_fluid, current_interface, \
                _, _, _, \
                next_solid, next_fluid, next_interface = eval_data

                total_time_steps = current_solid.shape[1]

                for j in range(total_time_steps):
                    if j == 0:
                        _current_solid = normalizers["current_solid"](current_solid[0][j : j + 1].to(device))
                        _current_fluid = normalizers["current_fluid"](current_fluid[0][j : j + 1].to(device))
                        _current_interface = normalizers["current_interface"](current_interface[0][j : j + 1].to(device))
                    else:
                        _current_solid = normalizers["current_solid"](pred_solid.to(device))
                        _current_fluid = normalizers["current_fluid"](pred_fluid.to(device))
                        _current_interface = normalizers["current_interface"](pred_interface.to(device))

                    _next_solid = next_solid[0][j : j + 1].to(device)
                    _next_fluid = next_fluid[0][j : j + 1].to(device)
                    _next_interface = next_interface[0][j : j + 1].to(device)

                    pred_solid, pred_fluid, pred_interface = net(
                        _current_solid, _current_fluid, _current_interface
                    )

                    pred_solid = normalizers["target_solid"].inverse(pred_solid) + normalizers["current_solid"].inverse(_current_solid)
                    pred_fluid = normalizers["target_fluid"].inverse(pred_fluid) + normalizers["current_fluid"].inverse(_current_fluid)
                    pred_interface = normalizers["target_interface"].inverse(pred_interface) + normalizers["current_interface"].inverse(_current_interface)

                    solid_losses = calculate_losses(pred_solid, _next_solid, criterion = criterion, root = True)
                    fluid_losses = calculate_losses(pred_fluid, _next_fluid, criterion = criterion, root = True)
                    interface_losses = calculate_losses(pred_interface, _next_interface, criterion = criterion, root = True)

                    loss = (criterion(normalizers["next_solid"](pred_solid), normalizers["next_solid"](_next_solid)) \
                        + criterion(normalizers["next_fluid"](pred_fluid), normalizers["next_fluid"](_next_fluid)) \
                        + criterion(normalizers["next_interface"](pred_interface), normalizers["next_interface"](_next_interface))).item() / 3

                    print(
                        f"epoch: {epoch},",
                        f"sample: {i},",
                        f"frame: {j},"
                        f"eval_loss: {loss},",
                        f"eval_solid_losses: {solid_losses},",
                        f"eval_fluid_losses: {fluid_losses},",
                        f"eval_interface_losses: {interface_losses}"
                    )
                
                    mean_eval_losses["loss"] += loss
                    mean_eval_losses["solid_losses"] = solid_losses if len(mean_eval_losses["solid_losses"]) == 0 \
                        else list(map(lambda x,y: x + y, mean_eval_losses["solid_losses"], solid_losses))
                    mean_eval_losses["fluid_losses"] = fluid_losses if len(mean_eval_losses["fluid_losses"]) == 0 \
                        else list(map(lambda x,y: x + y, mean_eval_losses["fluid_losses"], fluid_losses))
                    mean_eval_losses["interface_losses"] = interface_losses if len(mean_eval_losses["interface_losses"]) == 0 \
                        else list(map(lambda x,y: x + y, mean_eval_losses["interface_losses"], interface_losses))

                    batch_num += 1

            mean_eval_losses["loss"] /= batch_num
            mean_eval_losses["solid_losses"] = list(map(lambda x: x / batch_num, mean_eval_losses["solid_losses"]))
            mean_eval_losses["fluid_losses"] = list(map(lambda x: x / batch_num, mean_eval_losses["fluid_losses"]))
            mean_eval_losses["interface_losses"] = list(map(lambda x: x / batch_num, mean_eval_losses["interface_losses"]))

            print(
                f"epoch: {epoch},",
                f"mean_eval_loss: {mean_eval_losses['loss']},",
                f"mean_eval_solid_losses: {mean_eval_losses['solid_losses']},",
                f"mean_eval_fluid_losses: {mean_eval_losses['fluid_losses']},",
                f"mean_eval_interface_losses: {mean_eval_losses['interface_losses']}"
            )
            
            if mean_eval_losses["loss"] < min_eval_loss:
                min_eval_loss = mean_eval_losses["loss"]
                torch.save(
                    net.state_dict(), 
                    f"{assets_root}/checkpoints/{model_name}_best_model_{version}.pth"
                )


def test(data_path, net, config, normalizers, criterion):
    ret_save_path = f"{data_root}/ret/{model_name}_venous_valve_{version}.pkl"

    test_data_loader = DataLoader_Venous_Valve(
        data_path = data_path,
        mode = "test"
    )

    test_datas = DataLoader(
        test_data_loader,
        batch_size = 1,
        num_workers = 0
    )

    net.to(device)
    net.load_state_dict(
        torch.load(
            f"{assets_root}/checkpoints/{model_name}_best_model_{version}.pth", 
            weights_only = True, 
            map_location = device
        )
    )
    for key, value in normalizers.items():
        value.load_variable(f"{assets_root}/normalizers/{model_name}_{key}_normalizer_{version}.pth")
    net.eval()

    mean_test_losses = {
        "loss": 0.,
        "solid_losses": [],
        "fluid_losses": [],
        "interface_losses": []
    }

    with torch.no_grad():
        batch_num = 0
        ret_save_data = {}
        for i, test_data in enumerate(test_datas):
            current_solid, current_fluid, current_interface, \
            _, _, _, \
            next_solid, next_fluid, next_interface = test_data

            ret_save_data[i] = {
                "solid": [],
                "fluid": [],
                "interface": []
            }

            total_time_steps = current_solid.shape[1]

            for j in range(total_time_steps):
                if j == 0:
                    _current_solid = normalizers["current_solid"](current_solid[0][j : j + 1].to(device))
                    _current_fluid = normalizers["current_fluid"](current_fluid[0][j : j + 1].to(device))
                    _current_interface = normalizers["current_interface"](current_interface[0][j : j + 1].to(device))
                else:
                    _current_solid = normalizers["current_solid"](pred_solid.to(device))
                    _current_fluid = normalizers["current_fluid"](pred_fluid.to(device))
                    _current_interface = normalizers["current_interface"](pred_interface.to(device))

                _next_solid = next_solid[0][j : j + 1].to(device)
                _next_fluid = next_fluid[0][j : j + 1].to(device)
                _next_interface = next_interface[0][j : j + 1].to(device)

                pred_solid, pred_fluid, pred_interface = net(
                    _current_solid, _current_fluid, _current_interface
                )

                pred_solid = normalizers["target_solid"].inverse(pred_solid) + normalizers["current_solid"].inverse(_current_solid)
                pred_fluid = normalizers["target_fluid"].inverse(pred_fluid) + normalizers["current_fluid"].inverse(_current_fluid)
                pred_interface = normalizers["target_interface"].inverse(pred_interface) + normalizers["current_interface"].inverse(_current_interface)

                solid_losses = calculate_losses(pred_solid, _next_solid, criterion = criterion, root = True)
                fluid_losses = calculate_losses(pred_fluid, _next_fluid, criterion = criterion, root = True)
                interface_losses = calculate_losses(pred_interface, _next_interface, criterion = criterion, root = True)

                loss = (criterion(normalizers["next_solid"](pred_solid), normalizers["next_solid"](_next_solid)) \
                        + criterion(normalizers["next_fluid"](pred_fluid), normalizers["next_fluid"](_next_fluid)) \
                        + criterion(normalizers["next_interface"](pred_interface), normalizers["next_interface"](_next_interface))).item() / 3

                print(
                    f"sample: {i},",
                    f"frame: {j},",
                    f"test_loss: {loss},",
                    f"test_solid_losses: {solid_losses},",
                    f"test_fluid_losses: {fluid_losses},",
                    f"test_interface_losses: {interface_losses}"
                )
            
                mean_test_losses["loss"] += loss
                mean_test_losses["solid_losses"] = solid_losses if len(mean_test_losses["solid_losses"]) == 0 \
                    else list(map(lambda x,y: x + y, mean_test_losses["solid_losses"], solid_losses))
                mean_test_losses["fluid_losses"] = fluid_losses if len(mean_test_losses["fluid_losses"]) == 0 \
                    else list(map(lambda x,y: x + y, mean_test_losses["fluid_losses"], fluid_losses))
                mean_test_losses["interface_losses"] = interface_losses if len(mean_test_losses["interface_losses"]) == 0 \
                    else list(map(lambda x,y: x + y, mean_test_losses["interface_losses"], interface_losses))


                ret_save_data[i]["solid"].append(pred_solid.cpu().numpy())
                ret_save_data[i]["fluid"].append(pred_fluid.cpu().numpy())
                ret_save_data[i]["interface"].append(pred_interface.cpu().numpy())

                batch_num += 1

        mean_test_losses["loss"] /= batch_num
        mean_test_losses["solid_losses"] = list(map(lambda x: x / batch_num, mean_test_losses["solid_losses"]))
        mean_test_losses["fluid_losses"] = list(map(lambda x: x / batch_num, mean_test_losses["fluid_losses"]))
        mean_test_losses["interface_losses"] = list(map(lambda x: x / batch_num, mean_test_losses["interface_losses"]))

        print(
            f"mean_test_loss: {mean_test_losses['loss']},",
            f"mean_test_solid_losses: {mean_test_losses['solid_losses']},",
            f"mean_test_fluid_losses: {mean_test_losses['fluid_losses']},",
            f"mean_test_interface_losses: {mean_test_losses['interface_losses']}"
        )

        pickle.dump(ret_save_data, open(ret_save_path, "wb"))

        

def main(run_type):
    data_path = f"{data_root}/input"
    fisale_params = config["venous_valve"]["fisale"]
    net = Fisale(
        dim = fisale_params["dim"],
        input_quantity_dims = fisale_params["input_quantity_dims"],
        grid_num = fisale_params["grid_num"],
        hidden_dims = fisale_params["hidden_dims"],
        grid_shapes = fisale_params["grid_shapes"],
        coupling_steps = fisale_params["coupling_steps"],
        neighbors_nums = fisale_params["neighbors_nums"],
        heads_num = fisale_params["heads_num"],
        mlp_ratio = fisale_params["mlp_ratio"],
        dropout = fisale_params["dropout"],
        act = fisale_params["act"]
    )

    if config["venous_valve"]["hyperparameter"]["criterion"] == "relative l2":
        criterion = relative_l2_loss
    elif config["venous_valve"]["hyperparameter"]["criterion"] == "mse l2":
        criterion = mse_l2_loss
    else:
        raise NotImplementedError

    normalizers = {}
    for s1 in ["current", "target", "next"]:
        for s2 in ["solid", "fluid", "interface"]:
            normalizers[f"{s1}_{s2}"] = Normalizer(
                size = fisale_params["dim"] + fisale_params["input_quantity_dims"][s2],
                name = f"{s1}_{s2}",
                device = device
            )

    if run_type == "train":
        train(data_path, net, config, normalizers, criterion)
    elif run_type == "test":
        test(data_path, net, config, normalizers, criterion)


if __name__ == "__main__":
    config_path = "./config.toml"
    with open(config_path, 'r', encoding = 'utf-8') as f:
        config = toml.load(f)
    
    version = config["venous_valve"]["version"]
    assets_root = config["venous_valve"]["assets_root"]
    data_root = config["venous_valve"]["data_root"]
    model_name = config["venous_valve"]["hyperparameter"]["model_name"]
    device = config["venous_valve"]["hyperparameter"]["device"]
    log_path = f"{assets_root}/logs/{model_name}_log_{version}.log"
    log_writer = open(log_path, 'a', encoding = "utf8")

    sys.stdout = log_writer
    sys.stderr = log_writer

    print(config["venous_valve"]["description"])

    main("train")
    main("test")

    log_writer.close()
