import sys
import toml
import torch
import pickle

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import  CosineAnnealingLR
from data_loaders.flexible_wing import DataLoader_Flexible_Wing
from model.fisale import Fisale
from model.loss import relative_l2_loss, mse_l2_loss
from utils.normalizer import Normalizer

def train(data_path, net, config, normalizers, criterion):
    torch.manual_seed(config["flexible_wing"]["hyperparameter"]["seed"])

    lr = config["flexible_wing"]["hyperparameter"]["lr"]
    lr_decay = config["flexible_wing"]["hyperparameter"]["lr_decay"]
    epochs = config["flexible_wing"]["hyperparameter"]["epochs"]
    weight_decay = config["flexible_wing"]["hyperparameter"]["weight_decay"]
    batch_size = config["flexible_wing"]["hyperparameter"]["batch_size"]

    optimizer = torch.optim.AdamW(
        net.parameters(), 
        lr = lr, 
        weight_decay = weight_decay
    )

    train_data_loader = DataLoader_Flexible_Wing(
        data_path = data_path,
        mode = "train"
    )
    eval_data_loader = DataLoader_Flexible_Wing(
        data_path = data_path,
        mode = "eval"
    )

    train_datas = DataLoader(
        train_data_loader,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0
    )

    eval_datas = DataLoader(
        eval_data_loader,
        batch_size = batch_size,
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
        init_solid, init_fluid, init_interface, \
        final_solid, final_fluid, final_interface = train_data
        init_solid = normalizers["init_solid"](init_solid.to(device), accumulate = True)
        init_fluid = normalizers["init_fluid"](init_fluid.to(device), accumulate = True)
        init_interface = normalizers["init_interface"](init_interface.to(device), accumulate = True)
        final_solid = normalizers["final_solid"](final_solid.to(device), accumulate = True)
        final_fluid = normalizers["final_fluid"](final_fluid.to(device), accumulate = True)
        final_interface = normalizers["final_interface"](final_interface.to(device), accumulate = True)

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
        for i, train_data in enumerate(train_datas):
            init_solid, init_fluid, init_interface, \
            final_solid, final_fluid, final_interface = train_data

            init_solid = normalizers["init_solid"](init_solid.to(device))
            init_fluid = normalizers["init_fluid"](init_fluid.to(device))
            init_interface = normalizers["init_interface"](init_interface.to(device))
            final_solid = normalizers["final_solid"](final_solid.to(device))
            final_fluid = normalizers["final_fluid"](final_fluid.to(device))
            final_interface = normalizers["final_interface"](final_interface.to(device))

            pred_solid, pred_fluid, pred_interface = net(
                init_solid, init_fluid, init_interface
            )

            solid_loss = criterion(pred_solid, final_solid, is_train = True)
            fluid_loss = criterion(pred_fluid, final_fluid, is_train = True)
            interface_loss = criterion(pred_interface, final_interface, is_train = True)
            
            loss = (solid_loss + fluid_loss + interface_loss) / 3

            print(
                f"epoch: {epoch},",
                f"lr: {scheduler.get_last_lr()[0]},",
                f"batch: {i},",
                f"train_loss: {loss.data},",
                f"train_solid_loss: {solid_loss.data},",
                f"train_fluid_loss: {fluid_loss.data},",
                f"train_interface_loss: {interface_loss.data}"
            )
            
            mean_train_losses["loss"] += loss.data
            mean_train_losses["solid_loss"] += solid_loss.data
            mean_train_losses["fluid_loss"] += fluid_loss.data
            mean_train_losses["interface_loss"] += interface_loss.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        mean_train_losses["loss"] /= len(train_datas)
        mean_train_losses["solid_loss"] /= len(train_datas)
        mean_train_losses["fluid_loss"] /= len(train_datas)
        mean_train_losses["interface_loss"] /= len(train_datas)

        print(
            f"epoch: {epoch},",
            f"mean_train_loss: {mean_train_losses['loss']},",
            f"mean_train_solid_loss: {mean_train_losses['solid_loss']},", 
            f"mean_train_fluid_loss: {mean_train_losses['fluid_loss']},", 
            f"mean_train_interface_loss: {mean_train_losses['interface_loss']}"
        )
        if (epoch + 1) % config["flexible_wing"]["hyperparameter"]["eval_epoch_gap"] != 0:
            continue
        
        net.eval()

        mean_eval_losses = {
            "loss": 0.,
            "solid_loss": 0.,
            "fluid_loss": 0.,
            "interface_loss": 0.
        }

        with torch.no_grad():
            for i, eval_data in enumerate(eval_datas):
                init_solid, init_fluid, init_interface, \
                final_solid, final_fluid, final_interface = eval_data

                init_solid = normalizers["init_solid"](init_solid.to(device))
                init_fluid = normalizers["init_fluid"](init_fluid.to(device))
                init_interface = normalizers["init_interface"](init_interface.to(device))

                final_solid = final_solid.to(device)
                final_fluid = final_fluid.to(device)
                final_interface = final_interface.to(device)

                pred_solid, pred_fluid, pred_interface = net(
                    init_solid, init_fluid, init_interface
                )

                pred_solid = normalizers["final_solid"].inverse(pred_solid)
                pred_fluid = normalizers["final_fluid"].inverse(pred_fluid)
                pred_interface = normalizers["final_interface"].inverse(pred_interface)

                solid_loss = criterion(pred_solid, final_solid)
                fluid_loss = criterion(pred_fluid, final_fluid)
                interface_loss = criterion(pred_interface, final_interface)

                loss = (solid_loss + fluid_loss + interface_loss) / 3

                print(
                    f"epoch: {epoch},",
                    f"batch: {i},",
                    f"eval_loss: {loss.data},",
                    f"eval_solid_loss: {solid_loss.data},",
                    f"eval_fluid_loss: {fluid_loss.data},",
                    f"eval_interface_loss: {interface_loss.data}"
                )
                
                mean_eval_losses["loss"] += loss.data
                mean_eval_losses["solid_loss"] += solid_loss.data
                mean_eval_losses["fluid_loss"] += fluid_loss.data
                mean_eval_losses["interface_loss"] += interface_loss.data

            mean_eval_losses["loss"] /= len(eval_datas)
            mean_eval_losses["solid_loss"] /= len(eval_datas)
            mean_eval_losses["fluid_loss"] /= len(eval_datas)
            mean_eval_losses["interface_loss"] /= len(eval_datas)

            print(
                f"epoch: {epoch},",
                f"mean_eval_loss: {mean_eval_losses['loss']},",
                f"mean_eval_solid_loss: {mean_eval_losses['solid_loss']},",
                f"mean_eval_fluid_loss: {mean_eval_losses['fluid_loss']},",
                f"mean_eval_interface_loss: {mean_eval_losses['interface_loss']}"
            )
            
            if mean_eval_losses["loss"] < min_eval_loss:
                min_eval_loss = mean_eval_losses["loss"]
                torch.save(
                    net.state_dict(), 
                    f"{assets_root}/checkpoints/{model_name}_best_model_{version}.pth"
                )


def test(data_path, net, config, normalizers, criterion):
    ret_save_path = f"{data_root}/ret/{model_name}_flexible_wing_{version}.pkl"
    batch_size = config["flexible_wing"]["hyperparameter"]["batch_size"]

    test_data_loader = DataLoader_Flexible_Wing(
        data_path = data_path,
        mode = "test"
    )

    test_datas = DataLoader(
        test_data_loader,
        batch_size = batch_size,
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
        "solid_loss": 0.,
        "fluid_loss": 0.,
        "interface_loss": 0.
    }

    with torch.no_grad():
        ret_save_data = {
            "input_state": {
                "solid": [],
                "fluid": [],
                "interface": []    
            }, 
            "pred_state": {
                "solid": [],
                "fluid": [],
                "interface": []    
            },
            "ground_truth": {
                "solid": [],
                "fluid": [],
                "interface": []    
            }
        }
        for i, test_data in enumerate(test_datas):
            init_solid, init_fluid, init_interface, \
            final_solid, final_fluid, final_interface = test_data

            ret_save_data["input_state"]["solid"].append(init_solid.cpu().numpy())
            ret_save_data["input_state"]["fluid"].append(init_fluid.cpu().numpy())
            ret_save_data["input_state"]["interface"].append(init_interface.cpu().numpy())

            ret_save_data["ground_truth"]["solid"].append(final_solid.cpu().numpy())
            ret_save_data["ground_truth"]["fluid"].append(final_fluid.cpu().numpy())
            ret_save_data["ground_truth"]["interface"].append(final_interface.cpu().numpy())

            init_solid = normalizers["init_solid"](init_solid.to(device))
            init_fluid = normalizers["init_fluid"](init_fluid.to(device))
            init_interface = normalizers["init_interface"](init_interface.to(device))

            final_solid = final_solid.to(device)
            final_fluid = final_fluid.to(device)
            final_interface = final_interface.to(device)

            pred_solid, pred_fluid, pred_interface = net(
                init_solid, init_fluid, init_interface
            )

            pred_solid = normalizers["final_solid"].inverse(pred_solid)
            pred_fluid = normalizers["final_fluid"].inverse(pred_fluid)
            pred_interface = normalizers["final_interface"].inverse(pred_interface)

            solid_loss = criterion(pred_solid, final_solid)
            fluid_loss = criterion(pred_fluid, final_fluid)
            interface_loss = criterion(pred_interface, final_interface)

            loss = (solid_loss + fluid_loss + interface_loss) / 3

            print(
                f"batch: {i},",
                f"test_loss: {loss.data},",
                f"test_solid_loss: {solid_loss.data},",
                f"test_fluid_loss: {fluid_loss.data},",
                f"test_interface_loss: {interface_loss.data}"
            )
            
            mean_test_losses["loss"] += loss.data
            mean_test_losses["solid_loss"] += solid_loss.data
            mean_test_losses["fluid_loss"] += fluid_loss.data
            mean_test_losses["interface_loss"] += interface_loss.data


            ret_save_data["pred_state"]["solid"].append(pred_solid.cpu().numpy())
            ret_save_data["pred_state"]["fluid"].append(pred_fluid.cpu().numpy())
            ret_save_data["pred_state"]["interface"].append(pred_interface.cpu().numpy())

        mean_test_losses["loss"] /= len(test_datas)
        mean_test_losses["solid_loss"] /= len(test_datas)
        mean_test_losses["fluid_loss"] /= len(test_datas)
        mean_test_losses["interface_loss"] /= len(test_datas)

        print(
            f"mean_test_loss: {mean_test_losses['loss']},",
            f"mean_test_solid_loss: {mean_test_losses['solid_loss']},",
            f"mean_test_fluid_loss: {mean_test_losses['fluid_loss']},",
            f"mean_test_interface_loss: {mean_test_losses['interface_loss']}"
        )

        pickle.dump(ret_save_data, open(ret_save_path, "wb"))

        

def main(run_type):
    data_path = f"{data_root}/input"
    fisale_params = config["flexible_wing"]["fisale"]
    net = Fisale(
        dim = fisale_params["dim"],
        input_quantity_dims = fisale_params["input_quantity_dims"],
        output_quantity_dims = fisale_params["output_quantity_dims"],
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

    if config["flexible_wing"]["hyperparameter"]["criterion"] == "relative l2":
        criterion = relative_l2_loss
    elif config["flexible_wing"]["hyperparameter"]["criterion"] == "mse l2":
        criterion = mse_l2_loss
    else:
        raise NotImplementedError

    normalizers = {}
    for s1_1, s1_2 in zip(["init", "final"], ["input", "output"]):
        for s2 in ["solid", "fluid", "interface"]:
            normalizers[f"{s1_1}_{s2}"] = Normalizer(
                size = fisale_params["dim"] + fisale_params[f"{s1_2}_quantity_dims"][s2],
                name = f"{s1_1}_{s2}",
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
    
    version = config["flexible_wing"]["version"]
    assets_root = config["flexible_wing"]["assets_root"]
    data_root = config["flexible_wing"]["data_root"]
    model_name = config["flexible_wing"]["hyperparameter"]["model_name"]
    device = config["flexible_wing"]["hyperparameter"]["device"]
    log_path = f"{assets_root}/logs/{model_name}_log_{version}.log"
    log_writer = open(log_path, 'a', encoding = "utf8")

    sys.stdout = log_writer
    sys.stderr = log_writer

    print(config["flexible_wing"]["description"])

    main("train")
    main("test")

    log_writer.close()