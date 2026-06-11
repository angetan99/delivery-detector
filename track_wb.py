import wandb

wandb.init(
    project="delivery-detector",
    config={
        "model": "ResNet50",
        "epochs": 25,
        "batch_size": 4,
        "learning_rate": 0.001,
        "optimizer": "SGD"
    }
)

with open("results.txt", "r") as f:
    lines = f.readlines()

i = 0
while i < len(lines):
    line = lines[i].strip()
    if line.startswith("Epoch"):
        train_line = lines[i+2].strip()
        val_line = lines[i+3].strip()

        train_loss = float(train_line.split()[2])
        train_acc = float(train_line.split()[4])
        val_loss = float(val_line.split()[2])
        val_acc = float(val_line.split()[4])

        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
    i += 1

wandb.finish()
print("Done")