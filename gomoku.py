from connection import get_player
import torch

assert torch.cuda.is_available()

# This is the 0.2 second version. It uses the smaller-test-checkpoints
move = get_player("./model_small.pth.tar", "r", 6, 512, 0.19, 0.3)

# This is the 2 second version. It uses the very-deep-checkpoints
# move = get_player("./model_deep.pth.tar", "r", 19, 512, 1.99, 0)

if __name__ == "__main__":
    print("Ran")