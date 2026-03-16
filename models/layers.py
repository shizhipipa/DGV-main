import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


th.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size
    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)
    size = size + 1 if size % 2 != 0 else size
    return int(size * last_layer["out_channels"])


def encode_input(text, tokenizer):
    max_length = 512
    if isinstance(text, list):
        encoded = tokenizer(text, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    else:
        encoded = tokenizer([text], max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    return encoded.input_ids, encoded.attention_mask


class Conv(nn.Module):
    """The structural scoring head used after graph propagation."""

    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
        super().__init__()
        self.conv1d_1_args = conv1d_1
        self.conv1d_1 = nn.Conv1d(**conv1d_1)
        self.conv1d_2 = nn.Conv1d(**conv1d_2)
        fc1_size = get_conv_mp_out_size(fc_1_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        fc2_size = get_conv_mp_out_size(fc_2_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        self.fc1 = nn.Linear(fc1_size, 1)
        self.fc2 = nn.Linear(fc2_size, 1)
        self.drop = nn.Dropout(p=0.2)
        self.mp_1 = nn.MaxPool1d(**maxpool1d_1)
        self.mp_2 = nn.MaxPool1d(**maxpool1d_2)

    def forward(self, hidden, x):
        concat = torch.cat([hidden, x], 1)
        concat_size = hidden.shape[1] + x.shape[1]
        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)

        z = self.mp_1(F.leaky_relu(self.conv1d_1(concat)))
        z = self.mp_2(self.conv1d_2(z))

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])
        y = self.mp_1(F.leaky_relu(self.conv1d_1(hidden)))
        y = self.mp_2(self.conv1d_2(y))

        z = z.view(-1, int(z.shape[1] * z.shape[-1]))
        y = y.view(-1, int(y.shape[1] * y.shape[-1]))
        output_fc1 = self.fc1(z)
        output_fc2 = self.fc2(y)
        res = output_fc1 * output_fc2
        res = self.drop(res)
        return F.softmax(res, dim=1)
