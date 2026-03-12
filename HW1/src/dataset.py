import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS
from tqdm import tqdm


LABEL2ID = {
    "no": 0,
    "yes": 1,
}


class YesNoSpeechCommands(Dataset):
    def __init__(self, root: str, subset: str | None = None, max_length: int = 16000):
        self.dataset = SPEECHCOMMANDS(
            root=root,
            download=True,
            subset=subset
        )
        self.max_length = max_length
        self.items = []

        for waveform, sample_rate, label, speaker_id, utterance_number in tqdm(self.dataset):
            if label in LABEL2ID:
                self.items.append((waveform, LABEL2ID[label]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        waveform, label = self.items[idx]
        waveform = waveform.squeeze(0)

        if waveform.shape[0] < self.max_length:
            waveform = F.pad(waveform, (0, self.max_length - waveform.shape[0]))
        else:
            waveform = waveform[:self.max_length]

        return waveform, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.stack(ys, dim=0)
    return xs, ys