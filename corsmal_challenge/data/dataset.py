import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from corsmal_challenge.data.audio import load_wav
from corsmal_challenge.utils import fix_random_seeds


class AudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_data: Path = Path("./data/train/"),
        path_to_annotation_file: Path = Path("./data/train/ccm_train_annotation.json"),
        seed: int = 0,
        train: bool = True,
        query: str = "type",  # or "level"
        random_crop: bool = False,
        strong_crop: bool = False,
    ):
        fix_random_seeds(seed)
        # behave_deterministically()

        self.seed = seed
        self.type_label = {0: "none", 1: "pasta", 2: "rice", 3: "water"}
        self.level_label = {0: "empty", 1: "half-full", 2: "full"}
        self.annotations = self._get_annotations(path_to_annotation_file)
        self.val_idx, self.train_idx = self._divide_idx()
        self.audio_path: Path = path_to_data / "audio"
        self.train: bool = train
        self.query: str = query
        self.random_crop: bool = random_crop
        self.strong_crop: bool = strong_crop

    def _get_annotations(self, path_to_annotation_file) -> List[Dict[str, int]]:
        with open(str(path_to_annotation_file), "r") as f:
            dic: Dict = json.load(f)
        annotations = [{"type": data["filling type"], "level": data["filling level"]} for data in dic["annotations"]]
        return annotations

    def _get_classified_data(self) -> List[List[List[int]]]:
        classified_data: List[List[List[int]]] = [[[] for j in range(3)] for i in range(4)]
        for idx, data in enumerate(self.annotations):
            filling_type = data["type"]
            filling_level = data["level"]
            classified_data[filling_type][filling_level].append(idx)
        return classified_data

    def _divide_idx(self) -> Tuple[List, List]:
        classified_data: List[List[List[int]]] = self._get_classified_data()
        self.annotations
        val_idx: List[int] = []
        train_idx: List[int] = []
        for d in classified_data:
            for lis in d:
                val_list = set(random.sample(lis, len(lis) // 10 * 2))
                for id in lis:
                    if id == 377:  # anomaly
                        continue
                    if id in val_list:
                        val_idx.append(id)
                    else:
                        train_idx.append(id)
        return val_idx, train_idx

    def __len__(self):
        return len(self.train_idx) if self.train else len(self.val_idx)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        id: int = self.train_idx[idx] if self.train else self.val_idx[idx]
        data_path: Path = self.audio_path / (str(id).zfill(6) + ".wav")
        label = self.annotations[id][self.query]
        spectrogram = load_wav(data_path).generate_mel_spectrogram().log2()

        if self.random_crop:
            sequence_len: int = spectrogram.shape[-1]
            if self.strong_crop:
                start = random.randrange(0, sequence_len // 10 * 4)
                end = random.randrange(sequence_len // 10 * 6, sequence_len)
            else:
                start = random.randrange(0, sequence_len // 10 * 2)
                end = random.randrange(sequence_len // 10 * 8, sequence_len)
            return spectrogram[:, :, start : end + 1].transpose(-1, -2), label

        return spectrogram.transpose(-1, -2), label

    def generate_extracted_train_dataset(
        self, type_annotation: Optional[int] = 3, level_annotation: Optional[int] = 2
    ) -> torch.utils.data.Dataset:
        idx = [
            i
            for i in self.train_idx
            if (type_annotation is None or self.annotations[i]["type"] == type_annotation)
            and (level_annotation is None or self.annotations[i]["level"] == level_annotation)
        ]

        class ExtractedAudioDataset(torch.utils.data.Dataset):
            def __init__(
                self,
                path_to_audio: Path = Path("./data/train/audio"),
                seed: int = 0,
                idx: List[int] = [],
                random_crop: bool = False,
                strong_crop: bool = False,
            ):
                super(ExtractedAudioDataset, self).__init__()
                self.audio_path: Path = path_to_audio
                self.idx = idx
                self.seed = seed
                self.random_crop: bool = random_crop
                self.strong_crop: bool = strong_crop

            def __len__(self):
                return len(self.idx)

            def __getitem__(self, idx) -> torch.Tensor:
                id: int = self.idx[idx]
                data_path: Path = self.audio_path / (str(id).zfill(6) + ".wav")
                spectrogram = load_wav(data_path).generate_mel_spectrogram().log2()

                if self.random_crop:
                    sequence_len: int = spectrogram.shape[-1]
                    if self.strong_crop:
                        start = random.randrange(0, sequence_len // 10 * 4)
                        end = random.randrange(sequence_len // 10 * 6, sequence_len)
                    else:
                        start = random.randrange(0, sequence_len // 10 * 2)
                        end = random.randrange(sequence_len // 10 * 8, sequence_len)
                    return spectrogram[:, :, start : end + 1].transpose(-1, -2)

                return spectrogram.transpose(-1, -2)

        return ExtractedAudioDataset(self.audio_path, self.seed, idx, self.random_crop, self.strong_crop)
