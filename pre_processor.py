from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict


class LineSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: int
    event: int
    device: str
    channel: str
    code: int
    size: int
    data: npt.NDArray

    @classmethod
    def from_raw_line(cls, line: str):
        values = line.split()

        id = values[0]
        event = values[1]
        device = values[2]
        channel = values[3]
        code = values[4]
        size = values[5]

        data = np.fromstring(values[6], sep=",", dtype=np.float32)

        return cls(
            id=id,
            event=event,
            device=device,
            channel=channel,
            code=code,
            size=size,
            data=data,
        )


class PreProcessorDataset:
    def __init__(self, data: npt.NDArray, metadata: pd.DataFrame) -> None:
        self.data = data
        self.metadata = metadata

    @classmethod
    def from_txt_file(cls: Self, filepath: Path, padded_length=None):
        parsed_lines = []
        with open(filepath) as f:
            for line in f:
                parsed_line = LineSchema.from_raw_line(line)
                parsed_lines.append(parsed_line)

        df = pd.DataFrame([k.model_dump(exclude="data") for k in parsed_lines]).assign(
            device=lambda x: x["device"].astype("category"),
            channel=lambda x: x["channel"].astype("category"),
        )

        max_len = df["size"].max()

        if padded_length is not None:
            if padded_length < max_len:
                raise ValueError(
                    f"{padded_length=} is shorter than longest signal array {max_len=}"
                )
            max_len = padded_length

        arrs = []
        for idx, rec in enumerate(parsed_lines):
            tmp_arr = np.zeros((max_len,), dtype=np.float32)

            tmp_arr[0 : len(rec.data)] = rec.data
            arrs.append(tmp_arr)

        data_array = np.stack(arrs)

        return cls(data=data_array, metadata=df)

    def __getitem__(self, index):
        return self.metadata.iloc[index, :], self.data[index]

    @classmethod
    def load_from_disk(cls, filepath_data: Path, filepath_metadata: Path) -> Self:
        df = pd.read_parquet(filepath_metadata, engine="pyarrow")
        data_array = np.load(filepath_data)

        return cls(data=data_array, metadata=df)

    def save_to_disk(self, filename_prefix, directory=Path.cwd()):
        self.metadata.to_parquet(
            directory / f"{filename_prefix}_eeg_ppd_metadata.parquet", engine="pyarrow"
        )

        with open(directory / f"{filename_prefix}_eeg_ppd_data.npy", mode="wb") as f:
            np.save(f, self.data)

    def enumerate_events(self):
        events = self.metadata["event"].unique()

        for event in events:
            idxes = self.metadata[self.metadata["event"] == event].index
            yield self.metadata.loc[idxes, :], self.data[idxes]
