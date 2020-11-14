import os
import h5py
import numpy as np
import random


class HDF5DatasetWriter:
    def __init__(self, audio_dims, output_path, batch_size=128, buffer_size=10000):
        if os.path.exists(output_path):
            raise ValueError(output_path)
        self.db = h5py.File(output_path, "w")
        self.origins = self.db.create_dataset("origins", audio_dims, dtype="float", compression="gzip",
                                              chunks=(batch_size, audio_dims[1], audio_dims[2]))
        self.positives = self.db.create_dataset("positives", audio_dims, dtype="float", compression="gzip",
                                                chunks=(batch_size, audio_dims[1], audio_dims[2]))
        self.negatives = self.db.create_dataset("negatives", audio_dims, dtype="float", compression="gzip",
                                                chunks=(batch_size, audio_dims[1], audio_dims[2]))
        self.positives_label = self.db.create_dataset("positives_label", (audio_dims[0], 1), dtype="float", compression="gzip",
                                                chunks=(batch_size, 1))
        self.negatives_label = self.db.create_dataset("negatives_label", (audio_dims[0], 1), dtype="float", compression="gzip",
                                                chunks=(batch_size, 1))

        self.bufSize = buffer_size
        self.buffer = {"origins": [], "positives": [], "negatives": [], "positives_label": [], "negatives_label":[]}
        self.idx = 0

    def add(self, origins, positives, negatives, positives_label, negatives_label):
        self.buffer["origins"].extend(origins)
        self.buffer["positives"].extend(positives)
        self.buffer["negatives"].extend(negatives)
        self.buffer["positives_label"].extend(positives_label)
        self.buffer["negatives_label"].extend(negatives_label)
        if len(self.buffer["origins"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["origins"])
        self.origins[self.idx:i] = self.buffer["origins"]
        self.positives[self.idx:i] = self.buffer["positives"]
        self.negatives[self.idx:i] = self.buffer["negatives"]
        self.positives_label[self.idx:i] = self.buffer["positives_label"]
        self.negatives_label[self.idx:i] = self.buffer["negatives_label"]
        self.idx = i
        self.buffer = {"origins": [], "positives": [], "negatives": [], "positives_label": [], "negatives_label":[]}

    def close(self):
        if len(self.buffer["origins"]) > 0:
            self.flush()
        self.db.close()


class HDF5DatasetGenerator:
    def __init__(self, db_path, batch_size, number=None):
        self.batchSize = batch_size
        self.db = h5py.File(db_path)
        self.numAudios = self.db["origins"].shape[0]
        self.indexes = []
        for i in range(self.numAudios):
            self.indexes.append(i)
        if number is not None:
            self.indexes = self.indexes[: number]

    def get_total_samples(self):
        return len(self.indexes)

    def generator(self):
        random.shuffle(self.indexes)
        for i in range(0,  len(self.indexes), self.batchSize):
            origins = self.db["origins"][self.indexes[i]: self.indexes[i] + self.batchSize]
            positives = self.db["positives"][self.indexes[i]: self.indexes[i] + self.batchSize]
            negatives = self.db["negatives"][self.indexes[i]: self.indexes[i] + self.batchSize]
            positives_label = self.db["positives_label"][self.indexes[i]: self.indexes[i] + self.batchSize]
            negatives_label = self.db["negatives_label"][self.indexes[i]: self.indexes[i] + self.batchSize]
            yield np.array(origins), np.array(positives), np.array(negatives), np.array(positives_label), np.array(negatives_label)

    def close(self):
        self.db.close()