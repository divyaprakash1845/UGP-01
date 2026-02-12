import os
import glob
import shutil
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import subprocess

# Auto-install MNE if missing (for Colab/Kaggle convenience)
try:
    import mne
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mne"])
    import mne

# --- CONSTANTS ---
TARGET_SFREQ = 500
CLIP_LEN_SEC = 1.5
FIXED_LEN = int(TARGET_SFREQ * CLIP_LEN_SEC) # 750 samples
CHANNELS = ['Fz', 'FCz', 'Pz', 'Oz', 'C3', 'C4', 'P3', 'P4', 'ECG1']
NBACK_MAP = {'zeroBACK.set': 0, 'twoBACK.set': 1}
MATB_MAP  = {'MATBeasy.set': 0, 'MATBdiff.set': 1}
def detect_paths():
    """Points to raw_data folder located OUTSIDE the repository folder"""
    # Get the directory of the current script (e.g., /content/UGP-01)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go one level up to /content/
    parent_dir = os.path.dirname(script_dir)
    
    # Define paths
    data_root = os.path.join(parent_dir, 'raw_data')
    output_dir = os.path.join(script_dir, 'processed_data')

    # Verification
    if not os.path.exists(data_root):
        # Fallback for standard Colab root if parent_dir isn't /content
        data_root = '/content/raw_data'
        
    return data_root, output_dir
class MaestroPreprocessor:
    def __init__(self, root_dir, output_dir):
        self.root_dir = root_dir
        self.output_dir = output_dir

    def run(self):
        print(f"🚀 PREPROCESSING: Scanning {self.root_dir}...")
        if os.path.exists(self.output_dir): shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        subjects = glob.glob(os.path.join(self.root_dir, 'sub-*'))
        if not subjects and 'sub-' in os.path.basename(self.root_dir): subjects = [self.root_dir]
        
        for sub in subjects:
            self._process_subject(sub, os.path.basename(sub))

    def _process_subject(self, sub_path, sub_id):
        sessions = glob.glob(os.path.join(sub_path, 'ses-*'))
        for ses in sessions:
            eeg_path = os.path.join(ses, 'eeg')
            if not os.path.exists(eeg_path): continue
            
            ses_id = os.path.basename(ses)
            print(f"   Processing {sub_id}/{ses_id}...")
            self._process_pvt(eeg_path, sub_id, ses_id)
            self._process_flanker(eeg_path, sub_id, ses_id)
            self._process_continuous(eeg_path, sub_id, ses_id, 'NBACK', NBACK_MAP)
            self._process_continuous(eeg_path, sub_id, ses_id, 'MATB', MATB_MAP)

    def _save(self, data, label, task, sub, ses, idx):
        task_dir = os.path.join(self.output_dir, task)
        os.makedirs(task_dir, exist_ok=True)
        tensor = torch.tensor(data, dtype=torch.float32).transpose(0, 1) # [Time, Chan]
        
        # Pad or Truncate to 750
        if tensor.shape[0] != FIXED_LEN:
            if tensor.shape[0] > FIXED_LEN: tensor = tensor[:FIXED_LEN]
            else: tensor = torch.nn.functional.pad(tensor, (0, 0, 0, FIXED_LEN - tensor.shape[0]))
            
        torch.save({'data': tensor, 'label': label}, os.path.join(task_dir, f"{sub}_{ses}_{label}_{idx}.pt"))

    def _process_pvt(self, path, sub, ses):
        try:
            raw = self._load_raw(os.path.join(path, 'PVT.set'))
            if not raw: return
            events, eid = mne.events_from_annotations(raw, verbose=False)
            if '13' in eid and '14' in eid:
                rts = []
                for i in range(len(events)-1):
                    if events[i,2] == eid['13'] and events[i+1,2] == eid['14']:
                        rts.append(events[i+1,0] - events[i,0])
                if rts:
                    med = np.median(rts)
                    epochs = mne.Epochs(raw, events, event_id=eid['13'], tmin=-1.0, tmax=0.5, verbose=False)
                    for i, d in enumerate(epochs.get_data()):
                        label = 0 if rts[i] < med else 1
                        self._save(d, label, 'PVT', sub, ses, i)
        except: pass

    def _process_flanker(self, path, sub, ses):
        try:
            raw = self._load_raw(os.path.join(path, 'Flanker.set'))
            if not raw: return
            events, eid = mne.events_from_annotations(raw, verbose=False)
            mapping = {'2511': 0, '2521': 1}
            for m, l in mapping.items():
                if m in eid:
                    for i, d in enumerate(mne.Epochs(raw, events, eid[m], tmin=-1.0, tmax=0.5, verbose=False).get_data()):
                        self._save(d, l, 'FLANKER', sub, ses, f"{m}_{i}")
        except: pass

    def _process_continuous(self, path, sub, ses, task, fmap):
        for fname, label in fmap.items():
            try:
                raw = self._load_raw(os.path.join(path, fname))
                if not raw: continue
                d = raw.get_data()
                for i in range(d.shape[1] // FIXED_LEN):
                    self._save(d[:, i*FIXED_LEN:(i+1)*FIXED_LEN], label, task, sub, ses, i)
            except: pass

    def _load_raw(self, fpath):
        if not os.path.exists(fpath): return None
        raw = mne.io.read_raw_eeglab(fpath, preload=True, verbose=False)
        if not all(c in raw.ch_names for c in CHANNELS): return None
        raw.pick_channels(CHANNELS)
        if raw.info['sfreq'] != TARGET_SFREQ: raw.resample(TARGET_SFREQ)
        return raw

class MaestroDataset(Dataset):
    def __init__(self, folder): self.files = glob.glob(os.path.join(folder, "*.pt"))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        d = torch.load(self.files[idx], weights_only=False)
        x = d['data']
        # Z-Score Normalization
        return (x - x.mean(0)) / (x.std(0) + 1e-8), d['label']

if __name__ == "__main__":
    i, o = detect_paths()
    MaestroPreprocessor(i, o).run()
