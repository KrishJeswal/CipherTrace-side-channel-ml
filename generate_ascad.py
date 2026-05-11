import h5py
import numpy as np

# ANSSI standard parameters for ASCAD generation
INPUT_FILE    = "artifacts/raw/ASCAD.h5"
OUTPUT_FILE   = "artifacts/raw/ASCAD_processed.h5"
FIRST_SAMPLE  = 45400   # start of SubBytes window for byte 2
TRACE_LENGTH  = 700     # samples to keep
N_PROFILING   = 50000
N_ATTACK      = 10000

print("Loading raw traces file...")
print("WARNING: This loads 60,000 x 100,000 traces — may take a few minutes")

with h5py.File(INPUT_FILE, "r") as in_f:
    # Window the traces: take only samples 45400 to 46100
    print("Windowing traces to samples", FIRST_SAMPLE, "to", FIRST_SAMPLE + TRACE_LENGTH)
    traces_windowed = in_f["traces"][:, FIRST_SAMPLE:FIRST_SAMPLE + TRACE_LENGTH]
    # shape: (60000, 700)

    metadata = in_f["metadata"][:]
    # fields: plaintext, ciphertext, key, masks

print("Traces windowed. Shape:", traces_windowed.shape)

# Split into profiling and attack
X_prof  = traces_windowed[:N_PROFILING]           # (50000, 700)
X_atk   = traces_windowed[N_PROFILING:]           # (10000, 700)

pt_prof  = np.array([m["plaintext"] for m in metadata[:N_PROFILING]])   # (50000, 16)
pt_atk   = np.array([m["plaintext"] for m in metadata[N_PROFILING:]])   # (10000, 16)
key_prof = np.array([m["key"]       for m in metadata[:N_PROFILING]])   # (50000, 16)
key_atk  = np.array([m["key"]       for m in metadata[N_PROFILING:]])   # (10000, 16)
msk_prof = np.array([m["masks"]     for m in metadata[:N_PROFILING]])   # (50000, 16)
msk_atk  = np.array([m["masks"]     for m in metadata[N_PROFILING:]])   # (10000, 16)

print("Saving processed file to", OUTPUT_FILE)

with h5py.File(OUTPUT_FILE, "w") as out_f:
    # Profiling group
    prof_grp  = out_f.create_group("Profiling_traces")
    prof_grp.create_dataset("traces", data=X_prof)
    prof_meta = prof_grp.create_group("metadata")
    prof_meta.create_dataset("plaintext", data=pt_prof)
    prof_meta.create_dataset("key",       data=key_prof)
    prof_meta.create_dataset("masks",     data=msk_prof)

    # Attack group
    atk_grp  = out_f.create_group("Attack_traces")
    atk_grp.create_dataset("traces", data=X_atk)
    atk_meta = atk_grp.create_group("metadata")
    atk_meta.create_dataset("plaintext", data=pt_atk)
    atk_meta.create_dataset("key",       data=key_atk)
    atk_meta.create_dataset("masks",     data=msk_atk)

print("Done.")
print("Profiling traces shape:", X_prof.shape)
print("Attack traces shape:   ", X_atk.shape)
print("Key byte at index 2:   ", hex(key_prof[0, 2]))