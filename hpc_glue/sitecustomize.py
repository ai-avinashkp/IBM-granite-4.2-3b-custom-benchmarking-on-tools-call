"""MIG fix for vLLM 0.8.5, loaded automatically by Python for processes that have this folder on PYTHONPATH.

Only our `vllm serve` command gets this folder on PYTHONPATH (hpc/ibm_run.py, action `eval`).

Problem: the HPC wrapper selects a MIG slice with CUDA_VISIBLE_DEVICES=MIG-<uuid>. vLLM's NVML code turns
that value into a GPU index with int(...) and crashes ("invalid literal for int() ... 'MIG-...'").
Fix: map a MIG UUID to the NVML index of its parent GPU. NVML is used only for capability, name and
topology; memory sizing goes through CUDA (torch.cuda.mem_get_info), which respects the MIG slice.
"""
import os

if os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith("MIG-"):
    try:
        import vllm.platforms.cuda as _vc
        from vllm.utils import import_pynvml

        _original = _vc.device_id_to_physical_device_id

        def _mig_aware(device_id):
            ids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            uid = ids[device_id]
            if not uid.startswith("MIG-"):
                return _original(device_id)
            nv = import_pynvml()
            nv.nvmlInit()
            try:
                try:
                    mig = nv.nvmlDeviceGetHandleByUUID(uid)
                except (TypeError, nv.NVMLError):       # some pynvml versions want bytes
                    mig = nv.nvmlDeviceGetHandleByUUID(uid.encode())
                parent = nv.nvmlDeviceGetDeviceHandleFromMigDeviceHandle(mig)
                return nv.nvmlDeviceGetIndex(parent)
            finally:
                nv.nvmlShutdown()

        _vc.device_id_to_physical_device_id = _mig_aware
    except Exception as e:                      # never break interpreter start-up; vLLM will then fail visibly
        print(f"[sitecustomize] MIG patch not applied: {e!r}", flush=True)
