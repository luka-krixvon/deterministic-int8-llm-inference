"""Work-order item 5: extract, from the ACTUAL pinned vLLM environment, the
sources deciding whether both linear-kernel arms share the same activation
quantization op. Saved verbatim as an artifact with versions and digest."""
import inspect, json, os
import vllm
out = {"vllm_version": vllm.__version__,
       "image_digest": os.environ.get("IMAGE_DIGEST", "unset"), "sources": {}}
mods = {}
import importlib, pkgutil
for m in pkgutil.walk_packages(vllm.__path__, "vllm."):
    n = m.name
    if ("scaled_mm" in n and ("cutlass" in n or "triton" in n or "kernels" in n)) or n.endswith("scaled_mm"):
        try: mods[n] = importlib.import_module(n)
        except Exception: pass
targets = []
for n, mod in mods.items():
    for cls_name in dir(mod):
        if "Int8ScaledMM" in cls_name or "ScaledMMLinearKernel" in cls_name:
            targets.append((n, cls_name, getattr(mod, cls_name)))
for n, cls_name, cls in targets:
    try:
        src = inspect.getsource(cls)
        out["sources"][f"{n}.{cls_name}"] = {
            "file": inspect.getsourcefile(cls),
            "calls_scaled_int8_quant": "scaled_int8_quant" in src,
            "source": src}
    except Exception as e:
        out["sources"][f"{n}.{cls_name}"] = {"error": str(e)[:100]}
json.dump(out, open("/w/m6_sources.json", "w"), indent=2)
shared = [k for k, v in out["sources"].items()
          if isinstance(v, dict) and v.get("calls_scaled_int8_quant")]
print("classes calling scaled_int8_quant:", shared)
print("M6_EXTRACT_OK")
